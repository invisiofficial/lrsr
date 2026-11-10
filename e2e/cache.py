import gc
import re
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer


#region Configuration

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
CALIB_DATA_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "logits" / "data" / "c4.test.raw"

CACHEABLE_PARTITIONS = {"lrsr-1dos", "lrsr-kmeans"}
AWQ_METHODS = {"awq-per-channel", "awq-per-group", "awq-lrsr-naive"}
GPTQ_METHODS = {"gptq-per-channel", "gptq-per-group", "gptq-lrsr-naive"}
CALIBRATED_METHODS = AWQ_METHODS | GPTQ_METHODS

BLOCK_SIZE = 128

#endregion

#region Helpers

def _safe_model_name(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", model_id.strip("/"))

def resolve_model_id(module: nn.Module, explicit_id: Optional[str] = None) -> Optional[str]:
    if explicit_id:
        return explicit_id
    config = getattr(module, "config", None)
    name_or_path = getattr(config, "_name_or_path", None)
    if name_or_path and isinstance(name_or_path, str):
        return name_or_path
    return None

#endregion

class PartitionCache:
    """Disk cache for bit-width-independent layer partitions (1DOS masks and K-Means labels)."""

    def __init__(
        self,
        module: nn.Module,
        method: str,
        cache_dir: Path = CACHE_DIR,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.model_id = resolve_model_id(module, kwargs.get("model_id"))
        self.enabled = enabled and (method in CACHEABLE_PARTITIONS) and bool(self.model_id)
        self.dirty = False
        self.data: Dict[str, torch.Tensor] = {}
        self.path: Optional[Path] = None

        if self.enabled:
            self.path = self._build_path(cache_dir, self.model_id, method, kwargs)
            if self.path.is_file():
                self.data = torch.load(self.path, map_location="cpu", weights_only=True)

    @staticmethod
    def _build_path(cache_dir: Path, model_id: str, method: str, kwargs: Dict[str, Any]) -> Path:
        safe_model = _safe_model_name(model_id)
        parts = [safe_model, method]

        if "kmeans" in method:
            clusters = kwargs.get("clusters", 3)
            tolerance = kwargs.get("tolerance", 0.05)
            parts.append(f"c{clusters}_tol{tolerance}")

        filename = "__".join(parts) + ".pt"
        return cache_dir / filename

    def get(self, layer_name: str) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        return self.data.get(layer_name)

    def put(self, layer_name: str, partition: torch.Tensor) -> None:
        if not self.enabled:
            return
        self.data[layer_name] = partition.detach().cpu()
        self.dirty = True

    def save(self) -> None:
        if not self.enabled or not self.dirty or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".pt.tmp")
        torch.save(self.data, tmp_path)
        tmp_path.replace(self.path)
        self.dirty = False


class CalibrationCache:
    """Disk cache for AWQ activation statistics and Block-GPTQ (g=128) Cholesky factors."""

    def __init__(
        self,
        module: nn.Module,
        method: str,
        targets: List[Tuple[nn.Module, str, str, nn.Linear]],
        cache_dir: Path = CACHE_DIR,
        enabled: bool = True,
        calib_chunks: int = 32,
        calib_seq_len: int = 512,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.model_id = resolve_model_id(module, kwargs.get("model_id"))
        self.enabled = enabled and (method in CALIBRATED_METHODS) and bool(self.model_id)
        self.data: Dict[str, Any] = {}

        if not self.enabled:
            return

        safe_model = _safe_model_name(self.model_id)
        awq_path = cache_dir / f"{safe_model}__calib-awq.pt"
        gptq_path = cache_dir / f"{safe_model}__calib-gptq-b{BLOCK_SIZE}.pt"
        target_path = awq_path if method in AWQ_METHODS else gptq_path

        if not target_path.is_file():
            self._run_calibration(
                module, targets, awq_path, gptq_path, calib_chunks, calib_seq_len
            )

        self.data = torch.load(target_path, map_location="cpu", weights_only=True)

    def get(self, layer_name: str) -> Optional[Any]:
        if not self.enabled:
            return None
        return self.data.get(layer_name)

    def _run_calibration(
        self,
        module: nn.Module,
        targets: List[Tuple[nn.Module, str, str, nn.Linear]],
        awq_path: Path,
        gptq_path: Path,
        calib_chunks: int,
        calib_seq_len: int,
    ) -> None:
        if not CALIB_DATA_PATH.is_file():
            raise FileNotFoundError(
                f"Calibration data not found at {CALIB_DATA_PATH}. Run benchmarks.logits.setup first."
            )

        # Read disjoint tail of C4 to avoid overlap with evaluation chunks
        with open(CALIB_DATA_PATH, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400_000))
            tail_text = f.read().decode("utf-8", errors="ignore")

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        tokens = tokenizer(tail_text, return_tensors="pt").input_ids.squeeze()
        needed = calib_chunks * calib_seq_len
        if tokens.numel() < needed:
            calib_chunks = max(1, tokens.numel() // calib_seq_len)
        input_ids = tokens[-calib_chunks * calib_seq_len :].view(calib_chunks, calib_seq_len)

        # Group sibling layers sharing identical input activations (q/k/v and gate/up)
        groups: List[List[Tuple[nn.Module, str, str, nn.Linear]]] = []
        group_map: Dict[Tuple[int, str, int], int] = {}

        for item in targets:
            parent, name, full_name, child = item
            k = child.in_features
            if any(tag in name for tag in ("q_proj", "k_proj", "v_proj")):
                g_key = (id(parent), "qkv", k)
            elif any(tag in name for tag in ("gate_proj", "up_proj")):
                g_key = (id(parent), "mlp_up", k)
            else:
                g_key = (id(child), full_name, k)

            if g_key not in group_map:
                group_map[g_key] = len(groups)
                groups.append([item])
            else:
                groups[group_map[g_key]].append(item)

        sum_abs: Dict[int, torch.Tensor] = {}
        h_blocks: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}
        block_sizes: Dict[int, int] = {}

        for g_idx, grp in enumerate(groups):
            k = grp[0][3].in_features
            bsz = math.gcd(k, BLOCK_SIZE)
            block_sizes[g_idx] = bsz
            sum_abs[g_idx] = torch.zeros(k, dtype=torch.float64)
            h_blocks[g_idx] = torch.zeros((k // bsz, bsz, bsz), dtype=torch.float64)
            counts[g_idx] = 0

        def make_hook(idx: int):
            def hook(_mod: nn.Module, args: tuple) -> None:
                x = args[0].detach()
                if x.numel() == 0:
                    return
                k = x.size(-1)
                bsz = block_sizes[idx]
                x_flat = x.reshape(-1, k).float()
                sum_abs[idx] += x_flat.abs().sum(dim=0).double().cpu()
                xb = x_flat.view(-1, k // bsz, bsz).permute(1, 2, 0)
                hb = torch.bmm(xb, xb.transpose(1, 2)).double().cpu()
                h_blocks[idx] += hb
                counts[idx] += x_flat.size(0)
            return hook

        handles = [
             grp[0][3].register_forward_pre_hook(make_hook(g_idx))
            for g_idx, grp in enumerate(groups)
        ]

        use_cuda = torch.cuda.is_available()
        orig_device = next(module.parameters()).device
        dispatched = False
        if use_cuda:
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if use_cuda and orig_device.type == "cpu":
            try:
                from accelerate import infer_auto_device_map, dispatch_model
                no_split = list(getattr(module, "_no_split_modules", []) or [])
                max_mem = {0: "10.5GiB", "cpu": "32GiB"}
                if torch.cuda.device_count() > 1:
                    max_mem[1] = "4GiB"
                device_map = infer_auto_device_map(module, max_memory=max_mem, no_split_module_classes=no_split)
                dispatch_model(module, device_map=device_map)
                dispatched = True
            except Exception:
                module.to("cuda:0")

        exec_dev = next(module.parameters()).device
        try:
            with torch.inference_mode():
                for i in tqdm(range(input_ids.size(0)), desc="  Calibrating (AWQ/GPTQ)", leave=False):
                    module(input_ids[i].unsqueeze(0).to(exec_dev), use_cache=False)
        finally:
            for h in handles:
                h.remove()
            if dispatched:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(module)
            if orig_device.type == "cpu":
                module.to("cpu")
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()

        awq_data: Dict[str, Dict[str, torch.Tensor]] = {}
        gptq_data: Dict[str, torch.Tensor] = {}

        for g_idx, grp in enumerate(groups):
            n_tok = max(1, counts[g_idx])
            bsz = block_sizes[g_idx]

            act_mean = (sum_abs[g_idx] / n_tok).float().clamp(min=1e-7)
            hb = (2.0 / n_tok) * h_blocks[g_idx]
            diag_h = torch.diagonal(hb, dim1=-2, dim2=-1)
            act_rms = (diag_h / 2.0).reshape(-1).clamp(min=1e-14).sqrt().float()

            # Compute damped Cholesky inverse factor on CPU in float64 (zero GPU memory)
            eye = torch.eye(bsz, dtype=torch.float64).unsqueeze(0)
            dead = diag_h == 0
            if dead.any():
                hb = hb.clone()
                hb.diagonal(dim1=-2, dim2=-1)[dead] = 1.0
                diag_h = torch.diagonal(hb, dim1=-2, dim2=-1)

            mean_diag = diag_h.mean(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-6)
            u_norm = eye.expand_as(hb).half()
            for mult in (0.01, 0.05, 0.1, 0.25):
                try:
                    hb_damped = hb + (mult * mean_diag) * eye
                    L = torch.linalg.cholesky(hb_damped)
                    h_inv = torch.cholesky_inverse(L)
                    U = torch.linalg.cholesky(h_inv, upper=True)
                    u_diag = torch.diagonal(U, dim1=-2, dim2=-1).unsqueeze(-1).clamp(min=1e-7)
                    u_norm = (U / u_diag).half()
                    break
                except RuntimeError:
                    continue

            for _, _, full_name, _ in grp:
                awq_data[full_name] = {"act_mean": act_mean, "act_rms": act_rms}
                gptq_data[full_name] = u_norm

        awq_path.parent.mkdir(parents=True, exist_ok=True)
        for path, payload in ((awq_path, awq_data), (gptq_path, gptq_data)):
            tmp = path.with_suffix(".pt.tmp")
            torch.save(payload, tmp)
            tmp.replace(path)
