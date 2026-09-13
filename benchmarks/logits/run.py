import math
import json
import torch
import argparse
import hashlib
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2e.patcher import replace_layers


#region Configuration

DIR = Path(__file__).resolve().parent / "data"
RESULT = Path(__file__).resolve().parent / "result" / "results.json"
DATASET_PATH = DIR / "wiki.test.raw"

DEVICE = "cpu"  # "cpu" | "cuda" | "auto"

MODELS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
METHODS = ["per-channel", "per-group", "lrsr-naive", "lrsr-1dos", "lrsr-kmeans"]
BITS = [4, 8]

CHUNKS = 64
BATCHES = 512

GROUP = 128
CLUSTERS = 3
TOLERANCE = 0.05

TARGET_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

GB = 1024 ** 3
MB = 1024 ** 2
KB = 1024

#endregion

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, streamed in chunks."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def get_size(model_id: str) -> Optional[int]:
    """Download size of the model weights on the Hub."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().repo_info(model_id, files_metadata=True)
    except Exception:
        return None
    total = sum(
        s.size for s in info.siblings
        if s.rfilename.endswith((".safetensors", ".bin")) and s.size
    )
    return total or None

def format_bytes(n: float) -> str:
    """Human-readable byte count."""
    if n >= GB:
        return f"{n / GB:.2f} GB"
    if n >= MB:
        return f"{n / MB:.1f} MB"
    return f"{n / KB:.1f} KB"

def show_plan(
    device: str,
    dataset_hash: str,
    models: List[str],
    bits: List[int],
    methods: List[str],
    chunks: int,
    batch_size: int,
    sizes: Dict[str, Optional[int]],
) -> None:
    """Prints the benchmark plan."""
    total = sum(s or 0 for s in sizes.values())
    width = max(len(m) for m in models)
    print(f"\nLogits benchmark plan")
    print(f"  Dataset:          {DATASET_PATH.name} (SHA256: {dataset_hash[:16]}...)")
    print(f"  Results file:     {RESULT}")
    print(f"  Computing device: {device}")
    print(f"  Target models:    {len(models)}")
    for model_id in models:
        size = sizes.get(model_id)
        print(f"   - {model_id:<{width}}   {format_bytes(size) if size else 'unknown'}")
    if total:
        print(f"  Total download:   ~{format_bytes(total)}")
    else:
        print(f"  Total download:   unknown")
    print(f"  Evaluation:       {chunks} chunks x {batch_size} tokens")
    print(f"  Quantization:")
    print(f"   - Bits: {bits}")
    print(f"   - Methods: {methods}")

#region Helpers

def compute_metrics(
    logits_base: torch.Tensor,
    logits_quant: torch.Tensor,
    labels: torch.Tensor
) -> dict:
    # Shift for next-token prediction
    shift_logits_base = logits_base[..., :-1, :].contiguous().view(-1, logits_base.size(-1))
    shift_logits_quant = logits_quant[..., :-1, :].contiguous().view(-1, logits_quant.size(-1))
    shift_labels = labels[..., 1:].contiguous().view(-1)

    # Reject first half of chunk
    half = shift_logits_base.size(0) // 2
    base_valid = shift_logits_base[half:]
    quant_valid = shift_logits_quant[half:]
    labels_valid = shift_labels[half:]

    # Filter out padding
    valid_mask = labels_valid != -100
    base_valid = base_valid[valid_mask]
    quant_valid = quant_valid[valid_mask]
    labels_valid = labels_valid[valid_mask]

    # Apply stable log-probs
    log_p = F.log_softmax(base_valid.float(), dim=-1)
    p = torch.exp(log_p)
    log_q = F.log_softmax(quant_valid.float(), dim=-1)

    # Compute NLL
    nll_quant = -log_q.gather(dim=-1, index=labels_valid.unsqueeze(-1)).squeeze(-1)
    nll_base  = -log_p.gather(dim=-1, index=labels_valid.unsqueeze(-1)).squeeze(-1)
    mean_nll       = nll_quant.mean().item()
    mean_nll_base  = nll_base.mean().item()

    # Compute Probability Difference
    p_quant = torch.exp(-nll_quant)
    p_base  = torch.exp(-nll_base)
    mean_delta_p = (p_quant - p_base).mean().item() * 100.0

    # Compute KL Divergence
    mask_kld = log_p > -16.0
    kld_per_token = torch.sum(torch.where(mask_kld, p * (log_p - log_q), torch.zeros_like(p)), dim=-1)
    mean_kld = kld_per_token.mean().item()

    # Compute Same Top 1
    imax_base = torch.argmax(base_valid, dim=-1)
    imax_quant = torch.argmax(quant_valid, dim=-1)
    same_top_1 = (imax_base == imax_quant).float().mean().item() * 100.0

    return {
        "mean_nll": mean_nll,
        "mean_nll_base": mean_nll_base,
        "mean_kld": mean_kld,
        "mean_delta_p": mean_delta_p,
        "same_top_1": same_top_1
    }

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the logits benchmark.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=DEVICE)
    parser.add_argument("--models", nargs="*", default=MODELS, help="Models to evaluate")
    parser.add_argument("--bits", nargs="*", type=int, default=BITS, help="Bit widths to evaluate")
    parser.add_argument("--methods", nargs="*", default=METHODS, help="Quantization methods to evaluate")
    parser.add_argument("--chunks", type=int, default=CHUNKS, help="Number of chunks")
    parser.add_argument("--batches", type=int, default=BATCHES, help="Number of tokens per chunk")
    return parser.parse_args()

def pick(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Error: CUDA is not available on this machine.")
    return value

def table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    body = "\n  ".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    return "  " + line + "\n  " + body

def main() -> None:
    args = parse()
    device = pick(args.device)
    models = args.models
    bits_list = args.bits
    methods = args.methods
    chunks = args.chunks
    batch_size = args.batches

    dataset_hash = sha256_file(DATASET_PATH)
    if not dataset_hash:
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run setup.py first.")

    print("Querying model sizes...")
    sizes = {m: get_size(m) for m in models}
    show_plan(device, dataset_hash, models, bits_list, methods, chunks, batch_size, sizes)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    config = {
        "sha256": dataset_hash,
        "bits": bits_list,
        "chunks": chunks,
        "batches": batch_size,
        "groups": GROUP,
        "clusters": CLUSTERS,
        "tolerance": TOLERANCE,
    }

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    out_data = {"config": config, "records": []}
    if RESULT.exists():
        with open(RESULT, "r") as f:
            out_data = json.load(f)

    records = out_data.get("records", [])

    print(f"\nRunning benchmark...")

    for model_id in models:
        print(f"[{model_id}]")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.squeeze()

        required_tokens = chunks * batch_size
        if input_ids.size(0) < required_tokens:
            print(f"Warning: Dataset too small. Truncating chunks...")
            actual_chunks = input_ids.size(0) // batch_size
            input_ids = input_ids[:actual_chunks * batch_size].view(actual_chunks, batch_size)
        else:
            input_ids = input_ids[:required_tokens].view(chunks, batch_size)

        pending = [
            (bits, method)
            for bits in bits_list
            for method in methods
            if not any(r["model"] == model_id and r["method"] == method and r["bits"] == bits for r in records)
        ]
        for bits in bits_list:
            for method in methods:
                if (bits, method) not in pending:
                    print(f"  [Skipped] {method} (INT{bits})")

        if not pending:
            del tokenizer
            continue

        print("\n  Computing baseline logits...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        ).eval()
        baseline_logits = []
        with torch.inference_mode():
            for i in tqdm(range(input_ids.size(0)), desc="  Evaluating chunks", leave=False):
                chunk_input = input_ids[i].unsqueeze(0).to(device)
                baseline_logits.append(base_model(chunk_input).logits.cpu())
                del chunk_input

        del base_model
        if device == "cuda":
            torch.cuda.empty_cache()

        for bits, method in pending:
            print(f"\n  Processing {method} (INT{bits})...")
            quant_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            ).eval()
            replace_layers(
                quant_model,
                method,
                TARGET_LAYERS,
                bits=bits,
                group_size=GROUP,
                clusters=CLUSTERS,
                tolerance=TOLERANCE,
            )
            quant_model = quant_model.to(device).eval()

            chunk_metrics = []
            with torch.inference_mode():
                for i in tqdm(range(input_ids.size(0)), desc="  Evaluating chunks", leave=False):
                    chunk_input = input_ids[i].unsqueeze(0).to(device)
                    labels = chunk_input.clone()
                    logits_base = baseline_logits[i].to(device)
                    logits_quant = quant_model(chunk_input).logits
                    chunk_metrics.append(compute_metrics(logits_base, logits_quant, labels))
                    del chunk_input, labels, logits_base, logits_quant

            agg_metrics = {k: float(np.mean([m[k] for m in chunk_metrics])) for k in chunk_metrics[0]}
            agg_metrics["mean_ppl"]      = math.exp(agg_metrics["mean_nll"])
            agg_metrics["mean_ppl_base"] = math.exp(agg_metrics["mean_nll_base"])
            agg_metrics["delta_ppl"]     = agg_metrics["mean_ppl"] - agg_metrics["mean_ppl_base"]
            del agg_metrics["mean_nll"]
            del agg_metrics["mean_nll_base"]
            del agg_metrics["mean_ppl_base"]
            records.append({
                "model": model_id,
                "method": method,
                "bits": bits,
                "metrics": agg_metrics,
            })
            out_data["records"] = records
            with open(RESULT, "w") as f:
                json.dump(out_data, f, indent=2)

            del quant_model
            if device == "cuda":
                torch.cuda.empty_cache()

        del baseline_logits, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nBenchmark finished: {len(records)} records.")

    if records:
        print(f"\nResults")
        table_rows = [
            [r["model"].split("/")[-1], r["method"], str(r["bits"]),
             f"{r['metrics']['mean_ppl']:.4f}",
             f"{r['metrics']['delta_ppl']:.4f}",
             f"{r['metrics']['mean_delta_p']:.3f}%",
             f"{r['metrics']['mean_kld']:.4f}",
             f"{r['metrics']['same_top_1']:.2f}%"]
            for r in records
        ]
        headers = ["model", "method", "bits", "ppl", "delta_ppl", "delta_p", "kld", "same_top_1"]
        print(f"{table(headers, table_rows)}")

    print(f"\nResults saved to {RESULT}")

if __name__ == "__main__":
    main()
