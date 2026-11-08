import re
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
CACHEABLE_METHODS = {"lrsr-1dos", "lrsr-kmeans"}


class LayerCache:
    """Disk cache for layer partitions."""

    def __init__(
        self,
        module: nn.Module,
        method: str,
        cache_dir: Path = CACHE_DIR,
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.model_id = self._resolve_model_id(module, kwargs.get("model_id"))
        self.enabled = enabled and (method in CACHEABLE_METHODS) and bool(self.model_id)
        self.dirty = False
        self.data: Dict[str, torch.Tensor] = {}
        self.path: Optional[Path] = None

        if self.enabled:
            self.path = self._build_path(cache_dir, self.model_id, method, kwargs)
            if self.path.is_file():
                self.data = torch.load(self.path, map_location="cpu", weights_only=True)

    @staticmethod
    def _resolve_model_id(module: nn.Module, explicit_id: Optional[str] = None) -> Optional[str]:
        if explicit_id:
            return explicit_id
        config = getattr(module, "config", None)
        name_or_path = getattr(config, "_name_or_path", None)
        if name_or_path and isinstance(name_or_path, str):
            return name_or_path
        return None

    @staticmethod
    def _build_path(cache_dir: Path, model_id: str, method: str, kwargs: Dict[str, Any]) -> Path:
        safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "__", model_id.strip("/"))
        parts = [safe_model, method]

        if method == "lrsr-kmeans":
            clusters = kwargs.get("clusters", 3)
            tolerance = kwargs.get("tolerance", 0.05)
            parts.append(f"c{clusters}_tol{tolerance}")

        if kwargs.get("transform") not in (None, "none"):
            parts.append(f"tr-{kwargs['transform']}")

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
