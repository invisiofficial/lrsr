import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple

from e2e.cache import LayerCache
from e2e.layer import quantize_layer


def _get_layers(
    module: nn.Module,
    target_layers: List[str],
    prefix: str = "",
) -> List[Tuple[nn.Module, str, str, nn.Linear]]:
    """Collect target layers with their full module path."""
    targets = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and any(target in name for target in target_layers):
            targets.append((module, name, full_name, child))
        else:
            targets.extend(_get_layers(child, target_layers, prefix=full_name))
    return targets

def replace_layers(
    module: nn.Module,
    method: str,
    target_layers: List[str],
    use_cache: bool = True,
    **kwargs
) -> nn.Module:
    """Replaces target linear layers."""
    targets = _get_layers(module, target_layers)
    cache = LayerCache(module, method=method, enabled=use_cache, **kwargs)

    iterator = tqdm(
        targets,
        desc="  Patching layers",
        unit="layer",
        leave=False
    )
    try:
        for parent, name, full_name, child in iterator:
            partition = cache.get(full_name)
            layer = quantize_layer(child, method=method, partition=partition, **kwargs)
            if partition is None:
                if hasattr(layer, "minor_mask"):
                    cache.put(full_name, layer.minor_mask)
                elif hasattr(layer, "cluster_labels"):
                    cache.put(full_name, layer.cluster_labels)
            setattr(parent, name, layer)
    finally:
        cache.save()

    return module
