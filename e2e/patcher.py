import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple

from e2e.layer import quantize_layer


def _get_layers(
    module: nn.Module,
    target_layers: List[str],
) -> List[Tuple[nn.Module, str, nn.Linear]]:
    """Collect target layers."""
    targets = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(target in name for target in target_layers):
            targets.append((module, name, child))
        else:
            targets.extend(_get_layers(child, target_layers))
    return targets

def replace_layers(
    module: nn.Module,
    method: str,
    target_layers: List[str],
    **kwargs
) -> nn.Module:
    """Replaces target linear layers."""
    targets = _get_layers(module, target_layers)
    iterator = tqdm(
        targets,
        desc=f"  Patching layers",
        unit="layer",
        leave=False
    )
    for parent, name, child in iterator:
        setattr(parent, name, quantize_layer(child, method=method, **kwargs))

    return module
