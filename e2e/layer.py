import torch
import torch.nn as nn
from typing import Type

from e2e.scales import (
    scales_per_channel,
    scales_per_group,
    scales_lrsr_naive,
    scales_lrsr_1dos,
    scales_lrsr_kmeans,
)


class QuantizedLinear(nn.Module):
    """Base class for weight-only quantized linear layers."""

    def __init__(self, base_layer: nn.Linear, bits: int) -> None:
        super().__init__()
        self.qmax = float((2 ** (bits - 1)) - 1)
        self.register_buffer("weight_q", torch.empty(0, dtype=torch.int8))
        if base_layer.bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_buffer("bias", base_layer.bias.detach().clone())

    @staticmethod
    def _source_weight(base_layer: nn.Linear) -> torch.Tensor:
        return base_layer.weight.detach().t().float()

    def _store_weight(self, W_q: torch.Tensor) -> None:
        self.weight_q = W_q.t().contiguous().to(torch.int8)

    def _perform_matmul(self, x: torch.Tensor, weight_q: torch.Tensor = None) -> torch.Tensor:
        weight = self.weight_q if weight_q is None else weight_q
        return torch.matmul(x, weight.t().to(x.dtype))

    def _add_bias(self, output: torch.Tensor) -> torch.Tensor:
        return output if self.bias is None else output + self.bias


class PerChannelQuantizedLinear(QuantizedLinear):
    """Weight matrix with one output-channel scale per column."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, **_: object) -> None:
        super().__init__(base_layer, bits)
        W = self._source_weight(base_layer)
        scales = scales_per_channel(W, self.qmax)
        self._store_weight(torch.clamp(torch.round(W / scales), -self.qmax, self.qmax))
        self.register_buffer("output_scale", scales.to(base_layer.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._add_bias(self._perform_matmul(x) * self.output_scale)


class PerGroupQuantizedLinear(QuantizedLinear):
    """Weight matrix with compact `(input_groups, output_features)` scales."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, group_size: int = 128, **_: object) -> None:
        super().__init__(base_layer, bits)
        W = self._source_weight(base_layer)
        if W.size(0) % group_size != 0:
            raise ValueError(f"in_features ({W.size(0)}) must be divisible by group_size ({group_size}).")
        scales = scales_per_group(W, group_size, self.qmax)
        expanded_scales = scales.repeat_interleave(group_size, dim=0)
        self._store_weight(torch.clamp(torch.round(W / expanded_scales), -self.qmax, self.qmax))
        self.group_size = group_size
        self.register_buffer("group_scales", scales.to(base_layer.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        groups = self.group_scales.size(0)
        flat_x = x.reshape(-1, groups, self.group_size)
        x_groups = flat_x.transpose(0, 1)
        weight_groups = self.weight_q.reshape(self.weight_q.size(0), groups, self.group_size)
        weight_groups = weight_groups.permute(1, 2, 0).to(x.dtype)
        group_outputs = torch.matmul(x_groups, weight_groups).permute(1, 0, 2)
        output = (group_outputs * self.group_scales.unsqueeze(0)).sum(dim=1)
        return self._add_bias(output.reshape(*original_shape, -1))


class LRSRNaiveQuantizedLinear(QuantizedLinear):
    """Low-rank scale approximation fused into input and output vectors."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, **_: object) -> None:
        super().__init__(base_layer, bits)
        W = self._source_weight(base_layer)
        ideal_scales = torch.clamp(W.abs() / self.qmax, min=1e-7)
        input_scale, output_scale = scales_lrsr_naive(ideal_scales)
        W_q = torch.clamp(
            torch.round(W / (input_scale.unsqueeze(1) * output_scale.unsqueeze(0))),
            -self.qmax,
            self.qmax,
        )
        self._store_weight(W_q)
        self.register_buffer("input_scale", input_scale.to(base_layer.weight.dtype))
        self.register_buffer("output_scale", output_scale.to(base_layer.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._perform_matmul(x * self.input_scale) * self.output_scale
        return self._add_bias(output)


class LRSR1DOSQuantizedLinear(QuantizedLinear):
    """Two low-rank scale approximations selected by output-column partition."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, **_: object) -> None:
        super().__init__(base_layer, bits)
        W = self._source_weight(base_layer)
        ideal_scales = torch.clamp(W.abs() / self.qmax, min=1e-7)
        main_input, minor_input, output_scale, minor_mask = scales_lrsr_1dos(ideal_scales)
        input_scales = torch.where(minor_mask.unsqueeze(0), minor_input.unsqueeze(1), main_input.unsqueeze(1))
        W_q = torch.clamp(torch.round(W / (input_scales * output_scale.unsqueeze(0))), -self.qmax, self.qmax)
        self._store_weight(W_q)
        self.register_buffer("main_input_scale", main_input.to(base_layer.weight.dtype))
        self.register_buffer("minor_input_scale", minor_input.to(base_layer.weight.dtype))
        self.register_buffer("output_scale", output_scale.to(base_layer.weight.dtype))
        self.register_buffer("minor_mask", minor_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty(*x.shape[:-1], self.weight_q.size(0), device=x.device, dtype=x.dtype)
        main_mask = ~self.minor_mask
        if main_mask.any():
            output[..., main_mask] = self._perform_matmul(x * self.main_input_scale, self.weight_q[main_mask])
        if self.minor_mask.any():
            output[..., self.minor_mask] = self._perform_matmul(x * self.minor_input_scale, self.weight_q[self.minor_mask])
        return self._add_bias(output * self.output_scale)


class LRSRKMeansQuantizedLinear(QuantizedLinear):
    """One low-rank scale approximation per output-column K-Means cluster."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        clusters: int = 3,
        tolerance: float = 0.05,
        **_: object,
    ) -> None:
        super().__init__(base_layer, bits)
        W = self._source_weight(base_layer)
        ideal_scales = torch.clamp(W.abs() / self.qmax, min=1e-7)
        input_scales, output_scale, labels = scales_lrsr_kmeans(ideal_scales, clusters, tolerance)
        W_q = torch.clamp(
            torch.round(W / (input_scales[labels].t() * output_scale.unsqueeze(0))),
            -self.qmax,
            self.qmax,
        )
        self._store_weight(W_q)
        self.register_buffer("input_scales", input_scales.to(base_layer.weight.dtype))
        self.register_buffer("output_scale", output_scale.to(base_layer.weight.dtype))
        self.register_buffer("cluster_labels", labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty(*x.shape[:-1], self.weight_q.size(0), device=x.device, dtype=x.dtype)
        for cluster in range(self.input_scales.size(0)):
            mask = self.cluster_labels == cluster
            if mask.any():
                output[..., mask] = self._perform_matmul(x * self.input_scales[cluster], self.weight_q[mask])
        return self._add_bias(output * self.output_scale)


LAYER_TYPES: dict[str, Type[QuantizedLinear]] = {
    "per-channel": PerChannelQuantizedLinear,
    "per-group": PerGroupQuantizedLinear,
    "lrsr-naive": LRSRNaiveQuantizedLinear,
    "lrsr-1dos": LRSR1DOSQuantizedLinear,
    "lrsr-kmeans": LRSRKMeansQuantizedLinear,
}


def quantize_layer(base_layer: nn.Linear, method: str, **kwargs: object) -> QuantizedLinear:
    """Builds the dedicated quantized-layer class for given method."""
    try:
        return LAYER_TYPES[method](base_layer, **kwargs)
    except KeyError as error:
        raise ValueError(f"Unsupported quantization method: {method}") from error
