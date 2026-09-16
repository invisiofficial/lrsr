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
from e2e.kernels.backend import get_lib, ptr, pack_int4


class QuantizedLinear(nn.Module):
    """Base class for weight-only quantized linear layers."""

    def __init__(self, base_layer: nn.Linear, bits: int) -> None:
        super().__init__()
        self.qmax = float((2 ** (bits - 1)) - 1)
        self.backend = "torch"
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

    def _cutlass_gemm_block(self, x_block: torch.Tensor, weight_block: torch.Tensor, m: int, k: int) -> torch.Tensor:
        cols = weight_block.size(0)
        out = torch.empty((m, cols), dtype=torch.float16, device=x_block.device)
        stream = torch.cuda.current_stream(x_block.device).cuda_stream
        get_lib().cutlass_gemm(ptr(x_block.contiguous()), ptr(weight_block), ptr(out), m, cols, k, stream)
        return out

    def _cutlass_group(self, x: torch.Tensor, x_scaled: torch.Tensor, weight_block: torch.Tensor,
                       output: torch.Tensor, cols: torch.Tensor, output_scale: torch.Tensor) -> None:
        m, k = x.numel() // x.size(-1), x.size(-1)
        block = self._cutlass_gemm_block(x_scaled, weight_block, m, k) * output_scale
        flat = output.view(m, -1)
        flat[:, cols] = block

    def _perform_matmul(self, x: torch.Tensor, weight_q: torch.Tensor = None) -> torch.Tensor:
        weight = self.weight_q if weight_q is None else weight_q
        return torch.matmul(x, weight.t().to(x.dtype))

    def _add_bias(self, output: torch.Tensor) -> torch.Tensor:
        return output if self.bias is None else output + self.bias

    @torch.no_grad()
    def enable_cutlass(self) -> "QuantizedLinear":
        # Check requirements
        if self.backend == "cutlass":
            return self
        supported = [
            PerChannelQuantizedLinear,
            PerGroupQuantizedLinear,
            LRSRNaiveQuantizedLinear,
            LRSR1DOSQuantizedLinear,
            LRSRKMeansQuantizedLinear,
        ]
        if type(self) not in supported or self.qmax != 7:
            raise ValueError("CUTLASS W4A16 supports only 4-bits.")

        # Reorder output features for a contiguous column block
        if isinstance(self, LRSR1DOSQuantizedLinear):
            main_idx = (~self.minor_mask).nonzero(as_tuple=True)[0]
            minor_idx = self.minor_mask.nonzero(as_tuple=True)[0]
            self.register_buffer("main_cols", main_idx.contiguous())
            self.register_buffer("minor_cols", minor_idx.contiguous())
            self.weight_q = self.weight_q[torch.cat([main_idx, minor_idx])]
            self.output_scale = self.output_scale[torch.cat([main_idx, minor_idx])]
            self.main_count = int(main_idx.numel())
        elif isinstance(self, LRSRKMeansQuantizedLinear):
            order = torch.argsort(self.cluster_labels, stable=True)
            counts = torch.bincount(self.cluster_labels, minlength=self.input_scales.size(0))
            self.register_buffer("cluster_order", order.contiguous())
            self.register_buffer("cluster_offsets", torch.cat([counts.new_zeros(1), counts.cumsum(0)]))
            self.weight_q = self.weight_q[order]
            self.output_scale = self.output_scale[order]

        self.weight_q = pack_int4(self.weight_q)

        # Convert scales to fp16 contiguous
        for name in ("input_scale", "output_scale", "group_scales", "main_input_scale",
                     "minor_input_scale", "input_scales", "bias"):
            val = getattr(self, name, None)
            if val is not None:
                setattr(self, name, val.detach().to(device=self.weight_q.device, dtype=torch.float16).contiguous())

        self.backend = "cutlass"
        return self


class PerChannelQuantizedLinear(QuantizedLinear):
    """Weight matrix with one output-channel scale per column."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, **_: object) -> None:
        super().__init__(base_layer, bits)
        W = self._source_weight(base_layer)
        scales = scales_per_channel(W, self.qmax)
        self._store_weight(torch.clamp(torch.round(W / scales), -self.qmax, self.qmax))
        self.register_buffer("output_scale", scales.to(base_layer.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "cutlass":
            out = torch.empty((*x.shape[:-1], self.weight_q.size(0)), dtype=torch.float16, device=x.device)
            m, k = x.numel() // x.size(-1), x.size(-1)
            n = self.weight_q.size(0)
            stream = torch.cuda.current_stream(x.device).cuda_stream
            get_lib().cutlass_gemm(ptr(x), ptr(self.weight_q), ptr(out), m, n, k, stream)
            return self._add_bias(out * self.output_scale)
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
        if self.backend == "cutlass":
            out = torch.empty((*x.shape[:-1], self.weight_q.size(0)), dtype=torch.float16, device=x.device)
            m, k = x.numel() // x.size(-1), x.size(-1)
            n = self.weight_q.size(0)
            stream = torch.cuda.current_stream(x.device).cuda_stream
            get_lib().cutlass_pgmm(ptr(x), ptr(self.weight_q), ptr(self.group_scales), ptr(out), m, n, k, stream)
            return self._add_bias(out)
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
        x_scaled = x * self.input_scale
        if self.backend == "cutlass":
            out = torch.empty((*x_scaled.shape[:-1], self.weight_q.size(0)), dtype=torch.float16, device=x.device)
            m, k = x_scaled.numel() // x_scaled.size(-1), x_scaled.size(-1)
            n = self.weight_q.size(0)
            stream = torch.cuda.current_stream(x.device).cuda_stream
            get_lib().cutlass_gemm(ptr(x_scaled), ptr(self.weight_q), ptr(out), m, n, k, stream)
            return self._add_bias(out * self.output_scale)
        output = self._perform_matmul(x_scaled)
        return self._add_bias(output * self.output_scale)


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
        if self.backend == "cutlass":
            output = torch.empty((*x.shape[:-1], self.main_count + self.minor_cols.numel()),
                                 dtype=torch.float16, device=x.device)
            self._cutlass_group(x, x * self.main_input_scale, self.weight_q[:self.main_count],
                                output, self.main_cols, self.output_scale[:self.main_count])
            self._cutlass_group(x, x * self.minor_input_scale, self.weight_q[self.main_count:],
                                output, self.minor_cols, self.output_scale[self.main_count:])
            return self._add_bias(output)
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
        if self.backend == "cutlass":
            output = torch.empty((*x.shape[:-1], self.weight_q.size(0)), dtype=torch.float16, device=x.device)
            offsets = self.cluster_offsets
            for cluster in range(self.input_scales.size(0)):
                start, end = int(offsets[cluster]), int(offsets[cluster + 1])
                if end > start:
                    self._cutlass_group(x, x * self.input_scales[cluster], self.weight_q[start:end],
                                        output, self.cluster_order[start:end],
                                        self.output_scale[start:end])
            return self._add_bias(output)
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


def quantize_layer(base_layer: nn.Linear, method: str, backend: str = "torch", **kwargs: object) -> QuantizedLinear:
    """Builds the dedicated quantized-layer class for given method."""
    if backend not in ("torch", "cutlass"):
        raise ValueError(f"Unsupported backend: {backend}")
    if backend == "cutlass":
        if method not in ("per-channel", "per-group", "lrsr-naive", "lrsr-1dos", "lrsr-kmeans"):
            raise ValueError(f"Unsupported quantization method: {method}")
        if kwargs.get("bits", 4) != 4 or (method == "per-group" and kwargs.get("group_size", 128) != 128):
            raise ValueError("CUTLASS backend requires bits=4 and group_size=128.")
        if not base_layer.weight.is_cuda:
            raise ValueError("CUTLASS preparation requires the source layer on CUDA.")
    try:
        layer_type = LAYER_TYPES[method]
    except KeyError as error:
        raise ValueError(f"Unsupported quantization method: {method}") from error
    layer = layer_type(base_layer, **kwargs)
    return layer.enable_cutlass() if backend == "cutlass" else layer
