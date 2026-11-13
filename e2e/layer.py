import math
import torch
import torch.nn as nn
from typing import Type, Optional, Dict, Tuple

from e2e.scales import (
    scales_per_channel,
    scales_per_group,
    scales_lrsr_naive,
    scales_lrsr_1dos,
    scales_lrsr_kmeans,
)
from e2e.kernels.backend import get_lib, ptr, pack_int4


#region Helpers

_HADAMARD_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _get_hadamard_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Returns a cached normalized Sylvester Hadamard matrix of shape (size, size)."""
    key = (size, device, dtype)
    mat = _HADAMARD_CACHE.get(key)
    if mat is None:
        h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
        while h.size(0) < size:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        mat = (h / math.sqrt(size)).to(dtype=dtype).contiguous()
        _HADAMARD_CACHE[key] = mat
    return mat

def apply_hadamard(x: torch.Tensor, signs: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Applies randomized block-diagonal Hadamard transform with block_size=128."""
    k = x.size(-1)
    block = math.gcd(k, block_size)
    x_signed = x * signs
    if block <= 1:
        return x_signed
    h_mat = _get_hadamard_matrix(block, x.device, x.dtype)
    return torch.matmul(x_signed.reshape(*x.shape[:-1], k // block, block), h_mat).reshape_as(x)

def gptq_quantize_weight(
    W: torch.Tensor,
    scale_grid: torch.Tensor,
    u_blocks: torch.Tensor,
    qmax: float,
) -> torch.Tensor:
    """Block-GPTQ quantization with Cholesky error compensation (block_size=128)."""
    num_blocks, bsz, _ = u_blocks.shape
    n_out = W.size(1)

    w_blocks = W.clone().view(num_blocks, bsz, n_out)
    s_blocks = scale_grid.view(num_blocks, bsz, n_out)
    q_blocks = torch.empty_like(w_blocks)

    for j in range(bsz):
        w_col = w_blocks[:, j, :]
        s_col = s_blocks[:, j, :]
        q_col = torch.clamp(torch.round(w_col / s_col), -qmax, qmax)
        q_blocks[:, j, :] = q_col
        if j + 1 < bsz:
            err = w_col - q_col * s_col
            u_row = u_blocks[:, j, j + 1 :]
            w_blocks[:, j + 1 :, :] -= u_row.unsqueeze(-1) * err.unsqueeze(1)

    return q_blocks.reshape_as(W)

#endregion

class QuantizedLinear(nn.Module):
    """Base class for weight-only quantized linear layers."""

    def __init__(self, base_layer: nn.Linear, bits: int, hadamard: bool = False) -> None:
        super().__init__()
        self.qmax = float((2 ** (bits - 1)) - 1)
        self.backend = "torch"
        self.hadamard = hadamard
        self._target_device = base_layer.weight.device
        self._target_dtype = base_layer.weight.dtype
        self.register_buffer("weight_q", torch.empty(0, dtype=torch.int8, device=self._target_device))
        if base_layer.bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_buffer("bias", base_layer.bias.detach().clone())

        if hadamard:
            k = base_layer.in_features
            seed = k * 100003 + base_layer.out_features
            gen = torch.Generator(device="cpu").manual_seed(seed)
            signs = (torch.randint(0, 2, (k,), generator=gen, dtype=torch.int8) * 2 - 1).to(
                device=self._target_device, dtype=self._target_dtype
            )
            self.register_buffer("hadamard_signs", signs)
        else:
            self.register_parameter("hadamard_signs", None)

    def _source_weight(self, base_layer: nn.Linear) -> torch.Tensor:
        w = base_layer.weight.detach().float()
        if self.hadamard:
            w = apply_hadamard(w, self.hadamard_signs.to(device=w.device, dtype=torch.float32))
        return w.t().contiguous()

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.hadamard:
            return apply_hadamard(x, self.hadamard_signs)
        return x

    def _store_weight(self, W_q: torch.Tensor) -> None:
        self.weight_q = W_q.t().contiguous().to(device=self._target_device, dtype=torch.int8)

    def _to_target(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return tensor.to(device=self._target_device, dtype=self._target_dtype if dtype is None else dtype)

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
        if self.backend == "cutlass":
            return self
        if self.qmax != 7:
            raise ValueError("CUTLASS W4A16 supports only 4-bits.")

        if isinstance(self, LRSR1DOSQuantizedLinear):
            main_idx = (~self.minor_mask).nonzero(as_tuple=True)[0]
            minor_idx = self.minor_mask.nonzero(as_tuple=True)[0]
            self.register_buffer("main_cols", main_idx.contiguous())
            self.register_buffer("minor_cols", minor_idx.contiguous())
            self.weight_q = self.weight_q[torch.cat([main_idx, minor_idx])]
            self.output_scale = self.output_scale[torch.cat([main_idx, minor_idx])]
            self.main_count = int(main_idx.numel())
            self.stream_main = torch.cuda.Stream(device=self.weight_q.device)
            self.stream_minor = torch.cuda.Stream(device=self.weight_q.device)
        elif isinstance(self, LRSRKMeansQuantizedLinear):
            order = torch.argsort(self.cluster_labels, stable=True)
            counts = torch.bincount(self.cluster_labels, minlength=self.input_scales.size(0))
            self.register_buffer("cluster_order", order.contiguous())
            self.register_buffer("cluster_offsets", torch.cat([counts.new_zeros(1), counts.cumsum(0)]))
            self.weight_q = self.weight_q[order]
            self.output_scale = self.output_scale[order]
            offsets = self.cluster_offsets.tolist()
            self.cluster_ranges = [
                (int(offsets[c]), int(offsets[c + 1])) for c in range(self.input_scales.size(0))
            ]
            self.streams = [torch.cuda.Stream(device=self.weight_q.device) for _ in range(self.input_scales.size(0))]

        self.weight_q = pack_int4(self.weight_q)

        for name in ("input_scale", "output_scale", "group_scales", "main_input_scale",
                     "minor_input_scale", "input_scales", "bias", "hadamard_signs"):
            val = getattr(self, name, None)
            if val is not None:
                setattr(self, name, val.detach().to(device=self.weight_q.device, dtype=torch.float16).contiguous())

        self.backend = "cutlass"
        return self


#region Base Granularities

class PerChannelQuantizedLinear(QuantizedLinear):
    """Weight matrix with one output-channel scale per column."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, hadamard: bool = False, **_: object) -> None:
        super().__init__(base_layer, bits, hadamard=hadamard)
        W = self._source_weight(base_layer)
        scales = scales_per_channel(W, self.qmax)
        self._store_weight(torch.clamp(torch.round(W / scales), -self.qmax, self.qmax))
        self.register_buffer("output_scale", self._to_target(scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_input(x)
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

    def __init__(self, base_layer: nn.Linear, bits: int = 4, group_size: int = 128, hadamard: bool = False, **_: object) -> None:
        super().__init__(base_layer, bits, hadamard=hadamard)
        W = self._source_weight(base_layer)
        if W.size(0) % group_size != 0:
            raise ValueError(f"in_features ({W.size(0)}) must be divisible by group_size ({group_size}).")
        scales = scales_per_group(W, group_size, self.qmax)
        expanded_scales = scales.repeat_interleave(group_size, dim=0)
        self._store_weight(torch.clamp(torch.round(W / expanded_scales), -self.qmax, self.qmax))
        self.group_size = group_size
        self.register_buffer("group_scales", self._to_target(scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_input(x)
        if self.backend == "cutlass":
            out = torch.empty((*x.shape[:-1], self.weight_q.size(0)), dtype=torch.float16, device=x.device)
            m, k = x.numel() // x.size(-1), x.size(-1)
            n = self.weight_q.size(0)
            stream = torch.cuda.current_stream(x.device).cuda_stream
            get_lib().cutlass_pgmm(ptr(x), ptr(self.weight_q), ptr(self.group_scales), ptr(out), m, n, k, stream)
            return self._add_bias(out)
        n, k = self.weight_q.shape
        w_fp = self.weight_q.to(x.dtype)
        w_fp.view(n, -1, self.group_size).mul_(self.group_scales.t().unsqueeze(-1))
        return self._add_bias(torch.matmul(x, w_fp.t()))


class LRSRNaiveQuantizedLinear(QuantizedLinear):
    """Low-rank scale approximation fused into input and output vectors."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, hadamard: bool = False, **_: object) -> None:
        super().__init__(base_layer, bits, hadamard=hadamard)
        W = self._source_weight(base_layer)
        ideal_scales = torch.clamp(W.abs() / self.qmax, min=1e-7)
        output_scale, input_scale = scales_lrsr_naive(ideal_scales)
        W_q = torch.clamp(
            torch.round(W / (input_scale.unsqueeze(1) * output_scale.unsqueeze(0))),
            -self.qmax,
            self.qmax,
        )
        self._store_weight(W_q)
        self.register_buffer("input_scale", self._to_target(input_scale))
        self.register_buffer("output_scale", self._to_target(output_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_input(x)
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

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        partition: Optional[torch.Tensor] = None,
        hadamard: bool = False,
        **_: object,
    ) -> None:
        super().__init__(base_layer, bits, hadamard=hadamard)
        W = self._source_weight(base_layer)
        ideal_scales = torch.clamp(W.abs() / self.qmax, min=1e-7)
        main_input, minor_input, output_scale, minor_mask = scales_lrsr_1dos(
            ideal_scales, minor_mask=partition
        )
        input_scales = torch.where(minor_mask.unsqueeze(0), minor_input.unsqueeze(1), main_input.unsqueeze(1))
        W_q = torch.clamp(torch.round(W / (input_scales * output_scale.unsqueeze(0))), -self.qmax, self.qmax)
        self._store_weight(W_q)
        self.register_buffer("main_input_scale", self._to_target(main_input))
        self.register_buffer("minor_input_scale", self._to_target(minor_input))
        self.register_buffer("output_scale", self._to_target(output_scale))
        self.register_buffer("minor_mask", self._to_target(minor_mask, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_input(x)
        if self.backend == "cutlass":
            output = torch.empty((*x.shape[:-1], self.main_count + self.minor_cols.numel()),
                                 dtype=torch.float16, device=x.device)

            current_stream = torch.cuda.current_stream(x.device)

            if self.main_count > 0:
                self.stream_main.wait_stream(current_stream)
                with torch.cuda.stream(self.stream_main):
                    x_main = x * self.main_input_scale
                    self._cutlass_group(x, x_main, self.weight_q[:self.main_count],
                                        output, self.main_cols, self.output_scale[:self.main_count])

            if self.minor_cols.numel() > 0:
                self.stream_minor.wait_stream(current_stream)
                with torch.cuda.stream(self.stream_minor):
                    x_minor = x * self.minor_input_scale
                    self._cutlass_group(x, x_minor, self.weight_q[self.main_count:],
                                        output, self.minor_cols, self.output_scale[self.main_count:])

            if self.main_count > 0:
                current_stream.wait_stream(self.stream_main)
            if self.minor_cols.numel() > 0:
                current_stream.wait_stream(self.stream_minor)

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
        partition: Optional[torch.Tensor] = None,
        hadamard: bool = False,
        **_: object,
    ) -> None:
        super().__init__(base_layer, bits, hadamard=hadamard)
        W = self._source_weight(base_layer)
        ideal_scales = torch.clamp(W.abs() / self.qmax, min=1e-7)
        input_scales, output_scale, labels = scales_lrsr_kmeans(
            ideal_scales, clusters, tolerance, labels=partition
        )
        W_q = torch.clamp(
            torch.round(W / (input_scales[labels].t() * output_scale.unsqueeze(0))),
            -self.qmax,
            self.qmax,
        )
        self._store_weight(W_q)
        self.register_buffer("input_scales", self._to_target(input_scales))
        self.register_buffer("output_scale", self._to_target(output_scale))
        self.register_buffer("cluster_labels", self._to_target(labels, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._preprocess_input(x)
        if self.backend == "cutlass":
            output = torch.empty((*x.shape[:-1], self.weight_q.size(0)), dtype=torch.float16, device=x.device)
            current_stream = torch.cuda.current_stream(x.device)

            for cluster, (start, end) in enumerate(self.cluster_ranges):
                if end > start:
                    self.streams[cluster].wait_stream(current_stream)
                    with torch.cuda.stream(self.streams[cluster]):
                        x_scaled = x * self.input_scales[cluster]
                        self._cutlass_group(x, x_scaled, self.weight_q[start:end],
                                            output, self.cluster_order[start:end],
                                            self.output_scale[start:end])

            for stream in self.streams:
                current_stream.wait_stream(stream)

            return self._add_bias(output)
        output = torch.empty(*x.shape[:-1], self.weight_q.size(0), device=x.device, dtype=x.dtype)
        for cluster in range(self.input_scales.size(0)):
            mask = self.cluster_labels == cluster
            if mask.any():
                output[..., mask] = self._perform_matmul(x * self.input_scales[cluster], self.weight_q[mask])
        return self._add_bias(output * self.output_scale)

#endregion

#region QuaRot Granularities

class QuaRotPerChannelQuantizedLinear(PerChannelQuantizedLinear):
    """QuaRot (Block-128 Hadamard) + Per-Channel Granularity."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, **kwargs: object) -> None:
        super().__init__(base_layer, bits=bits, hadamard=True, **kwargs)


class QuaRotPerGroupQuantizedLinear(PerGroupQuantizedLinear):
    """QuaRot (Block-128 Hadamard) + Per-Group (g=128) Granularity."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, group_size: int = 128, **kwargs: object) -> None:
        super().__init__(base_layer, bits=bits, group_size=group_size, hadamard=True, **kwargs)


class QuaRotLRSRNaiveQuantizedLinear(LRSRNaiveQuantizedLinear):
    """QuaRot (Block-128 Hadamard) + LRSR-Naive Granularity."""

    def __init__(self, base_layer: nn.Linear, bits: int = 4, **kwargs: object) -> None:
        super().__init__(base_layer, bits=bits, hadamard=True, **kwargs)

#endregion

#region AWQ Granularities

class AWQPerChannelQuantizedLinear(LRSRNaiveQuantizedLinear):
    """AWQ + Per-Channel Granularity."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        calib: Optional[Dict[str, torch.Tensor]] = None,
        grid_size: int = 20,
        **_: object,
    ) -> None:
        QuantizedLinear.__init__(self, base_layer, bits)
        W = self._source_weight(base_layer)
        if calib is None:
            raise ValueError("AWQ requires calibration statistics ('calib').")

        act_mean = calib["act_mean"].to(device=W.device, dtype=W.dtype).clamp(min=1e-5)
        act_rms = calib["act_rms"].to(device=W.device, dtype=W.dtype).unsqueeze(1)

        best_loss = float("inf")
        best_in_scale = torch.ones(W.size(0), device=W.device, dtype=W.dtype)
        best_out_scale = scales_per_channel(W, self.qmax)

        for i in range(grid_size + 1):
            gamma = i / float(grid_size)
            alpha = act_mean.pow(gamma)
            alpha = alpha / (alpha.max() * alpha.min()).sqrt().clamp(min=1e-7)
            in_scale = 1.0 / alpha
            w_scaled = W / in_scale.unsqueeze(1)
            out_scale = scales_per_channel(w_scaled, self.qmax)
            scale_mat = in_scale.unsqueeze(1) * out_scale.unsqueeze(0)
            w_q = torch.clamp(torch.round(W / scale_mat), -self.qmax, self.qmax)
            diff = (W - w_q * scale_mat) * act_rms
            loss = float(torch.mean(diff * diff).item())
            if loss < best_loss:
                best_loss = loss
                best_in_scale = in_scale
                best_out_scale = out_scale

        W_q = torch.clamp(
            torch.round(W / (best_in_scale.unsqueeze(1) * best_out_scale.unsqueeze(0))),
            -self.qmax,
            self.qmax,
        )
        self._store_weight(W_q)
        self.register_buffer("input_scale", self._to_target(best_in_scale))
        self.register_buffer("output_scale", self._to_target(best_out_scale))


class AWQPerGroupQuantizedLinear(PerGroupQuantizedLinear):
    """AWQ + Per-Group (g=128) Granularity."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        calib: Optional[Dict[str, torch.Tensor]] = None,
        grid_size: int = 20,
        **_: object,
    ) -> None:
        QuantizedLinear.__init__(self, base_layer, bits)
        W = self._source_weight(base_layer)
        if W.size(0) % group_size != 0:
            raise ValueError(f"in_features ({W.size(0)}) must be divisible by group_size ({group_size}).")
        if calib is None:
            raise ValueError("AWQ requires calibration statistics ('calib').")

        act_mean = calib["act_mean"].to(device=W.device, dtype=W.dtype).clamp(min=1e-5)
        act_rms = calib["act_rms"].to(device=W.device, dtype=W.dtype).unsqueeze(1)

        best_loss = float("inf")
        best_in_scale = torch.ones(W.size(0), device=W.device, dtype=W.dtype)
        best_group_scales = scales_per_group(W, group_size, self.qmax)

        for i in range(grid_size + 1):
            gamma = i / float(grid_size)
            alpha = act_mean.pow(gamma)
            alpha = alpha / (alpha.max() * alpha.min()).sqrt().clamp(min=1e-7)
            in_scale = 1.0 / alpha
            w_scaled = W / in_scale.unsqueeze(1)
            g_scales = scales_per_group(w_scaled, group_size, self.qmax)
            exp_scales = g_scales.repeat_interleave(group_size, dim=0)
            scale_mat = in_scale.unsqueeze(1) * exp_scales
            w_q = torch.clamp(torch.round(W / scale_mat), -self.qmax, self.qmax)
            diff = (W - w_q * scale_mat) * act_rms
            loss = float(torch.mean(diff * diff).item())
            if loss < best_loss:
                best_loss = loss
                best_in_scale = in_scale
                best_group_scales = g_scales

        exp_best = best_group_scales.repeat_interleave(group_size, dim=0)
        W_q = torch.clamp(torch.round(W / (best_in_scale.unsqueeze(1) * exp_best)), -self.qmax, self.qmax)
        self._store_weight(W_q)
        self.group_size = group_size
        self.register_buffer("input_scale", self._to_target(best_in_scale))
        self.register_buffer("group_scales", self._to_target(best_group_scales))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x * self.input_scale)


class AWQLRSRNaiveQuantizedLinear(LRSRNaiveQuantizedLinear):
    """AWQ + LRSR-Naive Granularity."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        calib: Optional[Dict[str, torch.Tensor]] = None,
        grid_size: int = 20,
        **_: object,
    ) -> None:
        QuantizedLinear.__init__(self, base_layer, bits)
        W = self._source_weight(base_layer)
        if calib is None:
            raise ValueError("AWQ requires calibration statistics ('calib').")

        act_mean = calib["act_mean"].to(device=W.device, dtype=W.dtype).clamp(min=1e-5)
        act_rms = calib["act_rms"].to(device=W.device, dtype=W.dtype).unsqueeze(1)

        best_loss = float("inf")
        best_in_scale = torch.ones(W.size(0), device=W.device, dtype=W.dtype)
        best_out_scale = scales_per_channel(W, self.qmax)

        for i in range(grid_size + 1):
            gamma = i / float(grid_size)
            alpha = act_mean.pow(gamma)
            alpha = alpha / (alpha.max() * alpha.min()).sqrt().clamp(min=1e-7)
            s_awq = 1.0 / alpha
            w_scaled = W / s_awq.unsqueeze(1)
            ideal_scales = torch.clamp(w_scaled.abs() / self.qmax, min=1e-7)
            out_scale, v_lrsr = scales_lrsr_naive(ideal_scales)
            in_scale = s_awq * v_lrsr
            scale_mat = in_scale.unsqueeze(1) * out_scale.unsqueeze(0)
            w_q = torch.clamp(torch.round(W / scale_mat), -self.qmax, self.qmax)
            diff = (W - w_q * scale_mat) * act_rms
            loss = float(torch.mean(diff * diff).item())
            if loss < best_loss:
                best_loss = loss
                best_in_scale = in_scale
                best_out_scale = out_scale

        W_q = torch.clamp(
            torch.round(W / (best_in_scale.unsqueeze(1) * best_out_scale.unsqueeze(0))),
            -self.qmax,
            self.qmax,
        )
        self._store_weight(W_q)
        self.register_buffer("input_scale", self._to_target(best_in_scale))
        self.register_buffer("output_scale", self._to_target(best_out_scale))

#endregion

#region GPTQ Granularities

class GPTQPerChannelQuantizedLinear(PerChannelQuantizedLinear):
    """Block-GPTQ (g=128) + Per-Channel Granularity."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        calib: Optional[torch.Tensor] = None,
        **_: object,
    ) -> None:
        QuantizedLinear.__init__(self, base_layer, bits)
        W = self._source_weight(base_layer)
        if calib is None:
            raise ValueError("GPTQ requires block Cholesky factors ('calib').")

        scales = scales_per_channel(W, self.qmax)
        scale_grid = scales.unsqueeze(0).expand_as(W)
        u_blocks = calib.to(device=W.device, dtype=W.dtype)
        W_q = gptq_quantize_weight(W, scale_grid, u_blocks, self.qmax)
        self._store_weight(W_q)
        self.register_buffer("output_scale", self._to_target(scales))


class GPTQPerGroupQuantizedLinear(PerGroupQuantizedLinear):
    """Block-GPTQ (g=128) + Per-Group (g=128) Granularity."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
        calib: Optional[torch.Tensor] = None,
        **_: object,
    ) -> None:
        QuantizedLinear.__init__(self, base_layer, bits)
        W = self._source_weight(base_layer)
        if W.size(0) % group_size != 0:
            raise ValueError(f"in_features ({W.size(0)}) must be divisible by group_size ({group_size}).")
        if calib is None:
            raise ValueError("GPTQ requires block Cholesky factors ('calib').")

        scales = scales_per_group(W, group_size, self.qmax)
        scale_grid = scales.repeat_interleave(group_size, dim=0)
        u_blocks = calib.to(device=W.device, dtype=W.dtype)
        W_q = gptq_quantize_weight(W, scale_grid, u_blocks, self.qmax)
        self._store_weight(W_q)
        self.group_size = group_size
        self.register_buffer("group_scales", self._to_target(scales))


class GPTQLRSRNaiveQuantizedLinear(LRSRNaiveQuantizedLinear):
    """Block-GPTQ (g=128) + LRSR-Naive Granularity."""

    def __init__(
        self,
        base_layer: nn.Linear,
        bits: int = 4,
        calib: Optional[torch.Tensor] = None,
        **_: object,
    ) -> None:
        QuantizedLinear.__init__(self, base_layer, bits)
        W = self._source_weight(base_layer)
        if calib is None:
            raise ValueError("GPTQ requires block Cholesky factors ('calib').")

        ideal_scales = torch.clamp(W.abs() / self.qmax, min=1e-7)
        output_scale, input_scale = scales_lrsr_naive(ideal_scales)
        scale_grid = input_scale.unsqueeze(1) * output_scale.unsqueeze(0)
        u_blocks = calib.to(device=W.device, dtype=W.dtype)
        W_q = gptq_quantize_weight(W, scale_grid, u_blocks, self.qmax)
        self._store_weight(W_q)
        self.register_buffer("input_scale", self._to_target(input_scale))
        self.register_buffer("output_scale", self._to_target(output_scale))

#endregion

LAYER_TYPES: dict[str, Type[QuantizedLinear]] = {
    "per-channel": PerChannelQuantizedLinear,
    "per-group": PerGroupQuantizedLinear,
    "lrsr-naive": LRSRNaiveQuantizedLinear,
    "lrsr-1dos": LRSR1DOSQuantizedLinear,
    "lrsr-kmeans": LRSRKMeansQuantizedLinear,
    "quarot-per-channel": QuaRotPerChannelQuantizedLinear,
    "quarot-per-group": QuaRotPerGroupQuantizedLinear,
    "quarot-lrsr-naive": QuaRotLRSRNaiveQuantizedLinear,
    "awq-per-channel": AWQPerChannelQuantizedLinear,
    "awq-per-group": AWQPerGroupQuantizedLinear,
    "awq-lrsr-naive": AWQLRSRNaiveQuantizedLinear,
    "gptq-per-channel": GPTQPerChannelQuantizedLinear,
    "gptq-per-group": GPTQPerGroupQuantizedLinear,
    "gptq-lrsr-naive": GPTQLRSRNaiveQuantizedLinear,
}


def quantize_layer(base_layer: nn.Linear, method: str, backend: str = "torch", **kwargs: object) -> QuantizedLinear:
    """Builds the dedicated quantized-layer class for given method."""
    if backend not in ("torch", "cutlass"):
        raise ValueError(f"Unsupported backend: {backend}")
    if backend == "cutlass":
        if method not in LAYER_TYPES:
            raise ValueError(f"Unsupported quantization method: {method}")
        if kwargs.get("bits", 4) != 4 or ("per-group" in method and kwargs.get("group_size", 128) != 128):
            raise ValueError("CUTLASS backend requires bits=4 and group_size=128.")
        if not base_layer.weight.is_cuda:
            raise ValueError("CUTLASS preparation requires the source layer on CUDA.")
    try:
        layer_type = LAYER_TYPES[method]
    except KeyError as error:
        raise ValueError(f"Unsupported quantization method: {method}") from error
    layer = layer_type(base_layer, **kwargs)
    return layer.enable_cutlass() if backend == "cutlass" else layer
