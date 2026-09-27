# LRSR: Hardware-Friendly Approximation of Per-Element Quantization via Low-Rank Scale Reconstruction

> Post-training quantization is a crucial component for deploying Large Language Models (LLMs) in resource-constrained environments. While fine-grained methods (e.g., per-group) offer high accuracy, they heavily rely on hardware-specific implementations and suffer from memory-bound computations, limiting their applicability on standard matrix multiplication accelerators. Conversely, hardware-friendly quantizations (e.g., per-channel) experience severe accuracy degradation at low bit-widths. To address these limitations, we propose Low-Rank Scale Reconstruction (LRSR), a novel quantization method that approximates per-element scale matrices using an upper-bounding rank-1 decomposition, enabling mathematically correct reconstruction via general matrix multiplication (GEMM). Furthermore, we introduce several synergetic clusterization techniques to significantly boost accuracy and allow optimal hardware load balancing. Experimental evaluations on Qwen3 models demonstrate that at INT4, LRSR reduces KL-Divergence by up to 31% compared to the per-channel baseline, while utilizing 29× less metadata memory compared to per-group (g=128) quantization. This establishes a new Pareto frontier for near-zero overhead, hardware-friendly and calibration-free post-training quantizations.

This repository contains the official reference implementation of LRSR, the complete benchmark pipeline and visualization figures used in [the paper](https://doi.org/10.5281/zenodo.22976487).

The main directories are:
- [`lrsr/`](lrsr) - Core quantization math: upper-bounding rank-1 decomposition, clusterization, and shared metrics/utilities.
- [`e2e/`](e2e) - Quantized inference runtime: per-method scale computation, quantized layers and the CUTLASS W4A16 kernels.
- [`benchmarks/`](benchmarks) - Benchmark pipelines (layer, logits, runtime).
- [`notebooks/`](notebooks) - Jupyter notebooks that visualize concept and the benchmark results.
- [`dependencies/`](dependencies) - CUTLASS required to compile the W4A16 kernels.

The code is organized as a dependency chain: `lrsr` (math) is consumed by `e2e` (inference), which in turn powers `benchmarks`.

## Requirements

- Python 3.12+
- ~10 GB of RAM or VRAM
- A CUDA-capable NVIDIA GPU with a recent driver
- CUDA Toolkit and a C++ compiler

## Usage

### Setup

1. Clone the repository with submodules:

```
git clone --recurse-submodules https://github.com/invisiofficial/lrsr
cd lrsr
```

2. Create a virtual environment:

```
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux
python -m venv .venv
source .venv/bin/activate
```

3. Install the common project dependencies.

```
pip install -r requirements.txt
```

4. Install PyTorch with CUDA version [XYZ] for acceleration.

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu[XYZ] --force-reinstall
```

### Benchmarks

Install the benchmark dependencies once:

```
pip install -r benchmarks/requirements.txt
```

All benchmarks are launched as modules: `python -m benchmarks.[block].setup` and `python -m benchmarks.[block].run`. Results are saved to `benchmarks/[block]/result/results.json`.

#### 1. Layer benchmark - [`benchmarks/layer`](benchmarks/layer)

Download the target models and captures `W`/`A` tensors:
```
python -m benchmarks.layer.setup
```

| Flag | Default | Description |
|---|---|---|
| `--device` | `cpu` | Device for model hooks: `cpu`, `cuda`, or `auto`. |
| `--yes` | - | Skip the confirmation prompt. |

Evaluate Normalized MSE and Gap Closed on every captured file:
```
python -m benchmarks.layer.run
```

| Flag | Default | Description |
|---|---|---|
| `--device` | `cpu` | Device for the evaluation: `cpu`, `cuda`, or `auto`. |
| `--limit` | - | Evaluate at most `N` dataset files. |

#### 2. Logits benchmark - [`benchmarks/logits`](benchmarks/logits)

Download the WikiText-2 test split and extract `wiki.test.raw`:
```
python -m benchmarks.logits.setup
```

| Flag | Default | Description |
|---|---|---|
| `--yes` | - | Skip the confirmation prompt. |

Compute KL-Divergence, Top-1 token agreement, etc. per method, model, and bit-width:
```
python -m benchmarks.logits.run
```

| Flag | Default | Description |
|---|---|---|
| `--device` | `cpu` | Device for the evaluation: `cpu`, `cuda`, or `auto`. |
| `--models` | `Qwen/Qwen3-1.7B Qwen/Qwen3-4B` | Models to evaluate. |
| `--bits` | `4 8` | Bit widths to evaluate. |
| `--methods` | `per-channel per-group lrsr-naive lrsr-1dos lrsr-kmeans` | Quantization methods to evaluate. |
| `--chunks` | `64` | Number of chunks. |
| `--batches` | `512` | Tokens per chunk. |

#### 3. Runtime benchmark - [`benchmarks/runtime`](benchmarks/runtime)

Compile the W4A16 CUTLASS backend into a shared library:
```
python -m benchmarks.runtime.setup
```

| Flag | Default | Description |
|---|---|---|
| `--arch` | `75 80 86 89 90 120` | CUDA SM targets. |
| `--yes` | - | Skip the confirmation prompt. |

Profile prefill, decode, e2e latency and peak memory:
```
python -m benchmarks.runtime.run
```

| Flag | Default | Description |
|---|---|---|
| `--device` | `cuda:0` | CUDA device to run on. |
| `--models` | `Qwen/Qwen3-1.7B Qwen/Qwen3-4B` | Models to profile. |
| `--methods` | `per-channel per-group lrsr-naive lrsr-1dos lrsr-kmeans` | Quantization methods to profile. |
| `--prompt-length` | `4096` | Prompt length for the prefill phase. |
| `--decode-tokens` | `128` | Number of decode tokens. |
| `--warmup` | `3` | Number of warmup trials. |
| `--repeats` | `10` | Number of timed trials. |

### Visualization

Install visualization requirements:
```
pip install -r notebooks/requirements.txt
```

For basic concept visualization `benchmarks.layer.setup` is enough. For in-depth analysis all benchmarks (`benchmarks.layer.run`, `benchmarks.logits.run` and `benchmarks.runtime.run`) should be executed.

Figures are saved as PDFs into `notebooks/figures/`.

## Citation

If you use LRSR in your research, please cite the paper:

```bibtex
@misc{kashtanov2026lrsr,
  author = {Kashtanov, Ivan},
  title  = {LRSR: Hardware-Friendly Approximation of Per-Element Quantization via Low-Rank Scale Reconstruction},
  year   = {2026},
  doi    = {10.5281/zenodo.22976487},
  url    = {https://doi.org/10.5281/zenodo.22976487}
}
```
