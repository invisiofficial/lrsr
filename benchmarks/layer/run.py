import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from lrsr.metrics import compute_pairwise_angular_distances, compute_median_angular_distances, mse
from lrsr.utils import compute_otsu_threshold
from lrsr.decomposition.alternating_decomposition import AlternatingDecomposition
from lrsr.approximation.rank_approximation import RankApproximation
from lrsr.clusterization.kmeans_clusterization import KMeansClusterization


#region Configuration

DIR = Path(__file__).resolve().parent / "data"
RESULT = Path(__file__).resolve().parent / "result" / "results.json"

DEVICE = "cpu"  # "cpu" | "cuda" | "auto"

BITS = 4
QMAX = float((2 ** (BITS - 1)) - 1)
GROUP = 128
CLUSTERS = 3
TOLERANCE = 0.05

METHODS = ["per-channel", "per-group", "lrsr-naive", "lrsr-1dos", "lrsr-kmeans"]
GAPS = ["lrsr-naive", "lrsr-1dos", "lrsr-kmeans"]

EPS = 1e-7

#endregion

approximator = RankApproximation(AlternatingDecomposition(num_iterations=20))

def show_plan(files: int, device: str) -> None:
    """Prints the benchmark plan."""
    print(f"\nLayer benchmark plan")
    print(f"  Dataset files:    {files}")
    print(f"  Results file:     {RESULT}")
    print(f"  Computing device: {device}")
    print(f"  Bit width:        {BITS}")
    print(f"  Individual parameters:")
    print(f"   - per-group: group_size={GROUP}")
    print(f"   - lrsr-kmeans: clusters={CLUSTERS}, tolerance={TOLERANCE}")

def summarize_method(records: List[dict]) -> List[dict]:
    """Aggregate error statistics per method."""
    rows = []
    for m in METHODS:
        v = [r["nmse"][m] for r in records]
        g = [compute_gap(r["nmse"], m) for r in records] if m in GAPS else None
        rows.append({
            "method": m,
            "count": len(v),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "gap": float(np.mean(g)) if g else None,
        })
    return rows

#region Quantization

def quantize_tensor(W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Quantizes W with scales S and dequantizes back to float."""
    return torch.clamp(torch.round(W / S), -QMAX, QMAX) * S

def scales_per_channel(W: torch.Tensor) -> torch.Tensor:
    """Per-channel scales: one per out-feature column."""
    return torch.clamp(W.abs().max(dim=0, keepdim=True)[0] / QMAX, min=EPS)

def scales_per_group(W: torch.Tensor) -> torch.Tensor:
    """Per-group scales over the in-feature axis, group size GROUP."""
    n, p = W.shape
    G = W.view(n // GROUP, GROUP, p)
    return torch.clamp(G.abs().max(dim=1)[0] / QMAX, min=EPS).repeat_interleave(GROUP, dim=0)

def scales_lrsr_naive(S: torch.Tensor) -> torch.Tensor:
    """One low-rank approximation of the full scale matrix."""
    return torch.clamp(approximator.approximate(S, rank=1), min=EPS)

def scales_lrsr_1dos(S: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Two low-rank blocks split by threshold on distance of out-feature columns."""
    dist = compute_median_angular_distances(S)
    tau = compute_otsu_threshold(dist.cpu().numpy())
    idx_minor = np.where(dist.cpu().numpy() > tau)[0]
    idx_main = np.where(dist.cpu().numpy() <= tau)[0]

    out = torch.zeros_like(S)
    if len(idx_main) > 0:
        out[:, idx_main] = torch.clamp(approximator.approximate(S[:, idx_main], rank=1), min=EPS)
    if len(idx_minor) > 0:
        out[:, idx_minor] = torch.clamp(approximator.approximate(S[:, idx_minor], rank=1), min=EPS)
    return out, len(idx_minor)

def scales_lrsr_kmeans(S: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """Constrained K-Means over out-feature columns, one low-rank per cluster."""
    model = KMeansClusterization(distance_fn=compute_pairwise_angular_distances, tolerance=TOLERANCE)
    model.fit(S)
    labels = model.predict(CLUSTERS)

    out = torch.zeros_like(S)
    sizes = []
    for c in range(CLUSTERS):
        idx = np.where(labels == c)[0]
        sizes.append(len(idx))
        if len(idx) == 0:
            continue
        out[:, idx] = torch.clamp(approximator.approximate(S[:, idx], rank=1), min=EPS)
    return out, sizes

def compute_nmse(A: torch.Tensor, W: torch.Tensor, S: torch.Tensor, Y: torch.Tensor) -> float:
    """Normalized MSE of the matmul: mse(A @ W_q, Y) / var(Y)."""
    Y_q = A @ quantize_tensor(W, S)
    return mse(Y, Y_q) / torch.var(Y).item()

def compute_gap(nm: Dict[str, float], m: str) -> float:
    """Relative gap closed of given method between per-channel and per-group."""
    total = nm["per-channel"] - nm["per-group"]
    return (nm["per-channel"] - nm[m]) / total * 100.0 if total > 0 else 0.0

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the layer benchmark.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=DEVICE)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N files")
    return parser.parse_args()

def pick(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Error: CUDA is not available on this machine.")
    return value

def scan() -> List[Path]:
    files = sorted(DIR.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {DIR}. Run setup.py first.")
    return files

def record(model: str, layer: str, prompt: str, shapes: Tuple[int, int],
           nm: Dict[str, float], n_minor: int, sizes: List[int]) -> dict:
    return {
        "model": model,
        "layer": layer,
        "prompt": prompt,
        "shape": {"in": shapes[0], "out": shapes[1]},
        "nmse": nm,
        "lrsr_1dos_minor": n_minor,
        "lrsr_kmeans_sizes": sizes,
    }

def table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    body = "\n  ".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    return "  " + line + "\n  " + body

def main() -> None:
    args = parse()
    device = pick(args.device)
    files = scan()
    if args.limit:
        files = files[: args.limit]

    show_plan(len(files), device)

    print(f"\nRunning benchmark...")
    records = []
    skipped = []
    for i, path in enumerate(files, 1):
        data = torch.load(path, weights_only=False)
        model, layer, prompt = data["model"], data["layer"], data["prompt"]

        W = data["W"].to(torch.float32)
        A = data["A"].to(torch.float32)
        if A.dim() == 3:
            A = A.view(-1, A.shape[-1])
        n_in, n_out = W.shape

        if n_in % GROUP != 0:
            print(f"  [{i}/{len(files)}] {path.name}: skipped (in_features {n_in} % {GROUP} != 0)")
            skipped.append(path.name)
            continue
        W = W.to(device)
        A = A.to(device)
        with torch.inference_mode():
            Y = A @ W
            S = torch.clamp(W.abs() / QMAX, min=EPS)

            nm = {
                "per-channel": compute_nmse(A, W, scales_per_channel(W), Y),
                "per-group": compute_nmse(A, W, scales_per_group(W), Y),
                "lrsr-naive": compute_nmse(A, W, scales_lrsr_naive(S), Y),
            }
            s_1d, n_minor = scales_lrsr_1dos(S)
            nm["lrsr-1dos"] = compute_nmse(A, W, s_1d, Y)
            s_km, sizes = scales_lrsr_kmeans(S)
            nm["lrsr-kmeans"] = compute_nmse(A, W, s_km, Y)

        print(f"  [{i}/{len(files)}] {model} | {layer} | {prompt}: "
              f"pc={nm['per-channel']:.4f} pg={nm['per-group']:.4f} "
              f"naive={nm['lrsr-naive']:.4f} 1d={nm['lrsr-1dos']:.4f} km={nm['lrsr-kmeans']:.4f}")

        records.append(record(model, layer, prompt, (n_in, n_out), nm, n_minor, sizes))
        del W, A, Y, S, s_1d, s_km
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nBenchmark finished: {len(records)} layers evaluated, {len(skipped)} skipped.")

    print(f"\nResults")
    rows = [
        [r["method"], str(r["count"]), f"{r['mean']:.4f}", f"{r['median']:.4f}",
         f"{r['std']:.4f}", f"{r['min']:.4f}", f"{r['max']:.4f}",
         f"{r['gap']:.3f}" if r["gap"] is not None else "-"]
        for r in summarize_method(records)
    ]
    print(f"{table(["method", "count", "mean", "median", "std", "min", "max", "gap"], rows)}")

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "bits": BITS,
            "groups": GROUP,
            "clusters": CLUSTERS,
            "tolerance": TOLERANCE,
            "files": len(records),
        },
        "files": records,
    }
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULT}")

if __name__ == "__main__":
    main()
