import math
import json
import torch
import argparse
import statistics
import transformers
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from e2e.layer import QuantizedLinear
from e2e.patcher import replace_layers


#region Configuration

RESULT = Path(__file__).resolve().parent / "result" / "results.json"

DEVICE = "cuda:0"

MODELS = ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
METHODS = ["per-channel", "per-group", "lrsr-naive"]
PHASES = ["prefill", "decode", "e2e"]

PREFILL = 4096
DECODE = 128
WARMUP = 3
REPEATS = 10

TARGET_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

GB = 1024 ** 3
MB = 1024 ** 2
KB = 1024

#endregion

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
    models: List[str],
    methods: List[str],
    prompt_length: int,
    decode_tokens: int,
    warmup: int,
    repeats: int,
    sizes: Dict[str, Optional[int]]
) -> None:
    """Prints the benchmark plan."""
    width = max(len(m) for m in models)
    print(f"\nRuntime benchmark plan")
    print(f"  Results file:     {RESULT}")
    print(f"  Computing device: {device}")
    print(f"  Target models:    {len(models)}")
    for model_id in models:
        size = sizes.get(model_id)
        print(f"   - {model_id:<{width}}   {format_bytes(size) if size else 'unknown'}")
    print(f"  Target methods:   {', '.join(methods)}")
    print(f"  Workload:")
    print(f"   - Prompt length: {prompt_length}")
    print(f"   - Decode tokens: {decode_tokens}")
    print(f"   - Warmup trials: {warmup}")
    print(f"   - Timed trials:  {repeats}")

#region Helpers

def distribution(values: List[float]) -> dict:
    """Computes full statistical distribution for a set of measurements."""
    if not values:
        return {}
    ordered = sorted(values)
    def percentile(p):
        idx = (len(ordered) - 1) * p
        low, high = math.floor(idx), math.ceil(idx)
        return ordered[low] + (ordered[high] - ordered[low]) * (idx - low)

    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "p05": float(percentile(0.05)),
        "p95": float(percentile(0.95))
    }

def measure_phase(
    prefill_graph: torch.cuda.CUDAGraph,
    decode_graph: torch.cuda.CUDAGraph,
    phase: str,
    decode_tokens: int,
    device: torch.device
) -> Tuple[float, float]:
    """Measures time and memory for a specific phase using static CUDA Graphs."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    if phase == "prefill":
        prefill_graph.replay()
    elif phase == "decode":
        for _ in range(decode_tokens):
            decode_graph.replay()
    elif phase == "e2e":
        prefill_graph.replay()
        for _ in range(decode_tokens):
            decode_graph.replay()

    end_event.record()
    end_event.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    peak_mem = torch.cuda.max_memory_allocated(device)

    return elapsed_ms, peak_mem

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the runtime performance benchmark.")
    parser.add_argument("--device", default=DEVICE, help="CUDA device to run on")
    parser.add_argument("--models", nargs="*", default=MODELS, help="Models to evaluate")
    parser.add_argument("--methods", nargs="*", default=METHODS, help="Quantization methods to evaluate")
    parser.add_argument("--prompt-length", type=int, default=PREFILL, help="Prompt length for the prefill phase")
    parser.add_argument("--decode-tokens", type=int, default=DECODE, help="Number of decode tokens to generate")
    parser.add_argument("--warmup", type=int, default=WARMUP, help="Number of warmup trials")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="Number of timed trials")
    return parser.parse_args()

def table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    body = "\n  ".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    return "  " + line + "\n  " + body

def main() -> None:
    args = parse()
    device = torch.device(args.device)

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Error: Runtime benchmark requires a CUDA device.")
    torch.cuda.set_device(device)

    print("Querying model sizes...")
    sizes = {m: get_size(m) for m in args.models}
    show_plan(args.device, args.models, args.methods, args.prompt_length, args.decode_tokens, args.warmup, args.repeats, sizes)

    RESULT.parent.mkdir(parents=True, exist_ok=True)

    out_data = {
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "cuda": torch.version.cuda
        },
        "config": {
            "prompt_length": args.prompt_length,
            "decode_tokens": args.decode_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats
        },
        "records": []
    }

    if RESULT.exists():
        with open(RESULT, "r") as f:
            existing_data = json.load(f)
            out_data["records"] = existing_data.get("records", [])

    records = out_data["records"]

    print("\nRunning benchmark...")

    for model_id in args.models:
        print(f"\n[{model_id}]")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        vocab_size = tokenizer.vocab_size

        for method in args.methods:
            print(f"  Processing {method}...")

            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map="cpu", trust_remote_code=True
            ).eval()

            replace_layers(model, method, TARGET_LAYERS, bits=4, group_size=128)
            model = model.to(device)

            for _, module in model.named_modules():
                if isinstance(module, QuantizedLinear):
                    module.enable_cutlass()

            generator = torch.Generator(device=device).manual_seed(0)
            prompt_ids = torch.randint(0, vocab_size, (1, args.prompt_length), device=device, generator=generator)
            decode_id = torch.randint(0, vocab_size, (1, 1), device=device, generator=generator)

            with torch.inference_mode():
                # Warmup allocator
                for _ in range(args.warmup):
                    model(input_ids=prompt_ids, use_cache=False)
                    model(input_ids=decode_id, use_cache=False)
                torch.cuda.synchronize(device)

                # Initialize graphs
                prefill_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(prefill_graph):
                    model(input_ids=prompt_ids, use_cache=False)

                decode_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(decode_graph):
                    model(input_ids=decode_id, use_cache=False)

                # Bench phases
                for phase in PHASES:
                    print(f"    Benchmarking {phase}...")

                    # Warmup execution
                    for _ in range(args.warmup):
                        measure_phase(prefill_graph, decode_graph, phase, args.decode_tokens, device)

                    # Measure
                    times_ms = []
                    peaks_bytes = []
                    for _ in range(args.repeats):
                        ms, peak = measure_phase(prefill_graph, decode_graph, phase, args.decode_tokens, device)
                        times_ms.append(ms)
                        peaks_bytes.append(peak)

                    time_stats = distribution(times_ms)
                    memory_bytes = max(peaks_bytes)

                    print(f"      {time_stats['median']:.2f} ms, {memory_bytes / MB:.1f} mb")

                    existing_idx = next((i for i, r in enumerate(records)
                                         if r["model"] == model_id and r["method"] == method and r["phase"] == phase), None)

                    # Save record
                    record_entry = {
                        "model": model_id,
                        "method": method,
                        "phase": phase,
                        "time_ms": time_stats,
                        "memory_b": memory_bytes
                    }

                    if existing_idx is not None:
                        records[existing_idx] = record_entry
                    else:
                        records.append(record_entry)

                    out_data["records"] = records
                    with open(RESULT, "w") as f:
                        json.dump(out_data, f, indent=2)

            del prefill_graph
            del decode_graph
            del model
            torch.cuda.empty_cache()

    print(f"\nBenchmark finished: {len(records)} records.")

    if records:
        print(f"\nResults")
        table_rows = [
            [r["model"].split("/")[-1], r["method"], r["phase"],
             f"{r['time_ms']['median']:.2f}",
             f"{r['time_ms']['std']:.2f}",
             f"{r['time_ms']['p95']:.2f}",
             f"{r['memory_b'] / MB:.1f}"]
            for r in records
        ]
        headers = ["model", "method", "phase", "median (ms)", "std (ms)", "p95 (ms)", "memory (MB)"]
        print(f"{table(headers, table_rows)}")

    print(f"\nResults saved to {RESULT}")

if __name__ == "__main__":
    main()
