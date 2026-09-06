import gc
import torch
import argparse
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


#region Configuration

DIR = Path(__file__).resolve().parent / "data"

DEVICE = "cpu"  # "cpu" | "cuda" | "auto"

# Models to process
MODELS = [
    "Qwen/Qwen3-0.6B",
    "openbmb/MiniCPM5-1B",
    "HuggingFaceTB/SmolLM3-3B",
    "google/gemma-4-E2B-it",
    "allenai/OLMoE-1B-7B-0125-Instruct",
]

# Layers to put hook on
LAYERS = [
    "layers.0.mlp.down_proj",
    "layers.0.mlp.gate_proj",
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.k_proj",
    "layers.0.self_attn.o_proj",
]

# Prompts for capturing representative activations
PROMPTS = {
    "wiki": "The quick brown fox jumps over the lazy dog. In computer science, quantization is a method to map continuous values to discrete ones.",
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\nprint(fibonacci(10))",
    "math": "If Mary has 3 apples and gives 1 to John, she has 2 left. The integral of x^2 is x^3/3 + C. Solve for x: 2x + 4 = 10.",
    "inst": "Explain the process of photosynthesis in simple terms, as if to a 10-year-old.",
    "poem": "Roses are red, violets are blue,\nQuantization is hard,\nbut this test will do.",
}

DTYPE = torch.bfloat16

GB = 1024 ** 3
MB = 1024 ** 2
KB = 1024

#endregion

#region Planning

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

def show_plan(sizes: Dict[str, Optional[int]], device: str) -> None:
    """Prints the dataset generation plan."""
    total = sum(s or 0 for s in sizes.values())
    files = len(MODELS) * len(LAYERS) * len(PROMPTS)
    width = max(len(m) for m in MODELS)
    note = "  (CUDA is available; pass --device cuda to enable it)" if device == "cpu" and torch.cuda.is_available() else ""

    print(f"\nDataset generation plan")
    print(f"  Computing device: {device}{note}")
    print(f"  Target models: {len(MODELS)}")
    for model_id in MODELS:
        size = sizes.get(model_id)
        print(f"   - {model_id:<{width}}   {format_bytes(size) if size else 'unknown'}")
    if total:
        print(f"  Total download:   ~{format_bytes(total)}")
    else:
        print(f"  Total download:   unknown")
    print(f"  Data directory:   {DIR}")
    print(f"  Layer files:      up to {files} ({len(MODELS)} models x {len(LAYERS)} layers x {len(PROMPTS)} prompts)")
    print(f"  Data size:        up to ~{format_bytes(total / 10)}")
    print(f"  Existing files there will be overwritten")

#endregion

#region Capturing

def layer_hook(name: str, store: Dict[str, Dict[str, torch.Tensor]]) -> Callable:
    """Forward hook capturing activations A [tokens, in] and weights W [in, out]."""
    def fn(module: torch.nn.Module, inputs: Tuple, output) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        a = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
        w = module.weight.detach().t().cpu()
        store[name] = {"A": a, "W": w}
    return fn

def clean_memory() -> None:
    """Utility for cleaning memory of the process."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def run_model(model_id: str, device: str) -> None:
    """Loads model, puts hooks on layers, runs prompts, captures W and A, saves files."""
    print(f"\n[{model_id}]")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=DTYPE,
        device_map="cpu" if device == "cpu" else "auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    store: Dict[str, Dict[str, torch.Tensor]] = {}
    targets: Dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(p in name for p in LAYERS):
            targets[name] = module
            print(f"  Hook installed on: {name}")
    if not targets:
        print(f"Warning: no linear layers matched target layers. Skipping...")
        return

    hooks = [m.register_forward_hook(layer_hook(n, store)) for n, m in targets.items()]

    try:
        files = 0
        n_bytes = 0
        for name, text in PROMPTS.items():
            print(f"  Running prompt '{name}' ({len(text)} chars)...")
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            store.clear()
            with torch.inference_mode():
                model(**inputs)
            for layer, t in store.items():
                w, a = t["W"].to(DTYPE), t["A"].to(DTYPE)
                path = DIR / f"{model_id.replace('/', '_')}__{layer}__{name}.pt"
                torch.save({"model": model_id, "layer": layer, "prompt": name, "W": w, "A": a}, path)
                files += 1
                n_bytes += w.numel() + a.numel()
        print(f"  Saved {files} files (~{format_bytes(n_bytes * 2)})")
    finally:
        for h in hooks:
            h.remove()
        del model, tokenizer
        clean_memory()

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates the layer dataset.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=DEVICE)
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    return parser.parse_args()

def pick(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Error: CUDA is not available on this machine.")
    return value

def confirm(auto: bool) -> bool:
    if auto:
        return True
    try:
        return input("\nProceed? [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        return False

def main() -> None:
    args = parse()
    device = pick(args.device)

    print("Querying model sizes...")
    sizes = {m: get_size(m) for m in MODELS}
    show_plan(sizes, device)
    if not confirm(args.yes):
        print("Aborted.")
        return

    DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for model_id in MODELS:
        try:
            run_model(model_id, device)
        except Exception as e:
            print(f"Error: {e}")
            failed.append(model_id)
            clean_memory()

    print(f"\nGeneration finished: {len(MODELS) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
