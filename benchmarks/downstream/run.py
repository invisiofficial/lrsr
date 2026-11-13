import gc
import re
import sys
import json
import torch
import hashlib
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from e2e.patcher import replace_layers


#region Configuration

DIR = Path(__file__).resolve().parent / "data"
RESULT = Path(__file__).resolve().parent / "result" / "results.json"

TASKS: Dict[str, Path] = {
    "mmlu_pro": DIR / "mmlu_pro.jsonl",
    "arc_challenge": DIR / "arc_challenge.jsonl",
    "winogrande": DIR / "winogrande.jsonl",
    "gsm8k": DIR / "gsm8k.jsonl",
    "humaneval": DIR / "humaneval.jsonl",
}

DEVICE = "cpu"  # "cpu" | "cuda" | "auto"

MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "LiquidAI/LFM2.5-2.6B",
]
METHODS = ["bf16", "per-channel", "per-group", "lrsr-naive"]
BITS = 4

MMLU_SAMPLES = 128
ARC_SAMPLES = 2048
WINOGRANDE_SAMPLES = 2048
GSM8K_SAMPLES = 256
HUMANEVAL_SAMPLES = 164

BATCH_SIZE = 4
MMLU_MAX_TOKENS = 2048
GSM8K_MAX_TOKENS = 256
HUMANEVAL_MAX_TOKENS = 1024
GROUP = 128

TARGET_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

MMLU_STOP_SEQS = ("\nQuestion:", "Question:")
GSM8K_STOP_SEQS = ("\nQuestion:", "Question:")
HUMANEVAL_STOP_SEQS = ("\nclass ", "\ndef ", "\n#", "\nif ", "\nprint(", "\n```")

# Full 8-shot Chain-of-Thought prompt
GSM8K_FEWSHOT = (
    "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
    "After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
    "Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. "
    "So there must have been 21 - 15 = 6. #### 6\n\n"
    "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
    "Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5\n\n"
    "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
    "Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
    "After eating 35, they had 74 - 35 = 39. #### 39\n\n"
    "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
    "How many lollipops did Jason give to Denny?\n"
    "Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
    "So he gave Denny 20 - 12 = 8. #### 8\n\n"
    "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
    "How many toys does he have now?\n"
    "Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
    "5 + 4 = 9. #### 9\n\n"
    "Question: There were nine computers in the server room. Five more computers were installed each day, "
    "from monday to thursday. How many computers are now in the server room?\n"
    "Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. "
    "So 5 * 4 = 20 computers were added. 9 + 20 is 29. #### 29\n\n"
    "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
    "How many golf balls did he have at the end of wednesday?\n"
    "Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
    "After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33\n\n"
    "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n"
    "Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
    "So she has 23 - 15 = 8 dollars left. #### 8\n\n"
)

GB = 1024 ** 3
MB = 1024 ** 2
KB = 1024

#endregion

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, streamed in chunks."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

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
    task_hashes: Dict[str, str],
    task_counts: Dict[str, int],
    models: List[str],
    methods: List[str],
    bits: int,
    batch_size: int,
    sizes: Dict[str, Optional[int]],
) -> None:
    """Prints the benchmark plan."""
    width = max(len(m) for m in models)
    print(f"\nDownstream benchmark plan")
    print(f"  Downstream tasks:")
    for t_name, t_hash in task_hashes.items():
        print(f"   - {TASKS[t_name].name} ({task_counts[t_name]} samples, SHA256: {t_hash[:16]}...)")
    print(f"  Results file:         {RESULT}")
    print(f"  Computing device:     {device}")
    print(f"  Batch size:           {batch_size}")
    print(f"  Target models:        {len(models)}")
    for model_id in models:
        size = sizes.get(model_id)
        print(f"   - {model_id:<{width}}   {format_bytes(size) if size else 'unknown'}")
    print(f"  Quantization:")
    print(f"   - Bits: {bits}")
    print(f"   - Methods: {methods}")

#region Helpers

def load_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if 0 < limit <= len(items):
                break
    return items


def _truncate_at_stop(text: str, stop_seqs: Tuple[str, ...]) -> str:
    """Truncates decoded generation at the earliest occurrence of any stop string."""
    earliest = len(text)
    for s in stop_seqs:
        idx = text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


class PerSequenceStopCriteria(StoppingCriteria):
    """Tracks exact per-sequence stop positions."""

    def __init__(self, prompt_len: int, tokenizer: Any, mode: str, window_tokens: int = 24) -> None:
        super().__init__()
        self.prompt_len = prompt_len
        self.tokenizer = tokenizer
        self.mode = mode
        self.window_tokens = window_tokens
        self.done: Optional[torch.Tensor] = None
        self.stop_len: List[Optional[int]] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        bsz, cur_len = input_ids.shape
        if self.done is None or self.done.size(0) != bsz:
            self.done = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
            self.stop_len = [None] * bsz

        if cur_len - self.prompt_len < 4:
            return False

        win_start = max(self.prompt_len, cur_len - self.window_tokens)
        for i in range(bsz):
            if self.done[i]:
                continue
            tail_text = self.tokenizer.decode(input_ids[i, win_start:cur_len], skip_special_tokens=True)
            if self.mode == "mmlu_pro":
                if any(s in tail_text for s in MMLU_STOP_SEQS) or re.search(
                    r"answer is \(?[A-J]\)?\.?\s*\n", tail_text, flags=re.IGNORECASE
                ):
                    self.done[i] = True
                    self.stop_len[i] = cur_len
            elif self.mode == "gsm8k":
                if any(s in tail_text for s in GSM8K_STOP_SEQS) or re.search(
                    r"####\s*-?\d+(?:,\d+)*(?:\.\d+)?\s*\n", tail_text
                ):
                    self.done[i] = True
                    self.stop_len[i] = cur_len
            elif self.mode == "humaneval":
                if any(s in tail_text for s in HUMANEVAL_STOP_SEQS):
                    self.done[i] = True
                    self.stop_len[i] = cur_len

        return bool(self.done.all().item())


@torch.inference_mode()
def _score_continuations(
    model: torch.nn.Module,
    tokenizer: Any,
    pairs: List[Tuple[str, str]],
    batch_size: int,
    device: str,
    desc: str,
) -> Tuple[List[float], List[float]]:
    """Computes conditional log-likelihood log P(continuation | context)."""
    bos_prefix = [tokenizer.bos_token_id] if getattr(tokenizer, "bos_token_id", None) is not None else []
    encoded_spans: List[Tuple[List[int], int, int, int]] = []

    for ctx, cont in pairs:
        ctx_raw = tokenizer.encode(ctx, add_special_tokens=False)
        full_raw = tokenizer.encode(ctx + cont, add_special_tokens=False)
        if len(full_raw) <= len(ctx_raw):
            cont_raw = tokenizer.encode(cont, add_special_tokens=False)
            full_raw = ctx_raw + cont_raw

        if bos_prefix and (not ctx_raw or ctx_raw[0] != bos_prefix[0]):
            ctx_ids = bos_prefix + ctx_raw
            full_ids = bos_prefix + full_raw
        else:
            ctx_ids = ctx_raw
            full_ids = full_raw

        c_len = max(1, len(ctx_ids))
        f_len = len(full_ids)
        char_len = max(1, len(cont.strip()))
        encoded_spans.append((full_ids, c_len, f_len, char_len))

    sum_ll: List[float] = []
    norm_ll: List[float] = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pair_batch = max(16, batch_size * 4)

    for start in tqdm(range(0, len(encoded_spans), pair_batch), desc=desc, leave=False):
        chunk = encoded_spans[start : start + pair_batch]
        max_len = max(x[2] for x in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)

        for i, (f_ids, _, f_len, _) in enumerate(chunk):
            input_ids[i, :f_len] = torch.tensor(f_ids, dtype=torch.long, device=device)
            attn_mask[i, :f_len] = 1

        logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
        for i, (_, c_len, f_len, char_len) in enumerate(chunk):
            slice_logits = logits[i, c_len - 1 : f_len - 1, :].float()
            target_ids = input_ids[i, c_len:f_len]
            log_probs = F.log_softmax(slice_logits, dim=-1)
            token_ll = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            total = float(token_ll.sum().item())
            sum_ll.append(total)
            norm_ll.append(total / char_len)
        del input_ids, attn_mask, logits

    return sum_ll, norm_ll


def _extract_mmlu_pro_answer(text: str) -> Optional[str]:
    """Official 3-stage regex answer extractor."""
    text = _truncate_at_stop(text, MMLU_STOP_SEQS)
    match = re.search(r"answer is \(?([A-J])\)?", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match2 = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match2:
        return match2.group(1).upper()
    match3 = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, flags=re.DOTALL)
    if match3:
        return match3.group(0).upper()
    return None


@torch.inference_mode()
def evaluate_mmlu_pro(
    model: torch.nn.Module,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    batch_size: int,
    device: str,
) -> List[Optional[str]]:
    """Evaluates MMLU-Pro using the official 5-shot Chain-of-Thought generation protocol."""
    prompts = [item["prompt"] for item in items]
    predictions: List[Optional[str]] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="    Eval [mmlu_pro]", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        prompt_len = enc.input_ids.size(1)
        stopper = PerSequenceStopCriteria(prompt_len, tokenizer, mode="mmlu_pro")

        out = model.generate(
            **enc,
            max_new_tokens=MMLU_MAX_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        for i in range(out.size(0)):
            end_idx = stopper.stop_len[i] if stopper.stop_len[i] is not None else out.size(1)
            gen_text = tokenizer.decode(out[i, prompt_len:end_idx], skip_special_tokens=True)
            predictions.append(_extract_mmlu_pro_answer(gen_text))
        del enc, out

    return predictions


@torch.inference_mode()
def evaluate_arc_challenge(
    model: torch.nn.Module,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    batch_size: int,
    device: str,
) -> List[int]:
    """Evaluates ARC-Challenge via length-normalized continuation log-likelihood."""
    pairs: List[Tuple[str, str]] = []
    counts: List[int] = []
    for item in items:
        ctx = f"Question: {item['question']}\nAnswer:"
        choices = item["choices"]
        counts.append(len(choices))
        for ch in choices:
            pairs.append((ctx, f" {ch}"))

    _, norm_ll = _score_continuations(
        model, tokenizer, pairs, batch_size, device, desc="    Eval [arc_challenge]"
    )

    predictions: List[int] = []
    offset = 0
    for c_count in counts:
        scores = norm_ll[offset : offset + c_count]
        pred_idx = int(max(range(c_count), key=lambda k: scores[k]))
        predictions.append(pred_idx)
        offset += c_count

    return predictions


@torch.inference_mode()
def evaluate_winogrande(
    model: torch.nn.Module,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    batch_size: int,
    device: str,
) -> List[int]:
    """Evaluates WinoGrande via suffix log-likelihood conditioned on each option."""
    pairs: List[Tuple[str, str]] = []
    for item in items:
        sent = item["sentence"]
        idx = sent.index("_")
        prefix = sent[:idx]
        suffix = " " + sent[idx + 1 :].strip()
        pairs.append((prefix + item["option1"], suffix))
        pairs.append((prefix + item["option2"], suffix))

    sum_ll, _ = _score_continuations(
        model, tokenizer, pairs, batch_size, device, desc="    Eval [winogrande]"
    )

    predictions: List[int] = []
    for i in range(0, len(sum_ll), 2):
        predictions.append(0 if sum_ll[i] >= sum_ll[i + 1] else 1)

    return predictions


def _extract_gsm8k_number(text: str, strict: bool = False) -> Optional[str]:
    """Extracts answer number from GSM8K completion."""
    text = _truncate_at_stop(text, GSM8K_STOP_SEQS)
    match = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        val = match.group(1).replace(",", "").strip()
    elif strict:
        return None
    else:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        if not nums:
            return None
        val = nums[-1].replace(",", "").strip()
    return val.rstrip(".")


@torch.inference_mode()
def evaluate_gsm8k(
    model: torch.nn.Module,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    batch_size: int,
    device: str,
) -> List[Optional[str]]:
    """Evaluates GSM8K using 8-shot Chain-of-Thought greedy decoding."""
    prompts = [f"{GSM8K_FEWSHOT}Question: {item['question'].strip()}\nAnswer:" for item in items]
    predictions: List[Optional[str]] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="    Eval [gsm8k]", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        prompt_len = enc.input_ids.size(1)
        stopper = PerSequenceStopCriteria(prompt_len, tokenizer, mode="gsm8k")

        out = model.generate(
            **enc,
            max_new_tokens=GSM8K_MAX_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        for i in range(out.size(0)):
            end_idx = stopper.stop_len[i] if stopper.stop_len[i] is not None else out.size(1)
            gen_text = tokenizer.decode(out[i, prompt_len:end_idx], skip_special_tokens=True)
            predictions.append(_extract_gsm8k_number(gen_text, strict=False))
        del enc, out

    return predictions


def _clean_humaneval_completion(completion: str, entry_point: str) -> str:
    """Truncates completion at the first top-level stop sequence."""
    if "```" in completion:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", completion, flags=re.DOTALL)
        if blocks:
            code = blocks[0]
            def_match = re.search(rf"def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*(?:->[^:]+)?:\s*\n", code)
            if def_match:
                completion = code[def_match.end() :]
            else:
                completion = code

    truncated = _truncate_at_stop(completion, HUMANEVAL_STOP_SEQS)
    return truncated.rstrip() + "\n"


def _check_humaneval_item(args: Tuple[Dict[str, Any], str]) -> bool:
    """Executes a single HumanEval candidate against its unit tests in an isolated subprocess."""
    item, completion = args
    body = _clean_humaneval_completion(completion, item["entry_point"])
    program = f"{item['prompt']}{body}\n{item['test']}\ncheck({item['entry_point']})\n"
    try:
        res = subprocess.run(
            [sys.executable, "-I", "-c", program],
            capture_output=True,
            timeout=3.0,
            text=True,
        )
        return res.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


@torch.inference_mode()
def evaluate_humaneval(
    model: torch.nn.Module,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    batch_size: int,
    device: str,
) -> List[bool]:
    """Evaluates HumanEval functional correctness (pass@1)."""
    prompts = [item["prompt"] for item in items]
    completions: List[str] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="    Eval [humaneval]", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        prompt_len = enc.input_ids.size(1)
        stopper = PerSequenceStopCriteria(prompt_len, tokenizer, mode="humaneval")

        out = model.generate(
            **enc,
            max_new_tokens=HUMANEVAL_MAX_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        for i in range(out.size(0)):
            end_idx = stopper.stop_len[i] if stopper.stop_len[i] is not None else out.size(1)
            gen_text = tokenizer.decode(out[i, prompt_len:end_idx], skip_special_tokens=True)
            completions.append(gen_text)
        del enc, out

    with ThreadPoolExecutor(max_workers=8) as pool:
        passed = list(pool.map(_check_humaneval_item, zip(items, completions)))
    return passed

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the downstream tasks benchmark.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=DEVICE)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS.keys()), choices=list(TASKS.keys()))
    parser.add_argument("--models", nargs="*", default=MODELS, help="Models to evaluate")
    parser.add_argument("--methods", nargs="*", default=METHODS, help="Methods to evaluate")
    parser.add_argument("--bits", type=int, default=BITS, help="Bit width for quantized methods")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Evaluation batch size")
    parser.add_argument("--mmlu-samples", type=int, default=MMLU_SAMPLES, help="Max MMLU-Pro samples")
    parser.add_argument("--arc-samples", type=int, default=ARC_SAMPLES, help="Max ARC-Challenge samples")
    parser.add_argument("--winogrande-samples", type=int, default=WINOGRANDE_SAMPLES, help="Max WinoGrande samples")
    parser.add_argument("--gsm8k-samples", type=int, default=GSM8K_SAMPLES, help="Max GSM8K samples")
    parser.add_argument("--humaneval-samples", type=int, default=HUMANEVAL_SAMPLES, help="Max HumanEval samples")
    return parser.parse_args()

def pick(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Error: CUDA is not available on this machine.")
    return value

def table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    body = "\n  ".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)
    return "  " + line + "\n  " + body

def main() -> None:
    args = parse()
    device = pick(args.device)

    limits = {
        "mmlu_pro": args.mmlu_samples,
        "arc_challenge": args.arc_samples,
        "winogrande": args.winogrande_samples,
        "gsm8k": args.gsm8k_samples,
        "humaneval": args.humaneval_samples,
    }

    task_hashes: Dict[str, str] = {}
    task_data: Dict[str, List[Dict[str, Any]]] = {}
    for t_name in args.tasks:
        t_path = TASKS[t_name]
        t_hash = sha256_file(t_path)
        if not t_hash:
            raise FileNotFoundError(f"Dataset '{t_name}' not found at {t_path}. Run setup.py first.")
        task_hashes[t_name] = t_hash
        task_data[t_name] = load_jsonl(t_path, limits[t_name])
    task_counts = {t: len(task_data[t]) for t in args.tasks}

    print("Querying model sizes...")
    sizes = {m: get_size(m) for m in args.models}
    show_plan(device, task_hashes, task_counts, args.models, args.methods, args.bits, args.batch_size, sizes)

    config = {
        "sha256": task_hashes,
        "samples": task_counts,
        "bits": args.bits,
        "groups": GROUP,
    }

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    out_data = {"config": config, "records": []}
    if RESULT.exists():
        with open(RESULT, "r", encoding="utf-8") as f:
            out_data = json.load(f)
        out_data["config"] = config

    records: List[Dict[str, Any]] = out_data.get("records", [])

    print("\nRunning benchmark...")

    for model_id in args.models:
        print(f"\n[{model_id}]")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bf16_preds: Dict[str, List[Any]] = {}
        bf16_scores: Dict[str, float] = {}

        methods_to_run = list(args.methods)
        if "bf16" not in methods_to_run:
            methods_to_run.insert(0, "bf16")

        for method in methods_to_run:
            eff_bits = 16 if method == "bf16" else args.bits
            missing_tasks = [
                t_name for t_name in args.tasks
                if not any(
                    r["task"] == t_name
                    and r["model"] == model_id
                    and r["method"] == method
                    and r["bits"] == eff_bits
                    and r["metrics"].get("total") == task_counts[t_name]
                    for r in records
                )
            ]

            need_bf16_ref = (
                method == "bf16"
                and not missing_tasks
                and any(
                    not any(
                        r["task"] == t_name
                        and r["model"] == model_id
                        and r["method"] == m
                        and r["bits"] == args.bits
                        and r["metrics"].get("total") == task_counts[t_name]
                        for r in records
                    )
                    for m in args.methods if m != "bf16"
                    for t_name in args.tasks
                )
            )

            if not missing_tasks and not need_bf16_ref:
                if method in args.methods:
                    print(f"  [Skipped] {method} ({'BF16' if eff_bits == 16 else f'INT{eff_bits}'})")
                continue

            eval_tasks = args.tasks if method == "bf16" else missing_tasks
            print(f"\n  Processing {method} ({'BF16' if eff_bits == 16 else f'INT{eff_bits}'})...")

            if method == "bf16":
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    device_map="auto" if device == "cuda" else "cpu",
                    trust_remote_code=True,
                ).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    device_map="cpu",
                    trust_remote_code=True,
                ).eval()
                replace_layers(
                    model,
                    method,
                    TARGET_LAYERS,
                    bits=args.bits,
                    group_size=GROUP,
                )
                model = model.to(device).eval()

            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None

            for t_name in eval_tasks:
                items = task_data[t_name]
                if t_name == "mmlu_pro":
                    preds = evaluate_mmlu_pro(model, tokenizer, items, args.batch_size, device)
                    golds = [it["answer"] for it in items]
                    correct_flags = [(p is not None and p == g) for p, g in zip(preds, golds)]
                elif t_name == "arc_challenge":
                    preds = evaluate_arc_challenge(model, tokenizer, items, args.batch_size, device)
                    golds = [int(it["answer_idx"]) for it in items]
                    correct_flags = [p == g for p, g in zip(preds, golds)]
                elif t_name == "winogrande":
                    preds = evaluate_winogrande(model, tokenizer, items, args.batch_size, device)
                    golds = [int(it["answer_idx"]) for it in items]
                    correct_flags = [p == g for p, g in zip(preds, golds)]
                elif t_name == "gsm8k":
                    preds = evaluate_gsm8k(model, tokenizer, items, args.batch_size, device)
                    golds = [_extract_gsm8k_number(it["answer"], strict=True) for it in items]
                    correct_flags = [(p is not None and p == g) for p, g in zip(preds, golds)]
                elif t_name == "humaneval":
                    preds = evaluate_humaneval(model, tokenizer, items, args.batch_size, device)
                    correct_flags = list(preds)

                correct_count = int(sum(correct_flags))
                total_count = len(items)
                score = (correct_count / max(1, total_count)) * 100.0

                if method == "bf16":
                    bf16_preds[t_name] = preds
                    bf16_scores[t_name] = score

                base_score = bf16_scores.get(t_name)
                delta_score = (score - base_score) if base_score is not None else 0.0

                if t_name in bf16_preds and len(bf16_preds[t_name]) == len(preds):
                    agree_bf16: Optional[float] = (
                        sum(p == b for p, b in zip(preds, bf16_preds[t_name])) / max(1, total_count)
                    ) * 100.0
                else:
                    agree_bf16 = 100.0 if method == "bf16" else None

                agree_str = f"{agree_bf16:.2f}%" if agree_bf16 is not None else "n/a"
                print(
                    f"      [{t_name}] score: {score:.2f}% ({correct_count}/{total_count}), "
                    f"delta: {delta_score:+.2f}%, agree: {agree_str}"
                )

                if method in args.methods and t_name in missing_tasks:
                    records.append({
                        "task": t_name,
                        "task_sha256": task_hashes[t_name],
                        "model": model_id,
                        "method": method,
                        "bits": eff_bits,
                        "metrics": {
                            "score": score,
                            "delta_score": delta_score,
                            "correct": correct_count,
                            "total": total_count,
                            "agree_bf16": agree_bf16,
                        },
                    })
                    out_data["records"] = records
                    with open(RESULT, "w", encoding="utf-8") as f:
                        json.dump(out_data, f, indent=2)

            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        del tokenizer

    print(f"\nBenchmark finished: {len(records)} records.")

    if records:
        print(f"\nResults")
        table_rows = [
            [
                r["task"],
                r["model"].split("/")[-1],
                r["method"],
                str(r["bits"]),
                f"{r['metrics']['score']:.2f}%",
                f"{r['metrics']['delta_score']:+.2f}%",
                f"{r['metrics']['correct']}/{r['metrics']['total']}",
                f"{r['metrics']['agree_bf16']:.2f}%" if r["metrics"].get("agree_bf16") is not None else "n/a",
            ]
            for r in records
        ]
        headers = ["task", "model", "method", "bits", "score", "delta_score", "solved", "agree_bf16"]
        print(f"{table(headers, table_rows)}")

    print(f"\nResults saved to {RESULT}")

if __name__ == "__main__":
    main()
