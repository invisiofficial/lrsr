import sys
import gzip
import json
import hashlib
import argparse
import subprocess
import urllib.request
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List
from urllib.error import URLError


#region Configuration

DIR = Path(__file__).resolve().parent / "data"
CHOICE_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

DATASETS: Dict[str, Dict[str, Any]] = {
    "mmlu_pro": {
        "url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/5b0b058f4c747aa50a47f6efe0dd4550e05d6094/data/test-00000-of-00001.parquet?download=true",
        "archive": "mmlu_pro.parquet",
        "sha256": "3455843e88b5fd8ca7814fcb65551ec3e30b33b1bb6867e4e6af16d7c5ed86c5",
        "val_url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/5b0b058f4c747aa50a47f6efe0dd4550e05d6094/data/validation-00000-of-00001.parquet?download=true",
        "val_archive": "mmlu_pro_val.parquet",
        "val_sha256": "b8c6c31eee58843630cfaff7a47a4f1c66220bb9473e6d265ef195c57da9041f",
        "final": "mmlu_pro.jsonl",
        "kind": "parquet_mmlu",
    },
    "arc_challenge": {
        "url": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/test-00000-of-00001.parquet?download=true",
        "archive": "arc_challenge.parquet",
        "sha256": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",
        "final": "arc_challenge.jsonl",
        "kind": "parquet_arc",
    },
    "winogrande": {
        "url": "https://huggingface.co/datasets/allenai/winogrande/resolve/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande_xl/validation-00000-of-00001.parquet?download=true",
        "archive": "winogrande.parquet",
        "sha256": "f9d914d1818c0aba98cabdf8af2bc4bd943c462b9c5e7062647e05c2ce3315d1",
        "final": "winogrande.jsonl",
        "kind": "parquet_winogrande",
    },
    "gsm8k": {
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "archive": "gsm8k_raw.jsonl",
        "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "final": "gsm8k.jsonl",
        "kind": "jsonl",
    },
    "humaneval": {
        "url": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "archive": "HumanEval.jsonl.gz",
        "sha256": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
        "final": "humaneval.jsonl",
        "kind": "jsonl_gz",
    },
}

#endregion

#region Planning

def compute_hash(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, streamed in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def show_plan() -> None:
    """Prints the dataset download plan."""
    print("\nDownstream benchmark setup plan")
    print(f"  Data directory:   {DIR}")
    for name, cfg in DATASETS.items():
        exists = (DIR / cfg["final"]).exists()
        status = "Dataset exists" if exists else "Download required"
        print(f"\n  [{name}]")
        print(f"    Source URL:       {cfg['url']}")
        print(f"    Download archive: {cfg['archive']} (SHA256: {cfg['sha256'][:16]}...)")
        if "val_url" in cfg:
            print(f"    Validation CoT:   {cfg['val_archive']} (SHA256: {cfg['val_sha256'][:16]}...)")
        print(f"    Output file:      {cfg['final']}")
        print(f"    Current status:   {status}")

#endregion

#region Downloading

def _read_parquet_rows(parquet_path: Path) -> List[Dict[str, Any]]:
    """Reads rows from a Parquet file."""
    try:
        import pyarrow.parquet as pq
        return pq.read_table(parquet_path).to_pylist()
    except ImportError:
        pass
    try:
        import pandas as pd
        return pd.read_parquet(parquet_path).to_dict(orient="records")
    except ImportError:
        raise

def _download_and_verify(url: str, dst_path: Path, expected_sha256: str, label: str) -> None:
    """Downloads a file and verifies its SHA-256 checksum."""
    print(f"\n[{label}] Downloading {dst_path.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "LRSR-setup"})
    with urllib.request.urlopen(req) as resp, open(dst_path, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)

    actual_hash = compute_hash(dst_path)
    if actual_hash.lower() != expected_sha256.lower():
        dst_path.unlink()
        raise ValueError(
            f"Archive hash mismatch for {label}:\n"
            f"Expected: {expected_sha256}\n"
            f"Got:      {actual_hash}"
        )
    print(f"  Archive integrity verified (SHA256: {actual_hash})")

def _clean_mmlu_options(raw_options: List[Any]) -> List[str]:
    """Removes trailing N/A placeholders as in TIGER-Lab MMLU-Pro."""
    opts = [str(o) for o in raw_options]
    while opts and (not opts[-1].strip() or opts[-1].strip() == "N/A"):
        opts.pop()
    return opts

def _format_mmlu_cot_example(question: str, options: List[str], cot_content: str = "", include_answer: bool = True) -> str:
    """Formats a single MMLU-Pro example exactly as in TIGER-AI-Lab/MMLU-Pro evaluate_from_local.py."""
    prompt = f"Question:\n{question}\nOptions:\n"
    for i, opt in enumerate(options):
        prompt += f"{CHOICE_LETTERS[i]}. {opt}\n"
    if include_answer:
        cot = cot_content.replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += f"{cot}\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

def download_dataset(name: str, cfg: Dict[str, Any], force: bool = False) -> None:
    """Downloads a benchmark archive, verifies its SHA-256, and prepares JSONL."""
    final_path = DIR / cfg["final"]
    if final_path.exists() and not force:
        print(f"\n[{name}] Already exists at {final_path}, skipping.")
        return

    archive_path = DIR / cfg["archive"]
    _download_and_verify(cfg["url"], archive_path, cfg["sha256"], name)

    print(f"  Extracting to {cfg['final']}...")
    if cfg["kind"] == "jsonl":
        archive_path.replace(final_path)
    elif cfg["kind"] == "jsonl_gz":
        with gzip.open(archive_path, "rt", encoding="utf-8") as src, open(final_path, "w", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    dst.write(line)
        archive_path.unlink()
    elif cfg["kind"] == "parquet_mmlu":
        val_path = DIR / cfg["val_archive"]
        _download_and_verify(cfg["val_url"], val_path, cfg["val_sha256"], f"{name}:val")

        val_rows = _read_parquet_rows(val_path)
        val_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in val_rows:
            opts = _clean_mmlu_options(r.get("options", []))
            if len(opts) < 2:
                continue
            cat = str(r.get("category", "other")).strip()
            val_by_cat[cat].append({
                "question_id": int(r.get("question_id", 0)),
                "question": str(r["question"]).strip(),
                "options": opts[:10],
                "cot_content": str(r.get("cot_content", "")).strip(),
            })

        # Build canonical 5-shot CoT prefix per category
        cot_prefix_by_cat: Dict[str, str] = {}
        for cat, ex_list in val_by_cat.items():
            prefix = (
                f"The following are multiple choice questions (with answers) about {cat}. "
                'Think step by step and then finish your answer with "the answer is (X)" '
                "where X is the correct letter choice.\n\n"
            )
            for ex in ex_list[:5]:
                prefix += _format_mmlu_cot_example(
                    ex["question"], ex["options"], ex["cot_content"], include_answer=True
                )
            cot_prefix_by_cat[cat] = prefix

        rows = _read_parquet_rows(archive_path)
        by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        dropped = 0
        for r in rows:
            raw_opts = _clean_mmlu_options(r.get("options", []))
            if len(raw_opts) < 2 or any(o.strip() == "N/A" for o in raw_opts):
                dropped += 1
                continue
            cat = str(r.get("category", "other")).strip()
            q_text = str(r["question"]).strip()
            opts = raw_opts[:10]
            prompt = cot_prefix_by_cat.get(cat, "") + _format_mmlu_cot_example(
                q_text, opts, include_answer=False
            )
            item = {
                "question_id": int(r.get("question_id", 0)),
                "category": cat,
                "question": q_text,
                "options": opts,
                "prompt": prompt,
                "answer": str(r["answer"]).strip().upper(),
            }
            by_category[cat].append(item)

        categories = sorted(by_category.keys())
        for cat in categories:
            by_category[cat].sort(key=lambda x: x["question_id"])

        interleaved: List[Dict[str, Any]] = []
        max_len = max((len(v) for v in by_category.values()), default=0)
        for idx in range(max_len):
            for cat in categories:
                if idx < len(by_category[cat]):
                    interleaved.append(by_category[cat][idx])

        with open(final_path, "w", encoding="utf-8") as dst:
            for item in interleaved:
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
        archive_path.unlink()
        val_path.unlink()
        print(f"  Prepared {len(interleaved)} 5-shot CoT rows across {len(categories)} categories.")
    elif cfg["kind"] == "parquet_arc":
        rows = _read_parquet_rows(archive_path)
        kept, dropped = 0, 0
        with open(final_path, "w", encoding="utf-8") as dst:
            for r in rows:
                choices_dict = r.get("choices", {})
                texts = [str(t) for t in choices_dict.get("text", [])]
                labels = [str(l).strip() for l in choices_dict.get("label", [])]
                ans_key = str(r.get("answerKey", "")).strip()
                if not texts or ans_key not in labels:
                    dropped += 1
                    continue
                item = {
                    "id": str(r.get("id", "")),
                    "question": str(r["question"]).strip(),
                    "choices": texts,
                    "answer_idx": int(labels.index(ans_key)),
                }
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
        archive_path.unlink()
        print(f"  Prepared {kept} rows.")
    elif cfg["kind"] == "parquet_winogrande":
        rows = _read_parquet_rows(archive_path)
        kept, dropped = 0, 0
        with open(final_path, "w", encoding="utf-8") as dst:
            for r in rows:
                sent = str(r.get("sentence", "")).strip()
                ans = str(r.get("answer", "")).strip()
                if "_" not in sent or ans not in ("1", "2"):
                    dropped += 1
                    continue
                item = {
                    "sentence": sent,
                    "option1": str(r["option1"]).strip(),
                    "option2": str(r["option2"]).strip(),
                    "answer_idx": int(ans) - 1,
                }
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
        archive_path.unlink()
        print(f"  Prepared {kept} rows.")

    print(f"  [{name}] Finished successfully!")

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downloads datasets for Downstream Benchmark.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--force", action="store_true", help="Re-download existing datasets")
    return parser.parse_args()

def confirm(auto: bool) -> bool:
    if auto:
        return True
    try:
        return input("\nProceed? [y/N]: ").strip().lower() in ("y", "yes")
    except EOFError:
        return False

def main() -> None:
    args = parse()

    show_plan()
    if not confirm(args.yes):
        print("Aborted.")
        return

    DIR.mkdir(parents=True, exist_ok=True)

    try:
        for name, cfg in DATASETS.items():
            download_dataset(name, cfg, force=args.force)
        print("\nAll downstream datasets prepared successfully!")
    except (URLError, OSError, EOFError, ValueError) as e:
        print(f"\nError: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
