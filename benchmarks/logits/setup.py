import gzip
import json
import zipfile
import hashlib
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, Any
from urllib.error import URLError


#region Configuration

DIR = Path(__file__).resolve().parent / "data"

DATASETS: Dict[str, Dict[str, Any]] = {
    "wikitext": {
        "url": "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip?download=true",
        "archive": "wikitext-2-raw-v1.zip",
        "sha256": "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
        "member": "wikitext-2-raw/wiki.test.raw",
        "final": "wiki.test.raw",
        "kind": "zip",
    },
    "c4": {
        "url": "https://huggingface.co/datasets/allenai/c4/resolve/f998d2cd8b92435980789e3ecb2f89b4c68bfe1e/en/c4-validation.00000-of-00008.json.gz?download=true",
        "archive": "c4-validation.00000-of-00008.json.gz",
        "sha256": "bc35d7c1b1d14b90cd3a394cccbcbe191935edd04bf42ee965379c6e2987a5f0",
        "final": "c4.test.raw",
        "kind": "jsonl_gz",
        "max_docs": 2000,
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
    print("\nDataset download plan")
    print(f"  Data directory:   {DIR}")
    for name, cfg in DATASETS.items():
        exists = (DIR / cfg["final"]).exists()
        status = "Dataset exists" if exists else "Download required"
        print(f"\n  [{name}]")
        print(f"    Source URL:       {cfg['url']}")
        print(f"    Download archive: {cfg['archive']} (SHA256: {cfg['sha256'][:16]}...)")
        print(f"    Output file:      {cfg['final']}")
        print(f"    Current status:   {status}")

#endregion

#region Downloading

def download_dataset(name: str, cfg: Dict[str, Any], force: bool = False) -> None:
    """Downloads a dataset archive, verifies its hash, and extracts the raw text file."""
    final_path = DIR / cfg["final"]
    if final_path.exists() and not force:
        print(f"\n[{name}] Already exists at {final_path}, skipping.")
        return

    archive_path = DIR / cfg["archive"]
    print(f"\n[{name}] Downloading {cfg['archive']}...")

    req = urllib.request.Request(cfg["url"], headers={"User-Agent": "LRSR-setup"})
    with urllib.request.urlopen(req) as resp, open(archive_path, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)

    archive_hash = compute_hash(archive_path)
    if archive_hash.lower() != cfg["sha256"].lower():
        archive_path.unlink()
        raise ValueError(
            f"Archive hash mismatch for {name}:\n"
            f"Expected: {cfg['sha256']}\n"
            f"Got:      {archive_hash}"
        )
    print(f"  Archive integrity verified (SHA256: {archive_hash})")

    print(f"  Extracting to {cfg['final']}...")
    if cfg["kind"] == "zip":
        with zipfile.ZipFile(archive_path) as z:
            with z.open(cfg["member"]) as src, open(final_path, "wb") as dst:
                dst.write(src.read())
    elif cfg["kind"] == "jsonl_gz":
        decoder = json.JSONDecoder()
        max_docs = cfg.get("max_docs", 2000)
        docs = []
        with gzip.open(archive_path, "rt", encoding="utf-8", errors="replace") as gz:
            buffer = ""
            while len(docs) < max_docs:
                chunk = gz.read(65536)
                if not chunk:
                    break
                buffer += chunk
                idx = 0
                while idx < len(buffer):
                    while idx < len(buffer) and buffer[idx].isspace():
                        idx += 1
                    if idx >= len(buffer):
                        break
                    try:
                        item, end_idx = decoder.raw_decode(buffer, idx)
                    except json.JSONDecodeError:
                        break
                    text = item.get("text", "").strip()
                    if text:
                        docs.append(text)
                        if len(docs) >= max_docs:
                            break
                    idx = end_idx
                buffer = buffer[idx:]

        with open(final_path, "w", encoding="utf-8") as dst:
            dst.write("\n\n".join(docs))

    archive_path.unlink()
    print(f"  [{name}] Finished successfully!")

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downloads datasets for Logits Benchmark.")
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
    except (URLError, OSError, zipfile.BadZipFile, EOFError, ValueError) as e:
        print(f"\nError: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
