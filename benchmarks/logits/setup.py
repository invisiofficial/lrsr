import zipfile
import hashlib
import argparse
import urllib.request
from pathlib import Path
from urllib.error import URLError


#region Configuration

DIR = Path(__file__).resolve().parent / "data"

URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip?download=true"

ZIP_NAME = "wikitext-2-raw-v1.zip"
EXPECTED_HASH = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"

TARGET_FILE = "wikitext-2-raw/wiki.test.raw"
FINAL_FILE = "wiki.test.raw"

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
    exists = (DIR / FINAL_FILE).exists()
    print("\nDataset download plan")
    print(f"  Source URL:       {URL}")
    print(f"  Download archive: {ZIP_NAME} (SHA256: {EXPECTED_HASH[:16]}...)")
    print(f"  Extract target:   {TARGET_FILE} > {FINAL_FILE}")
    print(f"  Data directory:   {DIR}")
    if exists:
        print("  Current status:   Dataset exists")
    else:
        print("  Current status:   Download required")

#endregion

#region Downloading

def download_archive() -> None:
    """Downloads the target archive, verifies its hash, and extracts the raw test set."""
    final_path = DIR / FINAL_FILE
    zip_path = DIR / ZIP_NAME

    print(f"\nDownloading {ZIP_NAME}...")

    req = urllib.request.Request(URL, headers={"User-Agent": "LRSR-setup"})
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
        f.write(resp.read())

    zip_hash = compute_hash(zip_path)
    if zip_hash.lower() != EXPECTED_HASH:
        zip_path.unlink()
        raise ValueError(f"Archive hash mismatch:\n"
                          "Expected: {EXPECTED_HASH}\n"
                          "Got:      {zip_hash}")
    print(f"  Archive integrity verified (SHA256: {zip_hash})")

    print(f"Extracting {TARGET_FILE}...")
    with zipfile.ZipFile(zip_path) as z:
        with z.open(TARGET_FILE) as src, open(final_path, "wb") as dst:
            dst.write(src.read())

    zip_path.unlink()

    print(f"\nDownload finished sucessfully!")

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downloads dataset for Logits Benchmark.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
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
        download_archive()
    except (URLError, OSError, zipfile.BadZipFile, ValueError) as e:
        print(f"\nError: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
