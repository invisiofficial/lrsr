import os
import shutil
import argparse
import subprocess
from pathlib import Path


#region Configuration

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "e2e" / "kernels"
CUTLASS = ROOT / "dependencies" / "cutlass"

DIR = Path(__file__).resolve().parent / "data"

#endregion

#region Planning

def get_compiler_environment() -> dict:
    """Sets up MSVC environment variables."""
    env = os.environ.copy()
    if os.name != "nt" or shutil.which("cl", path=env.get("PATH")):
        return env

    vswhere_path = Path(env.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere_path.is_file():
        raise RuntimeError("MSVC not found. Please install Visual Studio C++ build tools.")

    install_path = subprocess.check_output([
        str(vswhere_path), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"
    ], text=True).strip()

    vcvars_path = Path(install_path) / "VC/Auxiliary/Build/vcvars64.bat"
    if not vcvars_path.is_file():
        raise RuntimeError("vcvars64.bat not found. Please install Visual Studio C++ build tools.")

    result = subprocess.check_output(f'cmd.exe /d /s /c ""{vcvars_path}" >nul && set"', text=True)
    for line in result.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            env[key.upper()] = value
    return env

def get_cuda_version(nvcc_path: str) -> str:
    """Gets CUDA version from NVCC."""
    try:
        out = subprocess.check_output([nvcc_path, "--version"], text=True)
        for line in out.splitlines():
            if "release" in line:
                return line.split("release")[-1].strip()
    except Exception:
        pass
    return "Unknown"

def get_cutlass_version() -> str:
    """Gets CUTLASS version from git tags."""
    try:
        return subprocess.check_output(["git", "-C", str(CUTLASS), "describe", "--tags", "--always"], text=True).strip()
    except Exception:
        return "Unknown"

def show_plan(nvcc_path: str, architectures: list, output_file: Path) -> None:
    """Prints the compilation plan."""
    print("\nRuntime backend compilation plan")
    print(f"  CUDA Compiler:    {get_cuda_version(nvcc_path)}")
    print(f"  CUTLASS Version:  {get_cutlass_version()}")
    print(f"  Architectures:    {', '.join(architectures)}")
    print(f"  Source directory: {KERNELS}")
    print(f"  Output binary:    {output_file}")
    if output_file.exists():
        print("  Current status:   Library exists")
    else:
        print("  Current status:   Compilation required")

#endregion

#region Compiling

def compile_backend(nvcc: str, env: dict, architectures: list, output_file: Path) -> None:
    """Compiles the W4A16 CUTLASS backend into a shared library."""
    command = [
        nvcc, "-std=c++17", "-O3", "-lineinfo", "--expt-relaxed-constexpr",
        "-I", str(CUTLASS / "include"), "-I", str(KERNELS),
        "--shared", str(KERNELS / "w4a16.cu"), "-o", str(output_file)
    ]

    for arch in architectures:
        command.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    command.append(f"-gencode=arch=compute_{architectures[-1]},code=compute_{architectures[-1]}")

    if os.name == "nt":
        command += ["-Xcompiler=/MD,/EHsc"]
    else:
        command += ["-Xcompiler=-fPIC"]

    subprocess.check_call(command, env=env)
    print(f"\nCompilation finished successfully!")

#endregion

def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compiles the native W4A16 backend for runtime benchmarks.")
    parser.add_argument("--arch", nargs="+", default=["75", "80", "86", "89", "90", "120"], help="CUDA SM targets")
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

    env = get_compiler_environment()
    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    if not nvcc:
        raise RuntimeError("nvcc not found. Please ensure CUDA Toolkit is installed.")

    DIR.mkdir(parents=True, exist_ok=True)
    lib_name = "w4a16.dll" if os.name == "nt" else "lib_w4a16.so"
    output_file = DIR / lib_name

    show_plan(nvcc, args.arch, output_file)
    if not confirm(args.yes):
        print("Aborted.")
        return

    print()

    try:
        compile_backend(nvcc, env, args.arch, output_file)
    except subprocess.CalledProcessError as e:
        print(f"\nError during compilation: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
