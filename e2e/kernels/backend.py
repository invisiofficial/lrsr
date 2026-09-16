import os
import torch
import ctypes
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY = None


def get_lib():
    global _LIBRARY
    if _LIBRARY is None:
        if os.name == "nt":
            for directory in (Path(torch.__file__).parent / "lib", Path(os.environ.get("CUDA_PATH", "")) / "bin"):
                if directory.is_dir():
                    os.add_dll_directory(str(directory))

        name = "w4a16.dll" if os.name == "nt" else "lib_w4a16.so"
        path = _ROOT / "benchmarks" / "runtime" / "data" / name
        if not path.is_file():
            raise RuntimeError("Build native backend using setup.py first.")

        lib = ctypes.CDLL(str(path))
        ptr = ctypes.c_void_p

        lib.cutlass_pack_int4.argtypes = [ptr, ptr, ctypes.c_int64, ptr]
        lib.cutlass_gemm.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]
        lib.cutlass_pgmm.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]

        _LIBRARY = lib
    return _LIBRARY

def ptr(tensor: torch.Tensor):
    return tensor.data_ptr() if tensor is not None else None

def pack_int4(weight_q: torch.Tensor) -> torch.Tensor:
    n, k = weight_q.shape
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=weight_q.device)
    stream = torch.cuda.current_stream(weight_q.device)
    get_lib().cutlass_pack_int4(ptr(weight_q), ptr(packed), n * k, stream.cuda_stream)
    return packed
