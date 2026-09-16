import ctypes

import pytest

import tilelang.testing
from tilelang import tvm
from tilelang.libinfo import find_lib_path


def _cudart_stub():
    try:
        path = find_lib_path("stub_cudart")
    except RuntimeError:
        pytest.skip("CUDA runtime stubs are not enabled in this build")
    return ctypes.CDLL(path)


def test_cudart_stub_exports_driver_version():
    # Symbol lookup does not initialize CUDA or require a GPU.
    assert _cudart_stub().cudaDriverGetVersion is not None


@tilelang.testing.requires_cuda
def test_cudart_stub_driver_version_matches_device():
    query = _cudart_stub().cudaDriverGetVersion
    query.argtypes = [ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    version = ctypes.c_int()
    assert query(ctypes.byref(version)) == 0
    assert version.value > 0
    assert int(tvm.cuda(0).driver_version) == version.value
