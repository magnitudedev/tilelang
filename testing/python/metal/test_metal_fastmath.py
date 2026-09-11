"""Metal lowering of the fast-math intrinsics to metal::fast functions."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

FAST = {
    "exp": T.__exp,
    "exp10": T.__exp10,
    "log": T.__log,
    "log2": T.__log2,
    "log10": T.__log10,
    "tan": T.__tan,
    "cos": T.__cos,
    "sin": T.__sin,
}
PLAIN = {
    "exp": T.exp,
    "exp10": T.exp10,
    "log": T.log,
    "log2": T.log2,
    "log10": T.log10,
    "tan": T.tan,
    "cos": T.cos,
    "sin": T.sin,
}
REFERENCE = {
    "exp": torch.exp,
    "exp10": lambda x: torch.pow(10.0, x),
    "log": torch.log,
    "log2": torch.log2,
    "log10": torch.log10,
    "tan": torch.tan,
    "cos": torch.cos,
    "sin": torch.sin,
}


@tilelang.jit(target="metal", execution_backend="torch")
def unary_kernel(name, dtype, fast, n=1024):
    apply = FAST[name] if fast else PLAIN[name]

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
            i = block * 128 + T.get_thread_binding(0)
            if i < n:
                B[i] = apply(A[i])

    return main


def inputs(name, dtype, n=1024):
    if name in ("log", "log2", "log10"):
        return torch.linspace(0.05, 40.0, n).to(dtype)
    if name == "tan":
        return torch.linspace(-1.2, 1.2, n).to(dtype)
    if name == "exp10":
        return torch.linspace(-4.0, 4.0, n).to(dtype)
    return torch.linspace(-10.0, 10.0, n).to(dtype)


@pytest.mark.parametrize("name", sorted(FAST))
def test_fastmath_codegen(name):
    program = unary_kernel.get_tir(name, "float32", True)
    with tvm.target.Target("metal"):
        source = tilelang.lower(program, target="metal").kernel_source
    assert f"metal::fast::{name}(" in source


@pytest.mark.parametrize("name", sorted(PLAIN))
def test_plain_math_codegen_is_unqualified(name):
    program = unary_kernel.get_tir(name, "float32", False)
    with tvm.target.Target("metal"):
        source = tilelang.lower(program, target="metal").kernel_source
    assert "metal::fast::" not in source
    assert "metal::precise::" not in source


@tilelang.testing.requires_metal
@pytest.mark.parametrize("name", sorted(FAST))
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_fastmath_execution(name, dtype):
    torch_dtype = getattr(torch, dtype)
    a = inputs(name, torch_dtype)
    expected = REFERENCE[name](a.to(torch.float64))
    b = torch.empty_like(a, device="mps")
    unary_kernel(name, dtype, True)(a.to("mps"), b)
    tolerance = 1e-2 if dtype == "float16" else 2e-3
    torch.testing.assert_close(b.cpu().to(torch.float64), expected, rtol=tolerance, atol=tolerance)


def test_fastmath_rejects_bfloat16():
    program = unary_kernel.get_tir("exp", "bfloat16", True)
    with tvm.target.Target("metal"), pytest.raises(tvm.error.InternalError, match="fast-math"):
        tilelang.lower(program, target="metal")
