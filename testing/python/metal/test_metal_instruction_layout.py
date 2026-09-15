"""Explicit portable layouts must agree with native instruction footprints."""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tilelang.jit(target="metal", execution_backend="torch")
def instruction_layout(dtype="float16", transposed=True, offset=8):
    @T.prim_func
    def main(A: T.Tensor((64, 64), dtype), B: T.Tensor((64, 64), dtype), O: T.Tensor((32, 32), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((64, 64), dtype)
            b = T.alloc_shared((64, 64), dtype)
            c = T.alloc_fragment((32, 32), "float32")
            T.annotate_layout(
                {
                    a: T.Layout((64, 64), lambda i, j: ((i // 8) * 8 + j // 8) * 64 + i % 8 * 8 + j % 8),
                    b: T.Layout((64, 64), lambda i, j: ((i // 8) * 8 + j // 8) * 64 + i % 8 * 8 + j % 8),
                }
            )
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a[offset : offset + 32, 16:48], b[16:48, 8:40], c, transpose_B=transposed, clear_accum=True)
            T.copy(c, O)

    return main


@pytest.mark.parametrize("transposed", [False, True])
def test_explicit_instruction_layout_uses_physical_pitch(transposed):
    with tvm.target.Target("metal"):
        source = tilelang.lower(instruction_layout.get_tir(transposed=transposed), target="metal").kernel_source
    loads = [line for line in source.splitlines() if "simdgroup_load(" in line]
    assert loads and all(re.search(r", 8, 0,", line) for line in loads)


def test_instruction_layout_rejects_a_cross_block_matrix_load():
    with tvm.target.Target("metal"), pytest.raises(Exception, match="constant-pitch 8x8 instruction tile"):
        tilelang.lower(instruction_layout.get_tir(offset=1), target="metal")


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("transposed", [False, True])
def test_instruction_layout_execution_with_offset_regions(dtype, transposed):
    torch.manual_seed(814)
    dt = getattr(torch, dtype)
    a, b = [(torch.randn(64, 64) * 0.05).to(dt) for _ in range(2)]
    output = torch.empty((32, 32), dtype=torch.float32, device="mps")
    instruction_layout(dtype, transposed)(a.to("mps"), b.to("mps"), output)
    right = b[16:48, 8:40].float()
    expected = a[8:40, 16:48].float() @ (right.T if transposed else right)
    torch.testing.assert_close(output.cpu(), expected, atol=3e-5, rtol=3e-4)
