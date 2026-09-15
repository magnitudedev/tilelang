"""The backend owns shared matrix pitch, including a reused K/V tile."""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tilelang.jit(target="metal", execution_backend="torch")
def shared_pitch(dtype, keys=32, paired=False):
    @T.prim_func
    def main(
        Q: T.Tensor((32, 256), dtype), K: T.Tensor((keys, 256), dtype), V: T.Tensor((keys, 256), dtype), O: T.Tensor((32, 256), "float32")
    ):
        with T.Kernel(1, threads=128):
            query = T.alloc_fragment((32, 256), dtype)
            kv = T.alloc_shared((2, keys, 256) if paired else (keys, 256), dtype)
            scores = T.alloc_fragment((32, keys), "float32")
            rounded = T.alloc_fragment((32, keys), dtype)
            output = T.alloc_fragment((32, 256), "float32")
            T.copy(Q, query)
            if paired:
                T.copy(K, kv[0, :, :])
                T.copy(V, kv[1, :, :])
                T.gemm(query, kv[0, :, :], scores, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            else:
                T.copy(K, kv)
                T.gemm(query, kv, scores, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(scores, rounded)
            if paired:
                T.gemm(rounded, kv[1, :, :], output, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            else:
                T.copy(V, kv)
                T.gemm(rounded, kv, output, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(output, O)

    return main


@pytest.mark.parametrize("keys,paired", [(32, False), (64, False), (32, True)])
def test_shared_pitch_codegen(keys, paired):
    program = shared_pitch.get_tir("float16", keys, paired)
    with tvm.target.Target("metal"):
        source = tilelang.lower(program, target="metal").kernel_source
    extent = int(re.search(r"kv\[(\d+)\]", source).group(1))
    if paired:
        assert extent == 2 * keys * 256
    else:
        assert keys * 256 <= extent <= (keys * 264 if keys == 32 else keys * 256)
    assert "simdgroup_load(" in source


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("keys,paired", [(32, False), (64, False), (32, True)])
def test_reused_transposed_and_untransposed_shared_matrix(dtype, keys, paired):
    torch.manual_seed(42)
    dt = getattr(torch, dtype)
    q = (torch.randn(32, 256) * 0.05).to(dt)
    k, v = [(torch.randn(keys, 256) * 0.05).to(dt) for _ in range(2)]
    output = torch.empty(32, 256, device="mps")
    shared_pitch(dtype, keys, paired)(q.to("mps"), k.to("mps"), v.to("mps"), output)
    reference = (q.float() @ k.float().T).to(dt).float() @ v.float()
    torch.testing.assert_close(output.cpu(), reference, atol=3e-4, rtol=3e-3)


@tilelang.testing.requires_metal
def test_shared_accumulator_reused_as_a_pitched_operand():
    @T.prim_func
    def program(
        A: T.Tensor((32, 32), "float32"),
        B: T.Tensor((32, 32), "float32"),
        Seed: T.Tensor((32, 32), "float32"),
        O: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((32, 32), "float32")
            b = T.alloc_shared((32, 32), "float32")
            c = T.alloc_shared((32, 32), "float32")
            result = T.alloc_fragment((32, 32), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(Seed, c)
            T.gemm(a, b, c)
            T.gemm(c, b, result, clear_accum=True)
            T.copy(result, O)

    torch.manual_seed(317)
    a, b, seed = [torch.randn(32, 32, device="mps") * 0.05 for _ in range(3)]
    output = torch.empty_like(seed)
    kernel = tilelang.compile(program, target="metal", execution_backend="torch", out_idx=[])
    kernel(a, b, seed, output)
    expected = (a.cpu() @ b.cpu() + seed.cpu()) @ b.cpu()
    torch.testing.assert_close(output.cpu(), expected, atol=3e-5, rtol=3e-4)
