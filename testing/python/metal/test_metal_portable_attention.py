"""Combined Metal coverage using only portable TileLang operations."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(target="metal", execution_backend="torch")
def portable_attention(dtype):
    m, n, k, d = 32, 64, 32, 32

    @T.prim_func
    def main(A: T.Tensor((m, k), dtype), B: T.Tensor((n, k), dtype), C: T.Tensor((n, d), dtype), D: T.Tensor((m, d), "float32")):
        with T.Kernel(1, threads=128):
            q = T.alloc_shared((m, k), dtype)
            key = T.alloc_shared((n, k), dtype)
            value = T.alloc_shared((n, d), dtype)
            scores = T.alloc_fragment((m, n), "float32")
            probs = T.alloc_fragment((m, n), dtype)
            maximum = T.alloc_fragment((m,), "float32")
            denominator = T.alloc_fragment((m,), "float32")
            output = T.alloc_fragment((m, d), "float32")
            T.copy(A, q)
            T.copy(B, key)
            T.copy(C, value)
            T.gemm(q, key, scores, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(m, n):
                scores[i, j] *= 0.125
            T.reduce_max(scores, maximum, dim=1)
            for i, j in T.Parallel(m, n):
                scores[i, j] = T.exp(scores[i, j] - maximum[i])
            T.reduce_sum(scores, denominator, dim=1)
            for i, j in T.Parallel(m, n):
                probs[i, j] = T.cast(scores[i, j] / denominator[i], dtype)
            T.gemm(probs, value, output, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(output, D)

    return main


@tilelang.testing.requires_metal
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_portable_attention(dtype):
    dt = getattr(torch, dtype)
    torch.manual_seed(0)
    q, k, v = (torch.randn(shape).to(dt) for shape in [(32, 32), (64, 32), (64, 32)])
    output = torch.empty(32, 32, device="mps")
    kernel = portable_attention(dtype)
    kernel(q.to("mps"), k.to("mps"), v.to("mps"), output)
    probability = torch.softmax((q.float() @ k.float().T) * 0.125, dim=-1).to(dt)
    expected = probability.float() @ v.float()
    torch.testing.assert_close(output.cpu(), expected, atol=2e-3 if dtype == "bfloat16" else 3e-4, rtol=2e-3)
