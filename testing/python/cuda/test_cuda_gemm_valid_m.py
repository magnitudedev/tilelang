import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _runtime_valid_m_gemm():
    m = n = k = 32

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float16"),
        B: T.Tensor((k, n), "float16"),
        ValidM: T.Tensor((1,), "int32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((m, k), "float16")
            b = T.alloc_shared((k, n), "float16")
            c = T.alloc_fragment((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True, valid_m=ValidM[0])
            for i, j in T.Parallel(m, n):
                if i < ValidM[0]:
                    C[i, j] = c[i, j]

    return main


def test_runtime_valid_m_lowers_to_uniform_mma_guard():
    with tvm.target.Target("cuda"):
        source = tilelang.lower(_runtime_valid_m_gemm(), target="cuda").kernel_source
    assert "mma_sync<" in source
    assert "ValidM" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("rows", [0, 1, 15, 16, 17, 31, 32])
def test_runtime_valid_m_computes_prefix(rows):
    torch.manual_seed(316)
    kernel = tilelang.compile(_runtime_valid_m_gemm(), out_idx=[], target="cuda")
    a = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    b = torch.randn((32, 32), dtype=torch.float16, device="cuda")
    valid_m = torch.tensor([rows], dtype=torch.int32, device="cuda")
    output = torch.full((32, 32), 17.0, dtype=torch.float32, device="cuda")
    kernel(a, b, valid_m, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output[:rows], a[:rows].float() @ b.float(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(output[rows:], torch.full_like(output[rows:], 17.0))


if __name__ == "__main__":
    tilelang.testing.main()
