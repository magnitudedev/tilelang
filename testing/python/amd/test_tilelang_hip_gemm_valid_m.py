import tilelang
import tilelang.language as T
from tilelang import tvm


def _runtime_valid_m_gemm():
    m, n, k = 64, 32, 32

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


def test_runtime_valid_m_lowers_to_uniform_mfma_guard():
    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx90a"})
    with target:
        source = tilelang.lower(_runtime_valid_m_gemm(), target=target).kernel_source
    assert "__builtin_amdgcn_mfma" in source
    assert "ValidM" in source


if __name__ == "__main__":
    tilelang.testing.main()
