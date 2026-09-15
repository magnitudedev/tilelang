import tilelang
import tilelang.language as T
from tilelang import tvm


def _runtime_valid_m_gemm():
    m = n = k = 8

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "float32"),
        B: T.Tensor((k, n), "float32"),
        ValidM: T.Tensor((1,), "int32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_local((m, k), "float32")
            b = T.alloc_local((k, n), "float32")
            c = T.alloc_local((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True, valid_m=ValidM[0])
            for i, j in T.grid(ValidM[0], n):
                C[i, j] = c[i, j]

    return main


def test_runtime_valid_m_lowers_to_dynamic_cpu_loop():
    target = tvm.target.Target("c")
    with target:
        source = tilelang.lower(_runtime_valid_m_gemm(), target=target, target_host="c").kernel_source
    assert "ValidM" in source


if __name__ == "__main__":
    tilelang.testing.main()
