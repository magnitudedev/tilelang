import pytest
import tilelang.language as T
from tilelang import tvm


@pytest.mark.parametrize("dtype,value", [("float32", 1.5), ("bool", 1)])
def test_valid_m_rejects_non_integer_extents(dtype, value):
    a = tvm.tirx.decl_buffer((8, 8), "float16", scope="shared")
    b = tvm.tirx.decl_buffer((8, 8), "float16", scope="shared")
    c = tvm.tirx.decl_buffer((8, 8), "float32", scope="local.fragment")
    extent = tvm.tirx.const(value, dtype)
    with pytest.raises(TypeError, match="scalar integer"):
        T.gemm(a, b, c, valid_m=extent)


def test_valid_m_buffer_read_participates_in_initialization_analysis(capfd):
    import tilelang

    @T.prim_func
    def program(A: T.Tensor((8, 8), "float16"), B: T.Tensor((8, 8), "float16")):
        with T.Kernel(1, threads=32):
            extent = T.alloc_shared((1,), "int32")
            c = T.alloc_fragment((8, 8), "float32")
            T.gemm(A, B, c, clear_accum=True, valid_m=extent[0])

    tilelang.transform.VerifyBufferInit()(tvm.IRModule.from_expr(program))
    warning = capfd.readouterr().err
    assert "Buffer read before initialization" in warning
    assert "extent" in warning
