"""Fragment/shared GEMM operand combinations use portable fragment buffers."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tilelang.jit(target="metal", execution_backend="torch")
def fragment_operands(m, n, k, a_fragment, b_fragment, trans_a=False, trans_b=False, shared_c=False, policy=T.GemmWarpPolicy.Square):
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    @T.prim_func
    def main(A: T.Tensor(a_shape, "float16"), B: T.Tensor(b_shape, "float16"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_fragment(a_shape, "float16") if a_fragment else T.alloc_shared(a_shape, "float16")
            b = T.alloc_fragment(b_shape, "float16") if b_fragment else T.alloc_shared(b_shape, "float16")
            c = T.alloc_shared((m, n), "float32") if shared_c else T.alloc_fragment((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            if a_fragment:
                for i, j in T.Parallel(*a_shape):
                    a[i, j] = a[i, j] * T.float16(0.5)
            if b_fragment:
                for i, j in T.Parallel(*b_shape):
                    b[i, j] = b[i, j] * T.float16(0.5)
            T.gemm(a, b, c, transpose_A=trans_a, transpose_B=trans_b, clear_accum=True, policy=policy)
            T.copy(c, C)

    return main


@tilelang.jit(target="metal", execution_backend="torch")
def offset_transposed_region():
    """A transposed operand keeps physical row/column offsets unswapped."""
    m = n = k = 32

    @T.prim_func
    def main(
        A: T.Tensor((48, 48), "float16"),
        B: T.Tensor((80, 48), "float16"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((48, 48), "float16")
            b = T.alloc_shared((80, 48), "float16")
            c = T.alloc_fragment((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(
                a[8:40, 16:48],
                b[24:56, 8:40],
                c,
                transpose_B=True,
                clear_accum=True,
            )
            T.copy(c, C)

    return main


@tilelang.jit(target="metal", execution_backend="torch")
def runtime_valid_m_gemm():
    """A runtime M prefix predicates complete 8-row simdgroup tiles."""
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
            c = T.alloc_shared((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(C, c)
            T.gemm(a, b, c, clear_accum=True, valid_m=ValidM[0])
            T.copy(c, C)

    return main


@pytest.mark.parametrize("a_fragment,b_fragment", [(True, False), (False, True), (True, True)])
def test_fragment_operand_codegen(a_fragment, b_fragment):
    program = fragment_operands.get_tir(32, 32, 32, a_fragment, b_fragment)
    with tvm.target.Target("metal"):
        source = tilelang.lower(program, target="metal").kernel_source
    assert "simdgroup_multiply_accumulate" in source
    assert ".thread_elements()" in source


@tilelang.testing.requires_metal
def test_transposed_buffer_region_preserves_physical_offsets():
    torch.manual_seed(314)
    a = torch.randn((48, 48), dtype=torch.float16)
    b = torch.randn((80, 48), dtype=torch.float16)
    output = torch.empty((32, 32), device="mps")
    offset_transposed_region()(a.to("mps"), b.to("mps"), output)
    expected = a[8:40, 16:48].float() @ b[24:56, 8:40].float().T
    torch.testing.assert_close(output.cpu(), expected, atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_metal
def test_runtime_valid_m_computes_unaligned_prefix():
    torch.manual_seed(315)
    a = torch.randn((64, 32), dtype=torch.float16)
    b = torch.randn((32, 32), dtype=torch.float16)
    valid_m = torch.tensor([25], dtype=torch.int32)
    output = torch.full((64, 32), 17.0, dtype=torch.float32, device="mps")
    runtime_valid_m_gemm()(a.to("mps"), b.to("mps"), valid_m.to("mps"), output)
    expected = a[:25].float() @ b.float()
    torch.testing.assert_close(output.cpu()[:25], expected, atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_metal
@pytest.mark.parametrize(
    "a_fragment,b_fragment,policy",
    [
        (True, False, T.GemmWarpPolicy.Square),
        (False, True, T.GemmWarpPolicy.Square),
        (True, True, T.GemmWarpPolicy.Square),
        (True, True, T.GemmWarpPolicy.FullRow),
        (True, True, T.GemmWarpPolicy.FullCol),
    ],
)
@pytest.mark.parametrize("trans_a,trans_b", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("shared_c", [False, True])
def test_fragment_operands_execution(a_fragment, b_fragment, trans_a, trans_b, shared_c, policy):
    m, n, k = 32, 64, 32
    a = torch.randn((k, m) if trans_a else (m, k), dtype=torch.float16)
    b = torch.randn((n, k) if trans_b else (k, n), dtype=torch.float16)
    c = torch.empty(m, n, device="mps")
    fragment_operands(m, n, k, a_fragment, b_fragment, trans_a, trans_b, shared_c, policy)(a.to("mps"), b.to("mps"), c)
    av = a.T if trans_a else a
    bv = b.T if trans_b else b
    if a_fragment:
        av = av * 0.5
    if b_fragment:
        bv = bv * 0.5
    torch.testing.assert_close(c.cpu(), av.float() @ bv.float(), atol=1e-4, rtol=1e-4)
