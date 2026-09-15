"""Inspect concrete GEMM plans through the ordinary compilation pipeline."""
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class GemmPlan:
    shape: tuple[int, int, int]
    instruction_shape: tuple[int, int, int]
    input_dtypes: tuple[str, str]
    accumulation_dtype: str
    warp_partition: tuple[int, int]
    input_precision: tuple[str, str]

    @classmethod
    def from_emitter(cls, gemm, emitter):
        return cls(
            (int(gemm.M), int(gemm.N), int(gemm.K)),
            (int(emitter.micro_size_x), int(emitter.micro_size_y), int(emitter.micro_size_k)),
            (str(gemm.a_dtype), str(gemm.b_dtype)), str(gemm.accum_dtype),
            (int(emitter.block_row_warps), int(emitter.block_col_warps)),
            (str(getattr(emitter, "a_dtype_abbrv", gemm.a_dtype)),
             str(getattr(emitter, "b_dtype_abbrv", gemm.b_dtype))),
        )


_plans: ContextVar[list[GemmPlan] | None] = ContextVar("tilelang_gemm_plans", default=None)


def record_gemm_plan(implementation, target, threads):
    plans = _plans.get()
    if plans is not None:
        plan = implementation.gemm_plan(target, threads)
        if plan is not None and plan not in plans:
            plans.append(plan)


def analyze_gemm(program, *, target, target_host=None) -> tuple[GemmPlan, ...]:
    """Lower a real program and report the matrix plans used by its implementations.

    Geometry comes from the same emitter construction used by lowering. All
    ordinary lowering/codegen checks run; errors are not converted to support
    flags. No device binary is built. An implementation without matrix-plan
    reporting contributes no plan. Vendor compilation and numerical validation
    remain separate obligations.
    """
    from tilelang import tvm
    from tilelang.engine.lower import lower_with_context
    from tilelang.backend.module import create_backend_context

    plans = []
    token = _plans.set(plans)
    try:
        context = create_backend_context(target, target_host)
        with tvm.transform.PassContext(), context.target:
            lower_with_context(program, context, enable_host_codegen=False, enable_device_compile=False)
        return tuple(plans)
    finally:
        _plans.reset(token)


def plan_gemm(*, target, m, n, k, input_dtype, accumulation_dtype="float32",
              threads=128, a_scope="shared", b_scope="shared", c_scope="local.fragment"):
    """Analyze a concrete portable matrix tile, including staging and writeback."""
    from tilelang import language as T

    @T.macro
    def body(A, B, C):
        with T.Kernel(1, threads=threads):
            a = A if a_scope == "global" else T.alloc_fragment((m, k), input_dtype, scope=a_scope)
            b = B if b_scope == "global" else T.alloc_fragment((k, n), input_dtype, scope=b_scope)
            c = C if c_scope == "global" else T.alloc_fragment((m, n), accumulation_dtype, scope=c_scope)
            if a_scope != "global":
                T.copy(A, a)
            if b_scope != "global":
                T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True)
            if c_scope != "global":
                T.copy(c, C)

    program = T.build_prim_func("matrix_plan", (
        ("A", T.Tensor((m, k), input_dtype)), ("B", T.Tensor((k, n), input_dtype)),
        ("C", T.Tensor((m, n), accumulation_dtype))), body)
    plans = analyze_gemm(program, target=target)
    if len(plans) > 1:
        raise ValueError("single matrix operation produced inconsistent plans")
    return plans[0] if plans else None
