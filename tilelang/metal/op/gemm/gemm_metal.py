from __future__ import annotations

from tilelang import tvm as tvm
from tilelang.layout import Fragment, Layout
from tilelang.metal import language as T
from tilelang.metal.utils import (
    is_metal_cooperative_tensor,
    is_metal_simdgroup,
)
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import (
    is_fragment,
    is_full_region,
    is_global,
    is_shared,
)
from tvm import arith
from tvm import tirx as tir
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_METAL = "metal.simdgroup"
GEMM_INST_METAL_COOPERATIVE_TENSOR = "metal.cooperative_tensor"


def _partial_m_extent(gemm: GemmBase):
    """Return a runtime M bound, or None for the ordinary full-tile path."""
    valid_m = gemm.valid_m
    if arith.Analyzer().can_prove_equal(valid_m, gemm.M):
        return None
    if isinstance(valid_m, tir.IntImm):
        extent = int(valid_m)
        if not 0 <= extent <= int(gemm.M):
            raise ValueError(f"Metal T.gemm valid_m must be in [0, {gemm.M}], got {extent}")
    return valid_m


def _padded_stride(buffer):
    continuous = int(buffer.shape[-1])
    element_bits = int(tvm.DataType(buffer.dtype).bits)
    padded = continuous
    if (element_bits * continuous) % 256 == 0:
        padded += 128 // element_bits
    return padded


def _make_padded_layout(buffer):
    shape = buffer.shape
    padded = _padded_stride(buffer)
    return Layout(shape, lambda i, j: i * padded + j)


def _simd_shared_stride(buffer, target):
    stride = _padded_stride(buffer)
    limit = int(target.attrs.get("max_shared_memory_per_block", 32768))
    if int(buffer.shape[0]) * stride * tvm.DataType(buffer.dtype).bits // 8 > limit:
        stride = int(buffer.shape[-1])
    return stride


def _make_simd_shared_layout(buffer, target):
    """Keep affine rows so native matrix loads consume the physical pitch."""
    stride = _simd_shared_stride(buffer, target)
    return Layout(buffer.shape, lambda i, j: i * stride + j)


def _simd_operand_pitch(region, layout_map):
    """Prove the physical 8x8 instruction footprint, not a logical row pitch."""
    buffer = region.buffer
    layout = layout_map.get(buffer)
    if layout is None:
        return buffer.strides[-2] if buffer.strides else buffer.shape[-1]
    analyzer = arith.Analyzer()
    tile_i, tile_j = tir.Var("matrix_tile_i", "int32"), tir.Var("matrix_tile_j", "int32")
    row, col = tir.Var("matrix_row", "int32"), tir.Var("matrix_col", "int32")
    analyzer.bind(row, Range(0, 8))
    analyzer.bind(col, Range(0, 8))
    base = [item.min for item in region.region]
    base[-2] += tile_i * 8
    base[-1] += tile_j * 8
    expression = layout.get_linearized_forward_index()
    variables = layout.get_forward_vars()

    def offset(i, j):
        indices = [*base[:-2], base[-2] + i, base[-1] + j]
        return analyzer.simplify(tir.stmt_functor.substitute(expression, dict(zip(variables, indices))))

    origin = offset(0, 0)
    pitch = analyzer.simplify(offset(1, 0) - origin)
    if not isinstance(pitch, tir.IntImm) or int(pitch) < 8 or not analyzer.can_prove_equal(offset(row, col), origin + row * pitch + col):
        raise ValueError("Metal SIMD-group shared layout must expose a contiguous, constant-pitch 8x8 instruction tile")
    return int(pitch)


class GemmMetalSimdGroup(GemmBase):
    supports_runtime_valid_m = True

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def infer_layout(self, target: Target, thread_nums: int):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_METAL)
        warp_m = int(self.M // m_warp)
        warp_n = int(self.N // n_warp)

        def matrix_layout(buffer, tile_rows, tile_cols, row_warps, transpose=False, replicate=1, operand=None):
            def forward(i, j, rep=0):
                if transpose:
                    i, j = j, i
                lane = (i % 8 // 4) * 16 + (i % 4) * 2 + (j % 8 // 4) * 8 + j % 4 // 2
                if operand == "A":
                    warp = i // tile_rows + rep * row_warps
                elif operand == "B":
                    warp = rep + (j // tile_cols) * row_warps
                else:
                    warp = i // tile_rows + (j // tile_cols) * row_warps
                index = ((i % tile_rows // 8) * (tile_cols // 8) + j % tile_cols // 8) * 2 + j % 2
                return warp * 32 + lane, index

            return Fragment(
                buffer.shape,
                forward_thread_fn=lambda i, j, rep=0: forward(i, j, rep)[0],
                forward_index_fn=lambda i, j: forward(i, j)[1],
                replicate=replicate,
            )

        result = {}
        # Shared operands and native matrix loads agree on their physical pitch.
        for buffer in (self.A, self.B):
            if is_shared(buffer) and len(buffer.shape) == 2:
                result[buffer] = _make_simd_shared_layout(buffer, target)
        if is_fragment(self.C):
            result[self.C] = matrix_layout(self.C, warp_m, warp_n, m_warp)
        if is_fragment(self.A):
            assert is_full_region(self.ARegion), "Fragment input A must be a full region"
            result[self.A] = matrix_layout(self.A, warp_m, int(self.K), m_warp, self.trans_A, n_warp, "A")
        if is_fragment(self.B):
            assert is_full_region(self.BRegion), "Fragment input B must be a full region"
            result[self.B] = matrix_layout(self.B, int(self.K), warp_n, m_warp, self.trans_B, m_warp, "B")
        return result

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tir.PrimExpr,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_METAL)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        from tilelang.metal.intrinsics.metal_macro_generator import MPSIntrinEmitter

        mps_emitter = MPSIntrinEmitter(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_index,
            use_cooperative_tensor=False,
            a_stride_override=_simd_operand_pitch(self.ARegion, layout_map) if is_shared(self.A) else None,
            b_stride_override=_simd_operand_pitch(self.BRegion, layout_map) if is_shared(self.B) else None,
        )

        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        accum_dtype = self.accum_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_k = mps_emitter.micro_size_k
        micro_size_x = mps_emitter.micro_size_x
        valid_m = _partial_m_extent(self)
        warp_m, _ = mps_emitter._get_warp_indices()
        warp_start_m = warp_m * warp_row_tiles
        # The outer warp predicate already proves its sole instruction row is
        # active. Do not repeat that dynamic condition in every load/MMA/clear;
        # apart from code size, repeated source-bound loads obstruct native
        # instruction scheduling. Multi-row warps still predicate each row.
        instruction_bound = valid_m if warp_rows > 1 else None

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        C_buf = C_region.buffer
        clear_accum = self.clear_accum
        c_in_register = is_fragment(C_buf) or is_metal_simdgroup(C_buf)

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"
        assert is_full_region(C_region), "Fragment output C must be a full region"
        assert c_in_register or is_shared(C_buf), (
            f"Metal GEMM requires C in local.fragment, metal.simdgroup or shared scope, got {C_buf.scope()}"
        )

        a_in_fragment = is_fragment(self.A)
        b_in_fragment = is_fragment(self.B)
        if not (is_shared(self.A) or a_in_fragment) or not (is_shared(self.B) or b_in_fragment):
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

        @T.macro
        def multiply(A_local, B_local, C_local, bound):
            for ki in T.serial(block_K // micro_size_k):
                if not a_in_fragment:
                    mps_emitter.ldmatrix_a(A_local, A_region, ki, valid_m=bound)
                if not b_in_fragment:
                    mps_emitter.ldmatrix_b(B_local, B_region, ki, valid_m=bound)
                mps_emitter.mma(
                    A_region.buffer if a_in_fragment else A_local,
                    B_region.buffer if b_in_fragment else B_local,
                    C_local,
                    ki,
                    bound,
                )

        if c_in_register:

            @T.prim_func
            def _gemm_ss_simdgroup() -> None:
                A_local = T.alloc_local((warp_rows * 2,), a_dtype)
                B_local = T.alloc_local((warp_cols * 2,), b_dtype)
                # A runtime prefix predicates instruction rows, not three
                # separately expanded copies of the whole contraction. Static
                # full tiles have valid_m=None and simplify to the unguarded
                # body; partial tiles retain the emitter's load/MMA predicates.
                if valid_m is None or warp_start_m < valid_m:
                    if clear_accum:
                        for _i in T.serial(num_simd_c):
                            if instruction_bound is None or warp_start_m + (_i // warp_cols) * micro_size_x < instruction_bound:
                                T.make_filled_simdgroup_matrix(C_buf.data, _i, T.cast(0, accum_dtype))
                    multiply(A_local, B_local, C_buf, instruction_bound)

            return _Simplify(_gemm_ss_simdgroup, inline_let=True)

        @T.prim_func
        def _gemm_ss_shared() -> None:
            A_local = T.alloc_local((warp_rows * 2,), a_dtype)
            B_local = T.alloc_local((warp_cols * 2,), b_dtype)
            C_simd = T.alloc_local((num_simd_c * 2,), accum_dtype)
            if valid_m is None or warp_start_m < valid_m:
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        if instruction_bound is None or warp_start_m + (_i // warp_cols) * micro_size_x < instruction_bound:
                            T.make_filled_simdgroup_matrix(C_simd.data, _i, T.cast(0, accum_dtype))
                else:
                    mps_emitter.simd_load(C_simd, C_buf, valid_m=instruction_bound)
                multiply(A_local, B_local, C_simd, instruction_bound)
                mps_emitter.simd_store(C_simd, C_buf, valid_m=instruction_bound)

        return _Simplify(_gemm_ss_shared, inline_let=True)


class GemmMetal(GemmBase):
    supports_runtime_valid_m = True

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_gg(self) -> bool:
        return is_global(self.A) and is_global(self.B)

    @staticmethod
    def _valid_gg_warp_partitions(M: int, N: int, num_warps: int):
        for m_warp in range(1, num_warps + 1):
            if num_warps % m_warp != 0:
                continue
            n_warp = num_warps // m_warp
            if M % (m_warp * 16) == 0 and N % (n_warp * 32) == 0:
                yield m_warp, n_warp

    def _make_mps_emitter(self, target: Target, thread_nums: int):
        from tilelang.metal.intrinsics.metal_macro_generator import MPSIntrinEmitter

        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_METAL_COOPERATIVE_TENSOR)
        if self.is_gemm_gg():
            if int(thread_nums) % 32 != 0:
                raise ValueError(f"Metal cooperative tensor GG requires threads to be a multiple of 32, got {thread_nums}")
            num_warps = int(thread_nums) // 32
            if num_warps <= 0:
                raise ValueError(f"Metal cooperative tensor GG requires at least one warp, got {thread_nums} threads")
            candidates = list(self._valid_gg_warp_partitions(int(self.M), int(self.N), num_warps))
            if not candidates:
                raise ValueError(
                    "Metal cooperative tensor GG requires a warp partition "
                    f"where M is divisible by m_warp*16 and N by n_warp*32; "
                    f"got tile ({self.M}, {self.N}) with {num_warps} warps"
                )
            # Prefer partitions where each simdgroup owns a balanced grid of
            # 16x32 cooperative tensor operations. This keeps A/B cooperative
            # tensor load counts balanced for direct GG tiles.
            m_warp, n_warp = min(
                candidates,
                key=lambda part: (
                    abs(int(self.M) // (part[0] * 16) - int(self.N) // (part[1] * 32)),
                    -part[1],
                ),
            )
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        return (
            MPSIntrinEmitter(
                a_dtype=self.a_dtype,
                b_dtype=self.b_dtype,
                accum_dtype=self.accum_dtype,
                a_transposed=self.trans_A,
                b_transposed=self.trans_B,
                block_row_warps=m_warp,
                block_col_warps=n_warp,
                warp_row_tiles=warp_row_tiles,
                warp_col_tiles=warp_col_tiles,
                chunk=self.chunk,
            ),
            m_warp,
            n_warp,
        )

    def infer_layout(self, target: Target, thread_nums: int):
        result = {}
        if self.is_gemm_ss():
            result[self.A] = _make_padded_layout(self.A)
            result[self.B] = _make_padded_layout(self.B)
        if is_fragment(self.C):
            emitter, _, _ = self._make_mps_emitter(target, thread_nums)
            result[self.C] = emitter.make_cooperative_tensor_store_layout(self.C)
        return result

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tir.PrimExpr,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        _, m_warp, n_warp = self._make_mps_emitter(target, int(thread_nums))
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)

        from tilelang.metal.intrinsics.metal_macro_generator import MPSIntrinEmitter

        a_stride = _padded_stride(self.A) if self.is_gemm_ss() else None
        b_stride = _padded_stride(self.B) if self.is_gemm_ss() else None

        c_bytes_per_thread = warp_row_tiles * warp_col_tiles * 64
        inner_k_steps = 2 if c_bytes_per_thread <= 128 else 1
        output_dtype = self.accum_dtype
        accum_dtype = T.float32 if self.is_gemm_gg() and str(output_dtype) in ("float16", "bfloat16") else output_dtype
        mps_emitter = MPSIntrinEmitter(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_index,
            a_stride_override=a_stride,
            b_stride_override=b_stride,
            inner_k_steps=inner_k_steps,
        )

        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        warp_rows = mps_emitter.warp_rows
        warp_cols = mps_emitter.warp_cols
        num_simd_c = warp_rows * warp_cols
        block_K = mps_emitter.chunk
        micro_size_x = mps_emitter.micro_size_x
        micro_size_y = mps_emitter.micro_size_y
        micro_size_k = mps_emitter.micro_size_k
        inner_k_steps = mps_emitter.inner_k_steps
        a_tile_elems = micro_size_x * micro_size_k
        b_tile_elems = micro_size_k * micro_size_y
        c_tile_elems = micro_size_x * micro_size_y
        valid_m = _partial_m_extent(self)
        warp_m, _ = mps_emitter._get_warp_indices()

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion
        C_buf = C_region.buffer
        clear_accum = self.clear_accum
        c_in_cooperative_tensor = is_metal_cooperative_tensor(C_buf) or is_fragment(C_buf)
        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"

        if not (self.is_gemm_ss() or self.is_gemm_gg()):
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

        if c_in_cooperative_tensor:
            assert is_full_region(C_region), "Fragment output C must be a full region"

            @T.prim_func
            def _gemm_cooperative_tensor() -> None:
                A_local = T.alloc_local((warp_rows * a_tile_elems * inner_k_steps), a_dtype, scope="metal.cooperative_tensor")
                B_local = T.alloc_local((warp_cols * b_tile_elems * inner_k_steps), b_dtype, scope="metal.cooperative_tensor")
                if clear_accum:
                    for _i in T.serial(num_simd_c):
                        if valid_m is None or warp_m * warp_row_tiles + (_i // warp_cols) * micro_size_x < valid_m:
                            T.cooperative_tensor_fill(C_buf.data, _i, T.cast(0, accum_dtype), micro_size_x, micro_size_y)
                for k_outer in T.serial(0, (block_K // (micro_size_k * inner_k_steps))):
                    for k_inner in T.serial(0, inner_k_steps):
                        ki = k_outer * inner_k_steps + k_inner
                        mps_emitter.ldmatrix_a(A_local, A_region, ki, k_inner, valid_m)
                        mps_emitter.ldmatrix_b(B_local, B_region, ki, k_inner, valid_m)
                    for k_inner in T.serial(0, inner_k_steps):
                        mps_emitter.mma(A_local, B_local, C_buf, k_inner, valid_m)

            return _Simplify(_gemm_cooperative_tensor, inline_let=True)

        @T.prim_func
        def _gemm_with_c_writeback() -> None:
            A_local = T.alloc_local((warp_rows * a_tile_elems * inner_k_steps), a_dtype, scope="metal.cooperative_tensor")
            B_local = T.alloc_local((warp_cols * b_tile_elems * inner_k_steps), b_dtype, scope="metal.cooperative_tensor")
            C_ct = T.alloc_local((num_simd_c * c_tile_elems), accum_dtype, scope="metal.cooperative_tensor")
            if clear_accum:
                for _i in T.serial(num_simd_c):
                    if valid_m is None or warp_m * warp_row_tiles + (_i // warp_cols) * micro_size_x < valid_m:
                        T.cooperative_tensor_fill(C_ct.data, _i, T.cast(0, accum_dtype), micro_size_x, micro_size_y)
            else:
                mps_emitter.simd_load(C_ct, C_region, valid_m=valid_m)
            for k_outer in T.serial(0, (block_K // (micro_size_k * inner_k_steps))):
                for k_inner in T.serial(0, inner_k_steps):
                    ki = k_outer * inner_k_steps + k_inner
                    mps_emitter.ldmatrix_a(A_local, A_region, ki, k_inner, valid_m)
                    mps_emitter.ldmatrix_b(B_local, B_region, ki, k_inner, valid_m)
                for k_inner in T.serial(0, inner_k_steps):
                    mps_emitter.mma(A_local, B_local, C_ct, k_inner, valid_m)
            mps_emitter.simd_store(C_ct, C_region, valid_m=valid_m)

        return _Simplify(_gemm_with_c_writeback, inline_let=True)
