"""Let-bound shared gathers retain physical loop coordinates after partitioning."""
import numpy as np
import pytest
import tilelang
import tilelang.language as T


def test_shared_gather_into_matrix_operand():
    torch = pytest.importorskip('torch')
    if not torch.backends.mps.is_available():
        pytest.skip('Metal device required')

    @T.prim_func
    def main(source: T.Tensor((4, 64), 'uint32'), centers: T.Tensor((16,), 'float32'),
             query: T.Tensor((8, 32), 'float32'), output: T.Tensor((8, 64), 'float32')):
        with T.Kernel(1, threads=128):
            packed = T.alloc_shared((4, 64), 'uint32')
            table = T.alloc_shared((16,), 'float32', scope='shared')
            q = T.alloc_fragment((8, 32), 'float32')
            k = T.alloc_fragment((32, 64), 'float32')
            scores = T.alloc_fragment((8, 64), 'float32')
            T.copy(source, packed)
            T.copy(centers, table)
            T.copy(query, q)
            for channel, key in T.Parallel(32, 64):
                coordinate = channel
                packed_word = packed[coordinate // 8, key]
                code = (packed_word >> ((coordinate % 8) * 4)) & 15
                k[channel, key] = table[T.cast(code, 'int32')]
            T.gemm(q, k, scores, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            T.copy(scores, output)

    kernel = tilelang.compile(main, target='metal', execution_backend='tvm_ffi', out_idx=[],
                              pass_configs={'tl.force_let_inline': False})
    rng = np.random.default_rng(2137)
    words = rng.integers(0, 2**32, (4, 64), dtype=np.uint32)
    centers = rng.normal(size=16).astype(np.float32)
    query = rng.normal(size=(8, 32)).astype(np.float32)
    codes = np.stack([(words[channel // 8] >> ((channel % 8) * 4)) & 15 for channel in range(32)])
    expected = query @ centers[codes]
    a = torch.from_numpy(words).to('mps')
    b = torch.from_numpy(centers).to('mps')
    q = torch.from_numpy(query).to('mps')
    out = torch.empty((8, 64), dtype=torch.float32, device='mps')
    kernel(a, b, q, out)
    np.testing.assert_allclose(out.cpu().numpy(), expected, rtol=2e-5, atol=2e-5)
