"""Exact constants survive the complete production Metal compilation path."""

import numpy as np
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_metal
def test_float32_literal_bits():
    @T.prim_func
    def constants(output: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding(0) == 0:
                output[0] = T.float32(1 / 4096)
                output[1] = T.float32(1.0000001192092896)
                output[2] = T.float32(2**-126)
                output[3] = T.float32(2**-114)
                output[4] = T.float32(3.4028234663852886e38)
                output[5] = T.float32(-3.4028234663852886e38)
                output[6] = T.float32(0.0)
                output[7] = T.float32(-0.0)

    kernel = tilelang.compile(constants, target="metal", execution_backend="tvm_ffi", out_idx=[])
    output = torch.empty(8, dtype=torch.float32, device="mps")
    kernel.bind({0: output}, ())()
    torch.mps.synchronize()
    expected = np.array(
        [1 / 4096, 1.0000001192092896, 2**-126, 2**-114, np.finfo(np.float32).max, -np.finfo(np.float32).max, 0.0, -0.0], dtype=np.float32
    )
    assert output.cpu().numpy().tobytes() == expected.tobytes()
