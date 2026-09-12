import struct

import tilelang
import tilelang.language as T
import tilelang.testing
import torch


@tilelang.testing.requires_metal
def test_reinterpret_narrow_integer_expression():
    @T.prim_func
    def kernel(source: T.Tensor((2,), T.uint8), output: T.Tensor((1,), T.float16)):
        with T.Kernel(1, threads=1):
            bits = T.cast(source[0], T.uint16)
            bits |= T.cast(source[1], T.uint16) << 8
            output[0] = T.reinterpret(bits, T.float16)

    compiled = tilelang.compile(kernel, out_idx=[], target="metal")
    source = torch.tensor(list(struct.pack("<e", 1.5)), dtype=torch.uint8, device="mps")
    output = torch.empty((1,), dtype=torch.float16, device="mps")
    compiled(source, output)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), torch.tensor([1.5], dtype=torch.float16))


if __name__ == "__main__":
    tilelang.testing.main()
