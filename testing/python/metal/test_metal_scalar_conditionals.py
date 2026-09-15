"""Lazy conditional operands must finish emitting before scalar statements."""

import re

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


def scalar_conditional(initialize):
    @T.prim_func
    def main(packed: T.Tensor((17,), "uint32"), output: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            lane = T.get_thread_binding(0)
            if initialize:
                scale = T.alloc_var(
                    "float32",
                    init=T.reinterpret(T.if_then_else(lane < 17, packed[lane], T.uint32(0)), "float32"),
                )
                output[lane] = scale
            else:
                scale = T.alloc_var("float32")
                scale = T.reinterpret(T.if_then_else(lane < 17, packed[lane], T.uint32(0)), "float32")
                output[lane] = scale

    return main


@pytest.mark.parametrize("initialize", [False, True])
def test_scalar_conditional_source(initialize):
    with tvm.transform.PassContext():
        source = tilelang.lower(scalar_conditional(initialize), target="metal").kernel_source
    assert "condval" in source, source
    assert not re.search(r"=\s*uint\s+condval", source), source
    declaration = re.search(r"uint\s+(condval\w*)\s*;", source)
    assert declaration is not None, source
    assignment = re.search(r"scale\w*\s*=\s*\(?as_type<float>", source)
    assert assignment is not None, source
    assert declaration.start() < assignment.start(), source


def test_annotated_scalar_initializer_source():
    # alloc_var(init=...) currently emits a BufferStore. Exercise the separate
    # allocation-annotation ABI as well, without relying on that DSL lowering.
    packed = tirx.decl_buffer((1,), "uint32", name="packed")
    output = tirx.decl_buffer((1,), "float32", name="output")
    scale = tirx.decl_buffer((1,), "float32", name="scale", scope="local.var")
    value = tirx.reinterpret("float32", tirx.if_then_else(packed[0] != 0, packed[0], tirx.const(0, "uint32")))
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(scale, annotations={"tl.local_var_init": value}),
            tirx.BufferStore(output, scale[0], [0]),
        ]
    )
    func = tirx.PrimFunc(
        [packed.data, output.data],
        body,
        buffer_map={packed.data: packed, output.data: output},
    ).with_attr("global_symbol", "scalar_initializer")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    func = func.with_attr("tirx.kernel_launch_params", [])
    build = tvm.get_global_func("target.build.tilelang_metal_without_compile")
    with tvm.transform.PassContext():
        source = build(tvm.IRModule({"scalar_initializer": func}), tvm.target.Target("metal")).inspect_source()
    assert not re.search(r"=\s*uint\s+condval", source), source
    assert source.index("uint condval;") < source.index("float scale ="), source


@tilelang.testing.requires_metal
@pytest.mark.parametrize("initialize", [False, True])
def test_scalar_conditional_execution(initialize):
    values = np.linspace(-3, 5, 17, dtype=np.float32)
    packed = torch.from_numpy(values.view(np.uint32).copy()).to("mps")
    output = torch.empty(32, dtype=torch.float32, device="mps")
    kernel = tilelang.compile(scalar_conditional(initialize), target="metal", execution_backend="tvm_ffi", out_idx=[])
    kernel.bind({0: packed, 1: output}, ())()
    torch.mps.synchronize()
    expected = np.zeros(32, dtype=np.float32)
    expected[:17] = values
    np.testing.assert_array_equal(output.cpu().numpy(), expected)
