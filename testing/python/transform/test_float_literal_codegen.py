"""FloatImm source serialization must round-trip, independently of a GPU."""

import re

import numpy as np
import pytest

from tilelang import tvm
from tvm import tirx
from tvm_ffi.registry import remove_global_func


@pytest.mark.parametrize(
    "backend,dtype",
    [
        ("tilelang_metal_without_compile", "float32"),
        ("tilelang_metal_without_compile", "float16"),
        ("metal", "float32"),
        ("metal", "float16"),
        ("c", "float32"),
        ("c", "float64"),
        ("tilelang_hip_without_compile", "float32"),
        ("tilelang_hip_without_compile", "float64"),
    ],
)
def test_float_literals_round_trip(backend, dtype):
    build = tvm.get_global_func(f"target.build.{backend}", allow_missing=True)
    if build is None:
        pytest.skip(f"{backend} code generator is not built")
    scalar = np.dtype(dtype).type
    limits = np.finfo(dtype)
    values = np.array(
        [
            1 / 4096,
            np.nextafter(scalar(1), scalar(2)),
            limits.tiny,
            np.nextafter(scalar(limits.tiny), scalar(1)),
            limits.max,
            -limits.max,
            0.0,
            -0.0,
        ],
        dtype=dtype,
    )
    output = tirx.decl_buffer((len(values),), dtype, name="output")
    body = tirx.SeqStmt([tirx.BufferStore(output, tirx.FloatImm(dtype, float(value)), [index]) for index, value in enumerate(values)])
    func = tirx.PrimFunc([output.data], body, buffer_map={output.data: output})
    func = func.with_attr("global_symbol", "literal_round_trip")
    if backend != "c":
        func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
        func = func.with_attr("tirx.kernel_launch_params", [])
    module = tvm.IRModule({"literal_round_trip": func})
    target = "metal" if "metal" in backend else "rocm" if "hip" in backend else "c"
    # TVM's builder otherwise invokes a globally registered compiler callback.
    # Removing it requests its supported source-only module path; restore it
    # even when code generation fails so other tests retain their environment.
    callback_name = "tvm_callback_metal_compile"
    callback = tvm.get_global_func(callback_name, allow_missing=True)
    if backend == "metal" and callback is not None:
        remove_global_func(callback_name)
    try:
        source = build(module, tvm.target.Target(target)).inspect_source()
    finally:
        if backend == "metal" and callback is not None:
            tvm.register_global_func(callback_name, callback, override=True)
    literals = re.findall(r"=\s*(?:\([^)]*\))?(-?\d+\.\d+e[+-]\d+)[fh]?;", source)
    assert len(literals) == len(values), source
    actual = np.array([float(value) for value in literals], dtype=dtype)
    # Byte comparison also distinguishes negative from positive zero.
    assert actual.tobytes() == values.tobytes(), source
