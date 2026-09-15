import pytest
from tilelang import tvm
from tilelang.analysis import plan_gemm
from tilelang.backend.module import create_backend_context


def test_gemm_analysis_reports_geometry_from_actual_fragment_lowering():
    target = create_backend_context("metal").target
    plan = plan_gemm(target=target, m=32, n=32, k=32, input_dtype="float32",
                     a_scope="local.fragment", b_scope="local.fragment")
    assert plan.shape == (32, 32, 32)
    assert plan.instruction_shape == (8, 8, 8)
    assert plan.input_precision == ("float32", "float32")
    assert plan.warp_partition == (2, 2)


def test_gemm_analysis_uses_selected_cooperative_geometry():
    config = dict(create_backend_context("metal").target.export())
    config["supports_metal4"] = True
    config["metal_language_version"] = 40
    target = tvm.target.Target(config)
    plan = plan_gemm(target=target, m=32, n=64, k=32, input_dtype="float16", c_scope="shared")
    assert plan.instruction_shape == (16, 32, 16)
    assert plan.shape == (32, 64, 32)


def test_analysis_propagates_real_lowering_rejection():
    with pytest.raises(Exception, match="divisible"):
        plan_gemm(target="metal", m=7, n=32, k=32, input_dtype="float32")
