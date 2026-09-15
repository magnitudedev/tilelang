"""Metal backend manifest."""

from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.capabilities import MatrixInstruction, target_limits
from tilelang.backend.host_codegen import HostCodegenHook, STANDARD_HOST_CODEGENS
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, execution_backend, pipeline
from .target import validate_metal_target


def _capabilities(target):
    attrs = validate_metal_target(target).attrs
    supports_bfloat16 = bool(attrs.get("supports_bfloat16", False))
    supports_matrix = bool(attrs.get("supports_simdgroup_matrix", False))
    matrix_instructions = []
    if supports_matrix:
        matrix_instructions.append(MatrixInstruction(8, 8, 8, "float16", "float32"))
        matrix_instructions.append(MatrixInstruction(8, 8, 8, "float32", "float32"))
        if supports_bfloat16:
            matrix_instructions.append(MatrixInstruction(8, 8, 8, "bfloat16", "float32"))
    supported_dtypes = {
        "bool",
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "float32",
    }
    if supports_bfloat16:
        supported_dtypes.add("bfloat16")
    features = {"gemm.runtime_valid_m", "atomic.add.int32"}
    if bool(attrs.get("supports_simdgroup_reduction", False)):
        features.add("subgroup_exchange")
    return target_limits(
        target,
        subgroup_width=32,
        matrix_instructions=tuple(matrix_instructions),
        features=frozenset(features),
        supported_dtypes=frozenset(supported_dtypes),
    )


BACKEND = register_backend(
    BackendModule(
        name="metal",
        target_kinds=("metal",),
        pipelines={"metal": PassPipeline("metal", pipeline.MetalPassPipelineBody)},
        device_codegens={
            "metal": DeviceCodegen(
                "metal",
                build=codegen.build_metal,
                build_without_compile=codegen.build_metal_without_compile,
            )
        },
        host_codegen_hooks={"metal": (HostCodegenHook("metal_context", codegen.mark_host_metal_context),)},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
        capabilities=_capabilities,
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
