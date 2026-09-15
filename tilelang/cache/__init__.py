"""The cache utils with class and database persistence - Init file"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal
from tvm.target import Target as TVMTarget
from tvm.tirx import PrimFunc
from tvm import IRModule
from tilelang.jit import JITKernel
from tilelang import env
from tilelang.jit.adapter.cutedsl.kernel_cache import CuTeDSLKernelCache
from tilelang.jit.adapter.cython.kernel_cache import CythonKernelCache
from tilelang.jit.adapter.nvrtc.kernel_cache import NVRTCKernelCache
from tilelang.jit.adapter.torch.kernel_cache import TorchKernelCache
from tilelang.jit.adapter.kernel_cache import TVMFFIKernelCache

if TYPE_CHECKING:
    from .kernel_cache import KernelCache

TargetLike = str | dict[str, object] | TVMTarget


def compiler_identity() -> str:
    """Identity of the compiler installation used by the persistent kernel cache.

    This excludes program, target, schedules and tuning inputs, which callers
    identify separately. Like the cache itself, editable compiler Python/native
    changes require a fresh process; this is not a hot-reload mechanism.
    """
    from hashlib import sha256
    import json

    from .kernel_cache import KernelCache

    provenance = dict(KernelCache._get_base_key())
    # Evidence provenance must not omit native identity because a caller disabled
    # that component of cache invalidation policy.
    native = KernelCache._get_tilelang_lib_stamp()
    if native is None:
        raise RuntimeError("cannot identify the installed TileLang native compiler libraries")
    provenance["tilelang_lib"] = native
    return sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# Create a map of singleton instance of KernelCaches
_dispatch_map: dict[str, KernelCache] = {
    "tvm_ffi": TVMFFIKernelCache(),
    "cython": CythonKernelCache(),
    "nvrtc": NVRTCKernelCache(),
    "cutedsl": CuTeDSLKernelCache(),
    "torch": TorchKernelCache(),
}


def _resolve_cache_dispatch(
    target: TargetLike | None,
    target_host: TargetLike | None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None,
    verbose: bool | None,
):
    if target is None:
        target = env.get_default_target()
    if execution_backend is None:
        execution_backend = env.get_default_execution_backend()
    if verbose is None:
        verbose = env.get_default_verbose()

    from tilelang.backend.module import create_backend_context

    requested_backend = execution_backend
    context = create_backend_context(target, target_host, requested_backend)
    resolved_backend = context.execution_backend.name
    if verbose:
        allowed_now = context.module.allowed_execution_backends(context.target, include_unavailable=False)
        if requested_backend in (None, "auto") or requested_backend != resolved_backend:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.info(
                "Execution backend resolved -> '%s' (requested='%s', target='%s', allowed: %s)",
                resolved_backend,
                requested_backend,
                context.target.kind.name,
                ", ".join(sorted(allowed_now)),
            )
    if resolved_backend not in _dispatch_map:
        raise ValueError(f'Cannot find support for execution backend "{resolved_backend}"')
    return _dispatch_map[resolved_backend], context, verbose


def cached(
    func: PrimFunc | IRModule = None,
    out_idx: list[int] = None,
    *args,
    target: TargetLike | None = None,
    target_host: TargetLike | None = None,
    execution_backend: Literal["auto", "tvm_ffi", "cython", "nvrtc", "torch", "cutedsl"] | None = None,
    verbose: bool | None = None,
    pass_configs: dict | None = None,
    compile_flags: list[str] | str | None = None,
) -> JITKernel:
    """
    Caches and reuses compiled kernels (using KernelCache class).
    """
    cache, backend_context, verbose = _resolve_cache_dispatch(target, target_host, execution_backend, verbose)
    return cache.cached(
        func,
        out_idx,
        *args,
        backend_context=backend_context,
        verbose=verbose,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
