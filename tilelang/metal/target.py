from __future__ import annotations

from platform import mac_ver

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def check_metal_availability() -> bool:
    mac_release, _, arch = mac_ver()
    if not mac_release:
        return False
    # todo: check torch version?
    return arch == "arm64"


def check_metal4_availability() -> bool:
    if not check_metal_availability():
        return False
    return bool(Target.from_device("metal").attrs["supports_metal4"])


def _detect_metal_target() -> Target | None:
    if not check_metal_availability():
        return None
    target = Target.from_device("metal")
    if bool(target.attrs.get("supports_metal4", False)):
        config = dict(target.export())
        config["keys"] = [*target.keys, "metal4"]
        return Target(config)
    return target


def validate_metal_target(target: Target) -> Target:
    """Reject contradictory explicit SIMD geometry without changing the target."""
    attrs = target.attrs
    if (attrs.get("supports_simdgroup_matrix", False) or attrs.get("supports_simdgroup_reduction", False)) and int(
        attrs.get("thread_warp_size", 16)
    ) != 32:
        raise ValueError("Metal SIMD-group operations require thread_warp_size=32")
    return target


def normalize_metal_target(target: TargetLike) -> Target | None:
    """Detect a bare backend name; preserve explicit compilation targets.

    A Target (or dictionary) describes the requested compilation environment,
    including offline targets. Never replace its limits or feature flags with
    properties of the machine performing compilation.
    """
    if isinstance(target, Target):
        return validate_metal_target(target) if target.kind.name == "metal" else None
    if isinstance(target, dict):
        return validate_metal_target(Target(target)) if target.get("kind") == "metal" else None
    if target.strip() == "metal":
        detected = _detect_metal_target()
        return detected if detected is not None else Target("metal")
    return None


def target_is_metal(target: Target) -> bool:
    return _target_ffi_api().TargetIsMetal(target)


def target_metal_supports_metal4(target: Target) -> bool:
    return bool(target.attrs.get("supports_metal4", False))


register_target_detector("metal", _detect_metal_target, override=True)
register_target_normalizer("metal", normalize_metal_target, override=True)
