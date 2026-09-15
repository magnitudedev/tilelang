"""Device detection and explicit compilation targets have separate meanings."""

import pytest

from tilelang import tvm
from tilelang.backend import create_backend_context
from tilelang.backend.target import determine_target
from tilelang.metal import target as metal_target


def test_explicit_targets_do_not_query_or_inherit_local_device(monkeypatch):
    def unexpected_detection():
        raise AssertionError("Explicit targets must not query the local device")

    monkeypatch.setattr(metal_target, "_detect_metal_target", unexpected_detection)
    config = {
        "kind": "metal",
        "thread_warp_size": 32,
        "max_threads_per_block": 64,
        "metal_language_version": 23,
        "supports_bfloat16": False,
    }
    explicit = tvm.target.Target(config)
    assert metal_target.normalize_metal_target(explicit) is explicit
    for value in (config, explicit):
        resolved = determine_target(value, return_object=True)
        tvm.ir.assert_structural_equal(resolved, explicit)
        context = create_backend_context(value, execution_backend="tvm_ffi")
        assert context.target.attrs["max_threads_per_block"] == 64
        assert not context.target.attrs["supports_bfloat16"]
        assert context.target.attrs["thread_warp_size"] == 32


def test_bare_backend_detects_once_and_is_idempotent(monkeypatch):
    detected = tvm.target.Target(
        {
            "kind": "metal",
            "thread_warp_size": 32,
            "max_threads_per_block": 1024,
            "metal_language_version": 31,
            "supports_bfloat16": True,
        }
    )
    calls = []

    def detect():
        calls.append(True)
        return detected

    monkeypatch.setattr(metal_target, "_detect_metal_target", detect)
    first = determine_target("metal", return_object=True)
    second = determine_target(first, return_object=True)
    assert calls == [True]
    tvm.ir.assert_structural_equal(first, detected)
    tvm.ir.assert_structural_equal(second, detected)


def test_active_explicit_target_is_preserved(monkeypatch):
    def unexpected_detection():
        raise AssertionError("An active target is explicit")

    monkeypatch.setattr(metal_target, "_detect_metal_target", unexpected_detection)
    target = tvm.target.Target({"kind": "metal", "max_threads_per_block": 64})
    with target:
        tvm.ir.assert_structural_equal(determine_target("auto", return_object=True), target)


def test_bare_backend_supports_offline_compilation(monkeypatch):
    monkeypatch.setattr(metal_target, "check_metal_availability", lambda: False)
    target = determine_target("metal", return_object=True)
    assert target.kind.name == "metal"
    assert int(target.attrs["thread_warp_size"]) == 16


@pytest.mark.parametrize("metal4", [False, True])
def test_detection_uses_tvm_device_target(monkeypatch, metal4):
    detected = tvm.target.Target(
        {
            "kind": "metal",
            "thread_warp_size": 32,
            "max_threads_per_block": 512,
            "max_shared_memory_per_block": 65536,
            "supports_metal4": metal4,
            "metal_language_version": 40 if metal4 else 32,
        }
    )
    monkeypatch.setattr(metal_target, "check_metal_availability", lambda: True)
    monkeypatch.setattr(tvm.target.Target, "from_device", staticmethod(lambda dev: detected))
    resolved = determine_target("metal", return_object=True)
    assert dict(resolved.attrs) == dict(detected.attrs)
    assert ("metal4" in resolved.keys) == metal4
    assert metal_target.check_metal4_availability() == metal4


@pytest.mark.parametrize("as_object", [False, True])
def test_explicit_simd_geometry_is_validated_before_compilation(as_object):
    config = {"kind": "metal", "thread_warp_size": 16, "supports_simdgroup_matrix": True}
    target = tvm.target.Target(config) if as_object else config
    with pytest.raises(ValueError, match="thread_warp_size=32"):
        create_backend_context(target, execution_backend="tvm_ffi")
