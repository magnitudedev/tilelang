from types import SimpleNamespace

import pytest
from tilelang import tvm
from tilelang.backend.resources import target_resources


def test_missing_cuda_shared_memory_is_unknown_not_zero():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_121a"})
    with pytest.raises(ValueError, match="max_shared_memory_per_block is unknown"):
        target_resources(target)
    device = SimpleNamespace(dlpack_device_type=lambda: 2, exist=True,
                             max_shared_memory_per_block=49152, warp_size=32,
                             max_threads_per_block=1024)
    resources = target_resources(target, device=device)
    assert resources.shared_memory_bytes == 49152
    assert resources.subgroup_width == 32
    assert str(target.attrs["arch"]) == "sm_121a"
    assert "max_shared_memory_per_block" not in target.attrs


def test_explicit_target_limits_override_execution_device():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80",
                               "max_threads_per_block": 64, "max_shared_memory_per_block": 8192})
    device = SimpleNamespace(dlpack_device_type=lambda: 2, exist=True,
                             max_shared_memory_per_block=49152, warp_size=32,
                             max_threads_per_block=1024)
    resources = target_resources(target, device=device)
    assert resources.shared_memory_bytes == 8192
    assert resources.threads_per_group == 64


def test_explicit_zero_is_not_replaced_by_device_capacity():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80", "max_shared_memory_per_block": 0})
    assert target_resources(target).shared_memory_bytes == 0


def test_device_must_match_target():
    with pytest.raises(ValueError, match="different device types"):
        target_resources(tvm.target.Target("cuda"), device=SimpleNamespace(dlpack_device_type=lambda: 8))
