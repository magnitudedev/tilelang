"""Native timestamps use the production bound FFI entrypoint and its stream."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.backend import kernel_capture


def two_passes():
    @T.prim_func
    def main(X: T.Tensor((256,), "float32"), M: T.Tensor((256,), "float32"), Y: T.Tensor((256,), "float32")):
        with T.Kernel(4, threads=64) as block:
            i = block * 64 + T.get_thread_binding(0)
            M[i] = X[i] + 1
        with T.Kernel(4, threads=64) as block:
            i = block * 64 + T.get_thread_binding(0)
            Y[i] = M[i] * 3

    return main


def capture(limit=2):
    result = kernel_capture(tvm.target.Target("metal"), max_kernels=limit)
    if result is None:
        pytest.skip("device does not expose stage-boundary timestamp counters")
    return result


@pytest.fixture(scope="module")
def bound():
    kernel = tilelang.compile(two_passes(), target="metal", execution_backend="tvm_ffi", out_idx=[])
    source = torch.arange(256, dtype=torch.float32, device="mps")
    middle, output = torch.empty_like(source), torch.empty_like(source)
    entry = kernel.bind({0: source, 1: middle, 2: output}, ())
    torch.mps.synchronize()
    return kernel, entry, output


def test_unsupported_backend_and_invalid_capacity_are_explicit():
    assert kernel_capture(tvm.target.Target("llvm")) is None
    for limit in (0, -1, 65537, True):
        with pytest.raises(ValueError, match="capacity"):
            kernel_capture(tvm.target.Target("metal"), max_kernels=limit)


@tilelang.testing.requires_metal
def test_native_compute_timestamps_on_bound_multi_dispatch_entrypoint(bound):
    kernel, entry, output = bound
    source = kernel.adapter.host_mod.inspect_source()
    assert "pthread_self()" in source
    assert "TVMFFIErrorMoveFromRaised" in source
    assert "TVMFFIErrorSetRaised" in source
    with_capture = capture()
    try:
        with_capture.start()
        entry()
        torch.mps.synchronize()
        timings = with_capture.finish()
        assert len(timings) == 2
        assert all(item.name and item.elapsed_ns > 0 for item in timings)
        assert [item.dispatch for item in timings] == [0, 1]
        assert all(0 <= item.started_ns < item.ended_ns for item in timings)
        # Endpoints and elapsed use the same calibrated, capture-relative clock.
        assert all(abs(item.ended_ns - item.started_ns - item.elapsed_ns) <= 1 for item in timings)
        assert with_capture.clock == "metal-stage-timestamps-capture-relative-v2"
        torch.testing.assert_close(output.cpu(), (torch.arange(256, dtype=torch.float32) + 1) * 3)
    finally:
        with_capture.close()
    # Instrumentation does not change ordinary execution after it is detached.
    entry()
    torch.mps.synchronize()


@tilelang.testing.requires_metal
def test_incomplete_capture_does_not_commit_or_wait(bound):
    _, entry, _ = bound
    active = capture()
    try:
        active.start()
        entry()
        with pytest.raises(Exception, match="successfully completed command buffers"):
            active.finish()
    finally:
        active.close()
        torch.mps.synchronize()


@tilelang.testing.requires_metal
def test_overflow_reaches_submitting_thread_and_capture_can_be_replaced(bound):
    _, entry, _ = bound
    active = capture(1)
    try:
        active.start()
        with pytest.raises(Exception, match="Kernel timestamp capacity exhausted"):
            entry()
        torch.mps.synchronize()
        with pytest.raises(Exception, match="partial timestamps are not valid"):
            active.finish()
    finally:
        active.close()
        torch.mps.synchronize()
    following = capture()
    try:
        following.start()
        entry()
        torch.mps.synchronize()
        assert len(following.finish()) == 2
    finally:
        following.close()


@tilelang.testing.requires_metal
def test_capture_owner_nesting_and_single_use():
    first, second = capture(), capture()
    try:
        with ThreadPoolExecutor(max_workers=1) as worker, pytest.raises(RuntimeError, match="creating thread"):
            worker.submit(first.start).result()
        first.start()
        with pytest.raises(Exception, match="already active"):
            second.start()
        second.close()
        assert first.finish() == ()
        with pytest.raises(Exception, match="single-use"):
            first.start()
    finally:
        first.close()
        second.close()


@tilelang.testing.requires_metal
def test_other_submitter_is_not_attributed_to_active_capture(bound):
    _, entry, _ = bound
    active = capture()
    try:
        active.start()
        with ThreadPoolExecutor(max_workers=1) as worker:
            worker.submit(entry).result()
        torch.mps.synchronize()
        assert active.finish() == ()
    finally:
        active.close()
