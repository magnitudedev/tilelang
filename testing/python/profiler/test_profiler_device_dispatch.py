"""Validation must synchronize the tensors' device, not ambient CUDA."""

import pytest
import torch

from tilelang.profiler import Profiler
from tilelang.utils.tensor import TensorSupplyType


def test_cpu_validation_does_not_initialize_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CPU validation attempted CUDA synchronization")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    profiler = Profiler([], [], TensorSupplyType.Auto, adapter=lambda x: x + 1)
    profiler.assert_allclose(lambda x: x + 1, input_tensors=[torch.ones(8)])
    with pytest.raises(AssertionError):
        profiler.assert_allclose(lambda x: x + 2, input_tensors=[torch.ones(8)], max_mismatched_ratio=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_validation():
    x = torch.ones(128, device="cuda")
    profiler = Profiler([], [], TensorSupplyType.Auto, adapter=lambda x: x + 1)
    profiler.assert_allclose(lambda x: x + 1, input_tensors=[x], max_mismatched_ratio=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")
def test_mps_validation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("MPS profiling attempted CUDA synchronization")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    x = torch.ones(128, device="mps")
    profiler = Profiler([], [], TensorSupplyType.Auto, adapter=lambda x: x + 1)
    profiler.assert_allclose(lambda x: x + 1, input_tensors=[x], max_mismatched_ratio=0)
    profiler.manual_assert_close(
        lambda x: x + 1, input_tensors=[x], manual_check_prog=lambda actual, expected: torch.testing.assert_close(actual, expected)
    )
