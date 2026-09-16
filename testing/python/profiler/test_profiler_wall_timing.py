"""Explicit wall timing includes completion on the supplied tensor device."""

import pytest
import torch

from tilelang.profiler import Profiler
from tilelang.utils.tensor import TensorSupplyType


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required")),
    ],
)
def test_wall_timing_uses_input_device(device, monkeypatch):
    if device != "cuda":

        def unexpected(*args, **kwargs):
            pytest.fail("non-CUDA profiling attempted CUDA synchronization")

        monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    x = torch.ones(128, device=device)
    profiler = Profiler([], [], TensorSupplyType.Auto, adapter=lambda x: x + 1)
    assert profiler.do_bench(input_tensors=[x], n_warmup=2, n_repeat=3, backend="wall") > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_event_timing_remains_available():
    x = torch.ones(128, device="cuda")
    profiler = Profiler([], [], TensorSupplyType.Auto, adapter=lambda x: x + 1)
    assert profiler.do_bench(input_tensors=[x], n_warmup=2, n_repeat=3, backend="event") > 0
