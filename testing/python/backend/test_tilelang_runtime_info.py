"""Runtime provenance uses device facts, never a fabricated performance profile."""

from types import SimpleNamespace

import pytest

from tilelang.backend.runtime import runtime_info


@pytest.mark.parametrize("backend,native", [("cuda", "cuda"), ("hip", "rocm"), ("metal", "metal")])
def test_runtime_identity_queries_the_selected_device(monkeypatch, backend, native):
    seen = []

    def device(kind, ordinal):
        seen.append((kind, ordinal))
        return SimpleNamespace(exist=True, api_version=12000, driver_version="12010")

    monkeypatch.setattr("tilelang.backend.runtime.tvm.device", device)
    result = runtime_info(SimpleNamespace(kind=SimpleNamespace(name=backend)), ordinal=2)
    assert seen == [(native, 2)]
    assert (result.api_version, result.driver_version) == ("12000", "12010")


def test_discrete_runtime_requires_reported_driver_version(monkeypatch):
    monkeypatch.setattr(
        "tilelang.backend.runtime.tvm.device", lambda *args: SimpleNamespace(exist=True, api_version=12000, driver_version=None)
    )
    with pytest.raises(RuntimeError, match="driver versions"):
        runtime_info(SimpleNamespace(kind=SimpleNamespace(name="cuda")))


def test_os_owned_metal_driver_is_explicitly_not_a_separate_version(monkeypatch):
    monkeypatch.setattr(
        "tilelang.backend.runtime.tvm.device", lambda *args: SimpleNamespace(exist=True, api_version=None, driver_version=None)
    )
    result = runtime_info(SimpleNamespace(kind=SimpleNamespace(name="metal")))
    assert result.driver_version is None
