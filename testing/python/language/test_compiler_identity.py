"""Public compiler provenance is the same authority as persistent kernel reuse."""

from tilelang.cache import compiler_identity
from tilelang.cache.kernel_cache import KernelCache


def test_compiler_identity_is_stable_and_responds_to_cache_provenance(monkeypatch):
    monkeypatch.setattr(KernelCache, "_get_tilelang_lib_stamp", staticmethod(lambda: "native-one"))
    monkeypatch.setattr(
        KernelCache,
        "_get_base_key",
        staticmethod(
            lambda: {
                "version": "test",
                "python_sources": "one",
                "tilelang_lib": "native-one",
            }
        ),
    )
    original = compiler_identity()
    assert compiler_identity() == original
    monkeypatch.setattr(
        KernelCache,
        "_get_base_key",
        staticmethod(
            lambda: {
                "tilelang_lib": "native-one",
                "python_sources": "one",
                "version": "test",
            }
        ),
    )
    assert compiler_identity() == original
    monkeypatch.setattr(
        KernelCache,
        "_get_base_key",
        staticmethod(
            lambda: {
                "version": "test",
                "python_sources": "two",
                "tilelang_lib": "native-one",
            }
        ),
    )
    assert compiler_identity() != original
