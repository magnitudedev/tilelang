"""Python compiler changes must invalidate cached kernels in editable builds."""

from tilelang.cache import kernel_cache


def test_native_compiler_changes_invalidate_the_default_cache(monkeypatch):
    monkeypatch.delenv("TILELANG_KERNEL_CACHE_USE_LIB_STAMP", raising=False)
    assert kernel_cache.env.should_use_kernel_cache_lib_stamp()
    cache = kernel_cache.KernelCache
    monkeypatch.setattr(cache, "_get_tilelang_lib_stamp", staticmethod(lambda: "native-first"))
    first = cache._get_base_key()
    monkeypatch.setattr(cache, "_get_tilelang_lib_stamp", staticmethod(lambda: "native-second"))
    second = cache._get_base_key()
    assert first["tilelang_lib"] == "native-first"
    assert second["tilelang_lib"] == "native-second"
    assert first != second


def test_python_compiler_identity_tracks_content_and_relative_paths(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "cache").mkdir(parents=True)
        (root / "lower.py").write_text("strategy = 1\n")
    stamp = kernel_cache.KernelCache._get_python_source_stamp
    try:
        monkeypatch.setattr(kernel_cache, "__file__", str(first / "cache" / "kernel_cache.py"))
        stamp.cache_clear()
        before = stamp()
        monkeypatch.setattr(kernel_cache, "__file__", str(second / "cache" / "kernel_cache.py"))
        stamp.cache_clear()
        assert stamp() == before
        (second / "lower.py").write_text("strategy = 2\n")
        stamp.cache_clear()
        assert stamp() != before
    finally:
        stamp.cache_clear()
