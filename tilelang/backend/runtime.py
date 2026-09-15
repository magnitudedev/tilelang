"""Public, read-only identity facts from the selected native runtime.

These are native API version values, not performance estimates or hardware rate
tables. In particular, CUDA's driver_version is the CUDA compatibility version
reported by cudaDriverGetVersion, not a unique installed-driver build identity.
"""

from dataclasses import dataclass

import tvm
from tvm.target import Target


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    backend: str
    ordinal: int
    api_version: str | None
    driver_version: str | None


def runtime_info(target: Target, *, ordinal: int = 0) -> RuntimeInfo:
    """Query the public runtime device attributes, without compiling a kernel.

    Metal's driver is part of the operating system and has no separate version in
    this interface. Callers must include OS build identity in persisted evidence.
    """
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("device ordinal must be a nonnegative integer")
    backend = target.kind.name
    devices = {"cuda": "cuda", "hip": "rocm", "metal": "metal", "llvm": "cpu", "c": "cpu"}
    if backend not in devices:
        raise ValueError(f"runtime identity is not exposed for target {backend!r}")
    device = tvm.device(devices[backend], ordinal)
    if not device.exist:
        raise ValueError(f"runtime device {backend}:{ordinal} is unavailable")
    api, driver = device.api_version, device.driver_version
    if backend in ("cuda", "hip") and (api is None or driver is None):
        raise RuntimeError(f"{backend} runtime must expose API and driver versions")
    return RuntimeInfo(backend, ordinal, None if api is None else str(api), None if driver is None else str(driver))
