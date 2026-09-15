"""Optional native kernel timestamps on the production execution stream.

No compilation, replay, queue commit or synchronization is performed here.
The caller owns execution completion before resolving a capture.
"""

from dataclasses import dataclass
from threading import get_ident

import tvm
from tvm.target import Target


@dataclass(frozen=True, slots=True)
class KernelTiming:
    name: str
    elapsed_ns: int


class KernelCapture:
    """A single bounded capture, confined to its creating thread."""

    def __init__(self, module, clock: str):
        self._module = module
        self._owner = get_ident()
        self.clock = clock

    def _require(self):
        if get_ident() != self._owner:
            raise RuntimeError("kernel capture belongs to its creating thread")
        if self._module is None:
            raise RuntimeError("kernel capture is closed")
        return self._module

    def start(self) -> None:
        self._require()["start"]()

    def finish(self) -> tuple[KernelTiming, ...]:
        values = self._require()["finish"]()
        return tuple(KernelTiming(str(value["name"]), int(value["elapsed_ns"])) for value in values)

    def close(self) -> None:
        if get_ident() != self._owner:
            raise RuntimeError("kernel capture belongs to its creating thread")
        if self._module is not None:
            self._module["close"]()
            self._module = None


def kernel_capture(target: Target, *, ordinal: int = 0, max_kernels: int = 1024) -> KernelCapture | None:
    """Create bounded counter storage, or return None for unsupported endpoints.

    This factory does not start observation. Once started, capacity exhaustion or
    invalid/unfinished samples fail the capture instead of returning partial data.
    Counter storage is finite and device-specific: requesting more sample storage
    than the driver supports fails allocation rather than shrinking the request.
    """
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("device ordinal must be a nonnegative integer")
    if type(max_kernels) is not int or not 1 <= max_kernels <= 65536:
        raise ValueError("kernel capture capacity must be between 1 and 65536")
    backend = target.kind.name
    if backend != "metal":
        return None
    factory = tvm.get_global_func("runtime.metal.CreateKernelCapture", allow_missing=True)
    if factory is None:
        return None
    module = factory(ordinal, max_kernels)
    return None if module is None else KernelCapture(module, "metal-stage-timestamps-calibrated-v1")
