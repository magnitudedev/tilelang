"""Resource limits from a compilation target and, explicitly, its execution device."""
from dataclasses import dataclass

from tvm.target import Target


@dataclass(frozen=True)
class TargetResources:
    subgroup_width: int
    threads_per_group: int
    shared_memory_bytes: int


def target_resources(target: Target, *, device=None) -> TargetResources:
    """Read target constraints; resolve absent facts only from the supplied device.

    Omitting a device is the offline path. An incomplete GPU target is an error,
    not evidence that the target has no shared memory. Device queries do not
    rewrite the compilation target or its architecture/feature suffix.
    """
    if device is not None and device.dlpack_device_type() != target.get_target_device_type():
        raise ValueError("execution device and compilation target have different device types")
    attrs = target.attrs
    cpu = target.get_target_device_type() == 1

    def limit(names, property_name, cpu_default):
        for name in names:
            value = attrs.get(name)
            if value is not None:
                return int(value)
        if cpu:
            return cpu_default
        if device is not None:
            if not device.exist:
                raise ValueError(f"execution device {device} is unavailable")
            value = getattr(device, property_name)
            if value is not None:
                return int(value)
        raise ValueError(f"target resource {names[0]} is unknown; supply an explicit limit or its execution device")

    result = TargetResources(
        limit(("thread_warp_size",), "warp_size", 1),
        limit(("max_threads_per_block", "max_num_threads"), "max_threads_per_block", 1),
        limit(("max_shared_memory_per_block",), "max_shared_memory_per_block", 0),
    )
    if result.subgroup_width <= 0 or result.threads_per_group <= 0 or result.shared_memory_bytes < 0:
        raise ValueError(f"invalid target resource limits: {result}")
    return result
