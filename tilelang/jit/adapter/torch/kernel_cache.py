"""Disk cache entries for the torch Metal backend.

The Metal source is compiled by ``torch.mps.compile_shader`` at load time, so
an entry stores the generated source, the kernel parameters, and the launch
metadata the adapter needs to relaunch that source; there is no library file.
"""

import json
import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class TorchKernelCache(KernelCache):
    device_kernel_path = "device_kernel.metal"
    host_kernel_path = "host_kernel.metal"
    launch_path = "launch.json"

    def _get_required_files(self, cache_path: str) -> list[str]:
        return [
            os.path.join(cache_path, self.params_path),
            os.path.join(cache_path, self.launch_path),
        ]

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        # The shader is compiled from the cached source; no library exists.
        return

    def _save_adapter_metadata_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        launch_path = os.path.join(cache_path, self.launch_path)
        if verbose:
            self.logger.debug(f"Saving Metal launch metadata to file: {launch_path}")
        metadata = kernel.adapter.launch_metadata
        KernelCache._safe_write_file(launch_path, "w", lambda file: json.dump(metadata, file, indent=2, sort_keys=True))

    def _load_adapter_metadata_from_disk(self, cache_path: str, verbose: bool = False) -> dict | None:
        launch_path = os.path.join(cache_path, self.launch_path)
        if verbose:
            self.logger.debug(f"Loading Metal launch metadata from file: {launch_path}")
        with open(launch_path, encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError(f"Metal launch metadata at {launch_path} is not a JSON object")
        return metadata
