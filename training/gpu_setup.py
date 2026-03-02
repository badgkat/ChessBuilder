# training/gpu_setup.py
"""Configure GPU environment for AMD ROCm.

RDNA 4 (gfx1151, e.g. Radeon 8060S) requires HSA_OVERRIDE_GFX_VERSION=11.0.0
because MIOpen's kernel database doesn't include gfx1151 yet. This makes it
compile kernels for gfx1100 (RDNA 3) which are instruction-compatible.

This must be set before torch imports any HIP/ROCm libraries, so import this
module before torch in any training entry point.
"""

import os
import subprocess


def configure_rocm():
    """Set HSA_OVERRIDE_GFX_VERSION if running on an unsupported AMD GPU."""
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return  # Already set by user

    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Name:") and "gfx" in line:
                gfx = line.split("gfx")[-1]
                # gfx1151 (RDNA 4) needs override to gfx1100 (RDNA 3)
                if gfx.startswith("115"):
                    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
                    return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # No rocminfo available, skip


configure_rocm()
