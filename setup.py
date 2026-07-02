from __future__ import annotations

import os

from setuptools import setup


def build_extensions():
    if os.environ.get("GPU_L2_BUILD_EXT", "0") != "1":
        return []
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception as exc:
        raise RuntimeError("Building _l2p_ext requires torch with CUDA extension support") from exc
    return [
        CUDAExtension(
            "gpu_l2_harness._l2p_ext",
            ["csrc/l2_persistent_ext.cpp", "csrc/l2_persistent_kernel.cu"],
        )
    ]


ext_modules = build_extensions()
cmdclass = {}
if ext_modules:
    from torch.utils.cpp_extension import BuildExtension

    cmdclass["build_ext"] = BuildExtension


setup(ext_modules=ext_modules, cmdclass=cmdclass)

