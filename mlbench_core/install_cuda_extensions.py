import os

from setuptools import setup

import mlbench_core

base__dir = os.path.dirname(mlbench_core.__file__)
ext_modules = []
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    dir_ = os.path.join(
        base__dir, "models/pytorch/transformer/modules/strided_batched_gemm"
    )
    strided_batched_gemm = CUDAExtension(
        name="mlbench_core.models.pytorch.transformer.modules.strided_batched_gemm",
        sources=[
            os.path.join(dir_, "strided_batched_gemm.cpp"),
            os.path.join(dir_, "strided_batched_gemm_cuda.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O2",],
            "nvcc": [
                "--gpu-architecture=compute_70",
                "--gpu-code=sm_70",
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
            ],
        },
    )
    ext_modules.append(strided_batched_gemm)
    cmdclass = {"build_ext": BuildExtension}

    setup(ext_modules=ext_modules, cmdclass=cmdclass)

except (ImportError, OSError) as e:
    raise ValueError("Cannot install extensions because CUDA was not found")
