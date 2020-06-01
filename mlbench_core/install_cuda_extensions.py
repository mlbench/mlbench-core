import os

from setuptools import setup

import mlbench_core

base__dir = os.path.dirname(mlbench_core.__file__)
ext_modules = []
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    dir_1 = os.path.join(
        base__dir, "models/pytorch/transformer/modules/strided_batched_gemm"
    )

    dir_2 = os.path.join(base__dir, "models/pytorch/gnmt/attn_score")
    strided_batched_gemm = CUDAExtension(
        name="mlbench_core.models.pytorch.transformer.modules.strided_batched_gemm",
        sources=[
            os.path.join(dir_1, "strided_batched_gemm.cpp"),
            os.path.join(dir_1, "strided_batched_gemm_cuda.cu"),
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

    attn_score = CUDAExtension(
        name="mlbench_core.models.pytorch.gnmt.attn_score",
        sources=[
            os.path.join(dir_2, "attn_score_cuda.cpp"),
            os.path.join(dir_2, "attn_score_cuda_kernel.cu"),
        ],
        extra_compile_args={"cxx": ["-O2",], "nvcc": ["--gpu-architecture=sm_70",]},
    )
    ext_modules.append(strided_batched_gemm)
    ext_modules.append(attn_score)
    cmdclass = {"build_ext": BuildExtension}

    setup(ext_modules=ext_modules, cmdclass=cmdclass)

except (ImportError, OSError) as e:
    raise ValueError("Cannot install extensions because CUDA was not found")
