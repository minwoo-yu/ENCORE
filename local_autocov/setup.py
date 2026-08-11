from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

# Compile the CUDA extension
setup(
    name="local_autocov",
    version="0.1.0",
    packages=["local_autocov"],
    package_dir={"local_autocov": "."},
    ext_modules=[
        CUDAExtension(
            "local_autocov_backend",
            [
                "local_autocov.cpp",
                "LocalAutocov2D.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
