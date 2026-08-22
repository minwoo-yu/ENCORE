from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="flying_conv",
    version="0.1.0",
    packages=["flying_conv"],
    package_dir={"flying_conv": "."},
    ext_modules=[
        CUDAExtension(
            "flying_conv_backend",
            [
                "flying_conv.cpp",
                "FlyingConv2D.cu",
                "cuda_utils.cu",
            ],
            extra_compile_args={
                'cxx': ["-O3"],
                'nvcc': ["-O3", "--use_fast_math"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

# pip install . --no-build-isolation -> for installation
