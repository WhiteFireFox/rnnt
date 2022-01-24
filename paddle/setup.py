from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='rnnt_ops',
    ext_modules=CUDAExtension(
        sources=['binding.cc', 'core.cu', 'core_gather.cu']
    )
)
