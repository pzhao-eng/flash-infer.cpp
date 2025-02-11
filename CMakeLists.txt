# int8 conv unit test
cmake_minimum_required(VERSION 3.17)  # for add_link_options
project (my_cublas VERSION 0.0.1 LANGUAGES C CXX)
set (APP_NAME main)
set(TARGET ${APP_NAME})

## for cross compile
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(TOOL_CHAINS /mnt/data/zhaop/repo/toolchains_6060/)
set(CMAKE_SYSROOT ${TOOL_CHAINS}/aarch64--glibc--stable-2022.03-1/aarch64-buildroot-linux-gnu/sysroot)
set(CUDAToolkit_TARGET_NAMES aarch64-linux)

# for c and cxx compiler
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA 11.4.20
set(CUDAToolkit_ROOT ${TOOL_CHAINS}/cuda-11.4.20)
set(CUDAToolkit_BIN_DIR ${CUDAToolkit_ROOT}/bin/)
set(CUDAToolkit_ROOT_DIR ${CUDAToolkit_ROOT})
set(CMAKE_CUDA_COMPILER_WORKS TRUE)
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc -m64 -ccbin ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -arch=compute_86;-code=sm_86)
    enable_language(CUDA)
    set(CUDAToolkit_LIBRARY_DIR ${TOOL_CHAINS}/cuda-11.4.20/targets/aarch64-linux/lib/)
    set(CUDA_DRIVER_LIBRARY ${TOOL_CHAINS}/drive-linux-6.0.6.0/drive-linux/filesystem/targetfs/usr/lib/libcuda.so)
    set(CUDART_LIBRARY ${TOOL_CHAINS}/cuda-11.4.20/targets/aarch64-linux/lib/libcudart.so)
    set(NVRTC_LIBRARY ${TOOL_CHAINS}/cuda-11.4.20/targets/aarch64-linux/lib/stubs/libnvrtc.so)
    # message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    # message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
    set(EXTRA_LIBS ${EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
    link_directories(${CUDAToolkit_LIBRARY_DIR})
else()
    message(FATAL_ERROR "CUDA not found")
endif()

add_executable(${TARGET} ./example/cpp/${APP_NAME}.cu ./utils/cuda_utils.cu)

target_include_directories(${TARGET} PUBLIC
                           ${PROJECT_SOURCE_DIR}/
                           /mnt/data/nfsroot/zhaop/tools/cutlass/tools/util/include/
                           /mnt/data/nfsroot/zhaop/tools/cutlass/include/
                           /mnt/data/nfsroot/zhaop/tools/cutlass/examples/common/
                           ${CUDAToolkit_INCLUDE_DIRS}
                           )

target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           # --generate-line-info
                           --use_fast_math
                           --ptxas-options=-v
                           --expt-relaxed-constexpr
                           --expt-extended-lambda
                           -DENABLE_BF16
                           --disable-warnings
                           >)
