cmake_minimum_required(VERSION 3.10)
project(cuda_mace LANGUAGES CXX CUDA)

set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install DLL/binaries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")

SET(CUDA_SEPARABLE_COMPILATION ON)

find_package(Python COMPONENTS Interpreter REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE)
else()
    message(FATAL_ERROR "Could not find CUDA Compiler. Make sure $CUDA_HOME is set.")
endif()

execute_process(
    COMMAND which ${Python_EXECUTABLE}
)

execute_process(
    COMMAND ${Python_EXECUTABLE} --version
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
    ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
)

if (NOT ${TORCH_CMAKE_PATH_RESULT} EQUAL 0)
message(FATAL_ERROR "failed to find your pytorch installation, error: ${TORCH_CMAKE_PATH_ERROR}\n")
endif()

string(STRIP ${TORCH_CMAKE_PATH_OUTPUT} TORCH_CMAKE_PATH_OUTPUT)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PATH_OUTPUT}")

execute_process(
    COMMAND sh -c "grep -q 'set(CUDA_ALL_GPU_ARCHITECTURES \"3.5\" \"5.0\")' ${TORCH_CMAKE_PATH_OUTPUT}/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake"
    RESULT_VARIABLE grep_result
    OUTPUT_VARIABLE grep_output
    ERROR_VARIABLE grep_error
)
# set(CUDA_ALL_GPU_ARCHITECTURES "3.5" "5.0")
#TORCH_CMAKE_PATH_OUTPUT: /opt/python/cp311-cp311/lib/python3.11/site-packages/torch/share/cmake
if (grep_result EQUAL 0)
    execute_process(
        COMMAND sh -c "sed -i -E 's/set\\(CUDA_COMMON_GPU_ARCHITECTURES \"3\\.5\" \"5\\.0\"\\)/set(CUDA_COMMON_GPU_ARCHITECTURES \"\")/' ${TORCH_CMAKE_PATH_OUTPUT}/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )

    if (NOT ${result} EQUAL 0)
        message(FATAL_ERROR "Error running sed: ${error}")
    else()
        message(STATUS "sed command ran successfully.")
    endif()
    
    execute_process(
        COMMAND sh -c "sed -i -E 's/set\\(CUDA_ALL_GPU_ARCHITECTURES \"3\\.5\" \"5\\.0\"\\)/set(CUDA_ALL_GPU_ARCHITECTURES \"\")/' ${TORCH_CMAKE_PATH_OUTPUT}/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )

    if (NOT ${result} EQUAL 0)
        message(FATAL_ERROR "Error running sed: ${error}")
    else()
        message(STATUS "sed command ran successfully.")
    endif()

endif()

find_package(Torch 1.13 REQUIRED)

#enable relocatable device code (e.g for cuda_utils.cu)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -lineinfo")

include_directories(
    ${CMAKE_SOURCE_DIR}/cuda/include
    ${CMAKE_SOURCE_DIR}/include
)

add_library(cuda_mace SHARED
    "cuda/include/cuda_utils.cuh"
    "cuda/src/cuda_utils.cu"

    "cuda/include/torch_utils.cuh"
    "cuda/src/torch_utils.cu"

    "cuda/include/invariant_message_passing_impl.cuh"
    "cuda/src/invariant_message_passing_impl.cu"

    "cuda/include/linear_wmma_impl.cuh"
    "cuda/src/linear_wmma_impl.cu"

    "cuda/include/symmetric_contraction_impl.cuh"
    "cuda/src/symmetric_contraction_impl.cu"

    "cuda/include/cubic_spline_impl.cuh"
    "cuda/src/cubic_spline_impl.cu"

    "cuda/include/spherical_harmonics_impl.cuh"
    "cuda/src/spherical_harmonics_impl.cu"

    "src/linear_wmma.cpp"
    "src/invariant_message_passing.cpp"
    "src/symmetric_contraction.cpp"
    "src/cubic_spline.cpp"
    "src/spherical_harmonics.cpp"
)

target_link_libraries(cuda_mace "${TORCH_LIBRARIES}")

target_compile_features(cuda_mace PRIVATE cxx_std_17)

set_target_properties(cuda_mace PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuda_mace PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(cuda_mace PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")

set(CMAKE_CUDA_STANDARD 17)

install(TARGETS cuda_mace
    LIBRARY DESTINATION "lib"
)