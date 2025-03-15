cmake_minimum_required(VERSION 3.18)

# this has to be set before project()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(test LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
enable_language(CUDA)

add_library(cuda_comp SHARED include/cuda_kernels.h src/cuda_kernels.cu)


# this command would only set the minimum standard,
# i.e. CMake can still decide to use -std=c++17 instead
# if the given compilers support C++17
target_compile_features(cuda_comp PRIVATE cuda_std_14)

set_target_properties(cuda_comp
        PROPERTIES
                CUDA_RUNTIME_LIBRARY Shared
                # CUDA_STANDARD 14 # this one cannot be changed by CMake
                # CUDA_SEPARABLE_COMPILATION ON # not needed for this example
)

set(SOURCE_FILES
    include/sim.h
    include/io.h
    include/sampling.h
    include/definitions.h
    src/sim.cpp
    src/main.cpp
    src/io.cpp
    src/sampling.cpp
)
add_executable(main ${SOURCE_FILES})

target_compile_features(main PRIVATE cxx_std_17)
# set_target_properties(main CXX_STANDARD 17)
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /fp:fast") # Parallel compilation in Visual Studio
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set( EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/Eigen )
if( NOT EIGEN3_INCLUDE_DIR )
    message( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
endif()

target_link_libraries(main PRIVATE cuda_comp)
target_include_directories(main PUBLIC ${EIGEN3_INCLUDE_DIR} )
target_include_directories(main PUBLIC ${PROJECT_SOURCE_DIR}/extern )
target_include_directories(cuda_comp PUBLIC ${EIGEN3_INCLUDE_DIR} )

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/res DESTINATION ${CMAKE_BINARY_DIR})
