
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fhnBaseModel LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 50)
# set(CMAKE_GENERATOR "Visual Studio 17 2022" CACHE INTERNAL "Name of generator.")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY run)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY run)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY run)

# Ensure cross-compiler toolchain is set if needed
# set(CMAKE_C_COMPILER arm-none-eabi-gcc)


add_library(fhnModel_lib STATIC
  ../PreCompiledHeader/FHNpch.cpp
  ../PreCompiledHeader/FHNpch.h
  ../CUDAfiles/ComputeFHN.cuh
  ../Structures/StructureManagment.h
  )


set_target_properties( fhnModel_lib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

add_executable(fhnModel main.cpp CUDAfiles/ComputeFHN.cu)

target_link_libraries(fhnModel PRIVATE fhnModel_lib)

# install(TARGETS fhnModel DESTINATION run)

target_include_directories(fhnModel PRIVATE /usr/AppData/local/NVIDIA)

SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 -std=c++17")
