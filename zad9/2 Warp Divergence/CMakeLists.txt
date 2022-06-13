cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab9 LANGUAGES CXX CUDA)

add_executable(lab9_warp_divergence_sequential
	reduction.h
	reduction.cpp
	reduction_kernel_sequential.cu
)

add_executable(lab9_warp_divergence_interleaving
	reduction.h
	reduction.cpp
	reduction_kernel_interleaving.cu
)

target_include_directories(lab9_warp_divergence_sequential PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lab9_warp_divergence_interleaving PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab9_warp_divergence_sequential PRIVATE lab9_common)
target_link_libraries(lab9_warp_divergence_interleaving PRIVATE lab9_common)
set_target_properties(lab9_warp_divergence_sequential
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
set_target_properties(lab9_warp_divergence_interleaving
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
