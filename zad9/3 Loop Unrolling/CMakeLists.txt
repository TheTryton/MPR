cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab9 LANGUAGES CXX CUDA)

add_executable(lab9_loop_unrolling_cg
	reduction.h
	reduction.cpp
	reduction_cg_kernel.cu
)
add_executable(lab9_loop_unrolling_wp
	reduction.h
	reduction.cpp
	reduction_wp_kernel.cu
)

target_include_directories(lab9_loop_unrolling_cg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lab9_loop_unrolling_wp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab9_loop_unrolling_cg PRIVATE lab9_common)
target_link_libraries(lab9_loop_unrolling_wp PRIVATE lab9_common)
set_target_properties(lab9_loop_unrolling_cg
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
set_target_properties(lab9_loop_unrolling_wp
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
