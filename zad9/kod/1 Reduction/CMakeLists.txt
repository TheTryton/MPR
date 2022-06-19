cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab9 LANGUAGES CXX CUDA)

add_executable(lab9_reduction_global
	reduction.h
	reduction_global.cpp
	reduction_global_kernel.cu
)

add_executable(lab9_reduction_shared
	reduction.h
	reduction_shared.cpp
	reduction_shared_kernel.cu
)

target_include_directories(lab9_reduction_global PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lab9_reduction_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab9_reduction_global PRIVATE lab9_common)
target_link_libraries(lab9_reduction_shared PRIVATE lab9_common)
set_target_properties(lab9_reduction_global
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
set_target_properties(lab9_reduction_shared
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
