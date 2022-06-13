cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab9 LANGUAGES CXX CUDA)

add_executable(lab9_atomic_operations_base
	reduction.h
	reduction.cpp
	reduction_kernel.cu
)
add_executable(lab9_atomic_operations_blk_atmc
	reduction.h
	reduction.cpp
	reduction_blk_atmc_kernel.cu
)
add_executable(lab9_atomic_operations_wrp_atmc
	reduction.h
	reduction.cpp
	reduction_wrp_atmc_kernel.cu
)

target_include_directories(lab9_atomic_operations_base PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lab9_atomic_operations_blk_atmc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lab9_atomic_operations_wrp_atmc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab9_atomic_operations_base PRIVATE lab9_common)
target_link_libraries(lab9_atomic_operations_blk_atmc PRIVATE lab9_common)
target_link_libraries(lab9_atomic_operations_wrp_atmc PRIVATE lab9_common)
set_target_properties(lab9_atomic_operations_base
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
set_target_properties(lab9_atomic_operations_blk_atmc
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
set_target_properties(lab9_atomic_operations_wrp_atmc
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
