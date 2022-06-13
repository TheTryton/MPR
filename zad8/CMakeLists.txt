cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab8 LANGUAGES CXX CUDA)

set(common_headers
	helper_cuda.hpp
	helper_string.hpp
	common.hpp
)

set(common_sources
	common.cpp
)

add_executable(lab8_transpose_1
	${common_headers}
	${common_sources}
	matrix_transpose_1.cu
)
add_executable(lab8_image_scaling_1
	${common_headers}
	${common_sources}
	scrImagePgmPpmPackage.hpp
	scrImagePgmPpmPackage.cpp
	image_scaling_1.cu
)

target_include_directories(lab8_transpose_1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lab8_image_scaling_1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(lab8_transpose_1
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
set_target_properties(lab8_image_scaling_1
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
