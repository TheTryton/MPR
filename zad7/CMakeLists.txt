cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab7 LANGUAGES CXX CUDA)

add_executable(lab7
	helper_cuda.h
	helper_string.h
	vectorAdd.cu
)
target_include_directories(lab7 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(lab7
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
        )