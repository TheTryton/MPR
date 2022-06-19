cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab9 LANGUAGES CXX CUDA)

add_executable(lab9_histogram
	gputimer.h
	histo.cu
)
add_executable(lab9_reduce
	gputimer.h
	reduce.cu
)

target_include_directories(lab9_histogram PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(lab9_reduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lab9_histogram PRIVATE lab9_common)
target_link_libraries(lab9_reduce PRIVATE lab9_common)
set_target_properties(lab9_histogram
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
set_target_properties(lab9_reduce
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
		CUDA_SEPARABLE_COMPILATION ON
		VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
