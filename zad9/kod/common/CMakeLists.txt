cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab9 LANGUAGES CXX CUDA)

set(common_headers
	helper_cuda.hpp
	helper_string.hpp
	helper_timer.h
	exception.h
	common.hpp
)

set(common_sources
	common.cpp
)

add_library(lab9_common STATIC ${common_headers} ${common_sources})
target_include_directories(lab9_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(lab9_common
        PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )
