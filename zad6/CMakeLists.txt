cmake_minimum_required(VERSION 3.20)

project(lab6)

find_package(OpenMP)

add_executable(lab6
        main.cpp
        )
target_link_libraries(lab6 PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(lab6 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(lab6
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )