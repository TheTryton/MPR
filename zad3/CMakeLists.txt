cmake_minimum_required(VERSION 3.16)

project(lab3)

find_package(OpenMP) 

add_executable(lab3 main.cpp)
target_link_libraries(lab3 PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(lab3
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )