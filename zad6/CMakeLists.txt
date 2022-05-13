cmake_minimum_required(VERSION 3.20)

project(lab6)

add_executable(lab6
        main.cpp
        )

set_target_properties(lab6
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )