cmake_minimum_required(VERSION 3.20)

project(lab6)
project(lab6mapper)
project(lab6reducer)

add_executable(lab6
        main.cpp
        )
add_executable(lab6mapper
        mapper.cpp
        )
add_executable(lab6reducer
        reducer.cpp
        )

set_target_properties(lab6
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )
set_target_properties(lab6mapper
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )
set_target_properties(lab6reducer
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )