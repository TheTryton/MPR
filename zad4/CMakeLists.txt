cmake_minimum_required(VERSION 3.20)

project(lab3)

find_package(OpenMP) 

add_executable(lab3
        parallel_for.hpp
        parallel_for.cpp

        schedule.hpp
        schedule.cpp

        range.hpp
        fixed_size_dynamic_array.hpp

        bucket.hpp
        lockables.hpp
        buckets.hpp

        bucket_sort.hpp

        time_it.hpp

        test_parameters.hpp

        main.cpp
        )
target_link_libraries(lab3 PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(lab3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(lab3
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )