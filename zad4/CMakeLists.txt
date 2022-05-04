cmake_minimum_required(VERSION 3.20)

project(lab4)

find_package(OpenMP) 


add_executable(lab4
        parallel_primitives.hpp
        parallel_primitives.cpp

        parallel_for.hpp
        parallel_for.cpp

        parallel_max.hpp
       
        value_range.hpp
        fixed_size_dynamic_array.hpp

        bucket.hpp
        lockables.hpp
        buckets.hpp

        bucket_sort.hpp

        time_it.hpp

        test_parameters.hpp

        main.cpp
        )
target_link_libraries(lab4 PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(lab4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if (CMAKE_GENERATOR MATCHES "Visual Studio")
    target_compile_options(lab4 PRIVATE "-openmp:llvm")
endif()


set_target_properties(lab4
        PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )