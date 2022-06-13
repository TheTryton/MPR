cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(lab9 LANGUAGES CXX CUDA)

add_subdirectory("common")
add_subdirectory("1 Reduction")
add_subdirectory("2 Warp Divergence")
add_subdirectory("3 Loop Unrolling")
add_subdirectory("4 Atomic Operations")
add_subdirectory("5 Histogram")
