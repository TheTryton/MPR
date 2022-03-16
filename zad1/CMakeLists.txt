project(lab1_throughput C)
project(lab1_latency C)

find_package(MPI) 

add_executable(lab1_throughput main_throughput.c)
target_include_directories(lab1_throughput PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(lab1_throughput PRIVATE ${MPI_C_LIBRARIES})

add_executable(lab1_latency main_latency.c)
target_include_directories(lab1_latency PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(lab1_latency PRIVATE ${MPI_C_LIBRARIES})