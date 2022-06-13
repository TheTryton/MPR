#pragma once

#include <cuda_runtime.h>

#include <system_error>
#include <memory>
#include <chrono>
#include <type_traits>
#include <string>
#include <string_view>
#include <array>
#include <ostream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <execution>
#include <iomanip>

#include <cstdlib>

struct cpu_t {};
struct gpu_t { size_t block_size; };

void throw_if_failed(const cudaError_t& error);

template<typename T>
void device_memory_deleter(T* ptr)
{
    throw_if_failed(cudaFree(ptr));
};
void device_array_deleter(cudaArray* array);
template<typename T>
void host_memory_deleter(T* ptr)
{
    free(ptr);
};
template<typename T>
using device_memory_deleter_t = void(*)(T*);
using device_array_deleter_t = void(*)(cudaArray*);
template<typename T>
using host_memory_deleter_t = void(*)(T*);

template<typename T>
std::unique_ptr<T[], device_memory_deleter_t<T>> allocate_device_memory(size_t count)
{
    static_assert(std::is_trivial_v<T>, "Type T must be trivial type!");
    T* ptr = nullptr;
    throw_if_failed(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return std::unique_ptr<T[], device_memory_deleter_t<T>>(ptr, device_memory_deleter<T>);
}
std::unique_ptr<cudaArray, device_array_deleter_t> allocate_device_array(size_t width, size_t height, cudaChannelFormatDesc channel_desc);
template<typename T>
std::unique_ptr<T[], host_memory_deleter_t<T>> allocate_host_memory(size_t count)
{
    static_assert(std::is_trivial_v<T>, "Type T must be trivial type!");
    T* ptr = reinterpret_cast<T*>(malloc(count * sizeof(T)));
    if(ptr == nullptr)
        throw std::bad_alloc();
    return std::unique_ptr<T[], host_memory_deleter_t<T>>(ptr, host_memory_deleter<T>);
}

using seconds_double = std::chrono::duration<double, std::ratio<1,1>>;
using millis_double = std::chrono::duration<double, std::milli>;
using micros_double = std::chrono::duration<double, std::micro>;
using nanos_double = std::chrono::duration<double, std::nano>;

struct measurements
{
    seconds_double copy_in;
    seconds_double calculation;
    seconds_double copy_out;
};

struct name_t { char name[256]; };
struct uuid_t { char bytes[16]; };
struct luid_t { char bytes[8]; };
struct properties_t
{
    name_t name;
    uuid_t uuid;
    luid_t luid;

    size_t total_global_memory;
    size_t shared_memory_per_block;
    size_t shared_memory_per_multiprocessor;
    int32_t registers_per_block;
    int32_t registers_per_multiprocessor;
    int32_t max_blocks_per_multiprocessor;
    size_t memory_pitch;
    int32_t max_threads_per_block;
    std::array<int32_t, 3> max_threads_dimensions;
    std::array<int32_t, 3> max_grid_size;
    int32_t clock_frequency_khz;
    int32_t memory_frequency_khz;
    int32_t memory_bus_width;

    size_t total_constant_memory;
    int32_t major_compute_capabilities;
    int32_t minor_compute_capabilities;
    int32_t number_of_multiprocessors;
    std::array<int32_t, 1> max_texture_1d_size;
    std::array<int32_t, 2> max_texture_2d_size;
    std::array<int32_t, 3> max_texture_3d_size;
};

int32_t get_devices_count();
int32_t get_current_device();
properties_t get_device_properties(int32_t device);

std::ostream& operator<<(std::ostream& o, const name_t& v);
std::ostream& operator<<(std::ostream& o, const uuid_t& v);
std::ostream& operator<<(std::ostream& o, const luid_t& v);

void print_properties(const properties_t& p);
