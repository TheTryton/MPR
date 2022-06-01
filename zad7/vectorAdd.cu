/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <execution>
#include <system_error>
#include <string>
#include <memory>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

struct cpu_t {};
struct gpu_t
{
    size_t block_size;
};
constexpr static auto cpu_v = cpu_t{};

void vector_add(cpu_t, const float* a, const float* a_end, const float* b, float* out)
{
    std::transform(std::execution::par_unseq, a, a_end, b, out, std::plus<>{});
}

__global__ void vector_add_impl_gpu(const float* a, const float* b, float* out, size_t length) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < length) {
        out[i] = a[i] + b[i];
    }
}

void throw_if_failed(const cudaError_t& error)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

void cuda_memory_deleter(float* ptr)
{
    throw_if_failed(cudaFree(ptr));
};
using cuda_memory_deleter_t = void(*)(float*);

std::unique_ptr<float[], cuda_memory_deleter_t> allocate_device_memory(size_t length)
{
    float* ptr = nullptr;
    throw_if_failed(cudaMalloc((void**)&ptr, length * sizeof(float)));
    return std::unique_ptr<float[], cuda_memory_deleter_t>(ptr, cuda_memory_deleter);
}

using seconds_double = std::chrono::duration<double>;

struct measurements
{
    seconds_double copy_in;
    seconds_double calculation;
    seconds_double copy_out;
};

measurements vector_add(gpu_t v, const float* a, const float* a_end, const float* b, float* out)
{
    measurements msrmnt{};

    auto length = static_cast<size_t>(a_end - a);
    auto a_device = allocate_device_memory(length);
    auto b_device = allocate_device_memory(length);
    auto out_device = allocate_device_memory(length);

    auto start = std::chrono::high_resolution_clock::now();
    throw_if_failed(cudaMemcpy(a_device.get(), a, length * sizeof(float), cudaMemcpyHostToDevice));
    throw_if_failed(cudaMemcpy(b_device.get(), b, length * sizeof(float), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    msrmnt.copy_in = std::chrono::duration_cast<seconds_double>(end - start);

    auto grid_size = (length + v.block_size - 1) / v.block_size;
    auto block_size = v.block_size;

    start = std::chrono::high_resolution_clock::now();
    vector_add_impl_gpu<<<grid_size, block_size>>>(a_device.get(), b_device.get(), out_device.get(), length);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    msrmnt.calculation = std::chrono::duration_cast<seconds_double>(end - start);

    throw_if_failed(cudaGetLastError());

    start = std::chrono::high_resolution_clock::now();
    throw_if_failed(cudaMemcpy(out, out_device.get(), length * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    msrmnt.copy_out = std::chrono::duration_cast<seconds_double>(end - start);

    return msrmnt;
}

std::vector<float> prepare_data(size_t length = 50000)
{
    auto result = std::vector<float>(length);
    std::iota(std::begin(result), std::end(result), 0.0f);
    //std::generate(std::begin(result), std::end(result), []() { return rand() / static_cast<float>(RAND_MAX); });
    return result;
}

bool compare_outputs(const float* a, const float* a_end, const float* b)
{
    return std::equal(a, a_end, b, [](auto&& a, auto&& b) { return std::abs(a - b) <= std::numeric_limits<float>::epsilon(); });
}

int main(int argc, const char* argv[])
{
    try
    {
        size_t length = 100000000;
        size_t block_size = 256;

        if (argc >= 2)
        {
            length = atoi(argv[1]);
        }

        if (argc >= 3)
        {
            block_size = atoi(argv[2]);
        }

        auto data_a = prepare_data(length);
        auto data_b = prepare_data(length);
        auto outputHost = std::vector<float>(data_a.size());
        auto outputDevice = std::vector<float>(data_a.size());

        auto msrmnt = vector_add(gpu_t{ block_size }, data_a.data(), data_a.data() + data_a.size(), data_b.data(), outputDevice.data());

        auto start_host = std::chrono::high_resolution_clock::now();
        vector_add(cpu_v, data_a.data(), data_a.data() + data_a.size(), data_b.data(), outputHost.data());
        auto end_host = std::chrono::high_resolution_clock::now();

        std::cout << "Vector Size=" << data_a.size() << std::endl;
        std::cout << "Device" << std::endl;
        std::cout << "CopyIn: " << msrmnt.copy_in.count() << 's' << std::endl;
        std::cout << "Calculation: " << msrmnt.calculation.count() << 's' << std::endl;
        std::cout << "CopyOut: " << msrmnt.copy_out.count() << 's' << std::endl;
        std::cout << "Host Time Taken: " << std::chrono::duration_cast<seconds_double>(end_host - start_host).count() << 's' << std::endl;
        std::cout << "Are Equal?: " << (compare_outputs(outputHost.data(), outputHost.data() + outputHost.size(), outputDevice.data()) ? "True" : "False") << std::endl;
    }
    catch (const std::runtime_error& error)
    {
        std::cout << "Error: " << error.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "Unknown error!" << std::endl;
    }

    return 0;
}
