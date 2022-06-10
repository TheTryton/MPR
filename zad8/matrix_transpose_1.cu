#include<stdio.h>
#include<stdlib.h>

#include <common.hpp>

template<typename T>
__global__ void matrix_transpose_naive_dc_store(T* input, T* output, size_t n, size_t m) {

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	if (indexX >= m || indexY >= n)
		return;

	int index = indexY * n + indexX;
	int transposedIndex = indexX * n + indexY;
 
	output[transposedIndex] = input[index];
}

template<typename T>
__global__ void matrix_transpose_naive_dc_load(T* input, T* output, size_t n, size_t m) {

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	if (indexX >= m || indexY >= n)
		return;

	int index = indexY * n + indexX;
	int transposedIndex = indexX * n + indexY;

	output[index] = input[transposedIndex];
}

template<typename T>
__global__ void matrix_transpose_shared(T* input, T* output, size_t n, size_t m, size_t block_size)
{
	extern __shared__ int sharedMemory[];
	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
	int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;

	int index = indexY * n + indexX;
	int transposedIndex = tindexY * n + tindexX;

	sharedMemory[localIndexX * block_size + localIndexY] = input[index];

	__syncthreads();

	output[transposedIndex] = sharedMemory[localIndexY * block_size + localIndexX];
}

struct naive_dc_store_t {};
struct naive_dc_load_t {};
struct shared_t {};
struct shared_p1_t {};

template<typename T>
void transpose(cpu_t, T* a, T* b, size_t n, size_t m)
{
	std::for_each(std::execution::par_unseq, a, a + n * m, [a, b, n, m](const auto& v)
		{
			auto index = &v - a;
			auto indexX = index % n;
			auto indexY = index / n;
			auto transposedIndex = indexX * n + indexY;
			b[transposedIndex] = v;
		});
}

template<typename T>
void transpose(gpu_t v, naive_dc_store_t, T* a, T* b, T* ad, T* bd, size_t n, size_t m)
{
	size_t matrix_data_size = n * m;

	throw_if_failed(cudaMemcpy(ad, a, matrix_data_size * sizeof(float), cudaMemcpyHostToDevice));
	dim3 grid_size =
	{
		static_cast<unsigned int>((m + v.block_size - 1) / v.block_size),
		static_cast<unsigned int>((n + v.block_size - 1) / v.block_size),
		1u,
	};
	dim3 block_size =
	{
		static_cast<unsigned int>(v.block_size),
		static_cast<unsigned int>(v.block_size),
		1u
	};
	matrix_transpose_naive_dc_store<T><<<grid_size, block_size>>>(ad, bd, n, m);
	throw_if_failed(cudaGetLastError());
	throw_if_failed(cudaDeviceSynchronize());
	throw_if_failed(cudaMemcpy(b, bd, matrix_data_size * sizeof(float), cudaMemcpyDeviceToHost));
}

template<typename T>
void transpose(gpu_t v, naive_dc_load_t, T* a, T* b, T* ad, T* bd, size_t n, size_t m)
{
	size_t matrix_data_size = n * m;

	throw_if_failed(cudaMemcpy(ad, a, matrix_data_size * sizeof(T), cudaMemcpyHostToDevice));
	dim3 grid_size =
	{
		static_cast<unsigned int>((m + v.block_size - 1) / v.block_size),
		static_cast<unsigned int>((n + v.block_size - 1) / v.block_size),
		1u,
	};
	dim3 block_size =
	{
		static_cast<unsigned int>(v.block_size),
		static_cast<unsigned int>(v.block_size),
		1u
	};
	matrix_transpose_naive_dc_load<T><<<grid_size, block_size>>>(ad, bd, n, m);
	throw_if_failed(cudaGetLastError());
	throw_if_failed(cudaDeviceSynchronize());
	throw_if_failed(cudaMemcpy(b, bd, matrix_data_size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void transpose(gpu_t v, shared_t, T* a, T* b, T* ad, T* bd, size_t n, size_t m)
{
	size_t matrix_data_size = n * m;

	throw_if_failed(cudaMemcpy(ad, a, matrix_data_size * sizeof(T), cudaMemcpyHostToDevice));
	dim3 grid_size =
	{
		static_cast<unsigned int>((m + v.block_size - 1) / v.block_size),
		static_cast<unsigned int>((n + v.block_size - 1) / v.block_size),
		1u,
	};
	dim3 block_size =
	{
		static_cast<unsigned int>(v.block_size),
		static_cast<unsigned int>(v.block_size),
		1u
	};
	matrix_transpose_shared<T><<<grid_size, block_size, v.block_size*v.block_size*sizeof(T)>>>(ad, bd, n, m, v.block_size);
	throw_if_failed(cudaDeviceSynchronize());
	throw_if_failed(cudaMemcpy(b, bd, matrix_data_size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void transpose(gpu_t v, shared_p1_t, T* a, T* b, T* ad, T* bd, size_t n, size_t m)
{
	size_t matrix_data_size = n * m;

	auto a_d = allocate_device_memory<T>(matrix_data_size);
	auto b_d = allocate_device_memory<T>(matrix_data_size);

	throw_if_failed(cudaMemcpy(ad, a, matrix_data_size * sizeof(T), cudaMemcpyHostToDevice));
	dim3 grid_size =
	{
		static_cast<unsigned int>((m + v.block_size - 1) / v.block_size),
		static_cast<unsigned int>((n + v.block_size - 1) / v.block_size),
		1u,
	};
	dim3 block_size =
	{
		static_cast<unsigned int>(v.block_size),
		static_cast<unsigned int>(v.block_size),
		1u
	};
	matrix_transpose_shared<T><<<grid_size, block_size, v.block_size * (v.block_size + 1) * sizeof(T)>>>(ad, bd, n, m, v.block_size);
	throw_if_failed(cudaDeviceSynchronize());
	throw_if_failed(cudaMemcpy(b, bd, matrix_data_size * sizeof(T), cudaMemcpyDeviceToHost));
}

void print_header()
{
	using std::cout;
	using std::endl;
	using std::setw;
	using std::left;

	cout <<
		setw(6) << left << "N" << ", " <<
		setw(6) << left << "M" << ", " <<
		setw(12) << left << "Block Size" << ", " <<
		setw(12) << left << "Thread Count" << ", " <<
		setw(12) << left << "CPU time [s]" << ", " <<
		setw(18) << left << "GPU ndcs time [s]" << ", " <<
		setw(18) << left << "GPU ndcs == CPU" << ", " <<
		setw(18) << left << "GPU ndcl time [s]" << ", " <<
		setw(18) << left << "GPU ndcl == CPU" << ", " <<
		setw(18) << left << "GPU shm time [s]" << ", " <<
		setw(18) << left << "GPU shm == CPU" << ", " <<
		setw(18) << left << "GPU shm+1 time [s]" << ", " <<
		setw(18) << left << "GPU shm+1 == CPU" <<
		endl;
}

void run(size_t n, size_t m, size_t block_size)
{
	using std::cout;
	using std::endl;
	using std::setw;
	using std::left;
	using std::chrono::duration_cast;

	size_t matrix_data_size = n * m;
	size_t thread_count = n / block_size;

	auto a_h = allocate_host_memory<int>(matrix_data_size);
	auto b_h = allocate_host_memory<int>(matrix_data_size);
	auto c_h = allocate_host_memory<int>(matrix_data_size);
	
	auto a_d = allocate_device_memory<int>(matrix_data_size);
	auto b_d = allocate_device_memory<int>(matrix_data_size);

	std::iota(a_h.get(), a_h.get() + matrix_data_size, 0);

	auto start_cpu = std::chrono::high_resolution_clock::now();
	transpose(cpu_t{}, a_h.get(), c_h.get(), n, m);
	auto end_cpu = std::chrono::high_resolution_clock::now();

	auto start_gpu_dcs = std::chrono::high_resolution_clock::now();
	transpose(gpu_t{ block_size }, naive_dc_store_t{}, a_h.get(), b_h.get(), a_d.get(), b_d.get(), n, m);
	auto end_gpu_dcs = std::chrono::high_resolution_clock::now();
	bool dcs_equal = std::equal(b_h.get(), b_h.get() + matrix_data_size, c_h.get(), c_h.get() + matrix_data_size);

	auto start_gpu_dcl = std::chrono::high_resolution_clock::now();
	transpose(gpu_t{ block_size }, naive_dc_load_t{}, a_h.get(), b_h.get(), a_d.get(), b_d.get(), n, m);
	auto end_gpu_dcl = std::chrono::high_resolution_clock::now();
	bool dcl_equal = std::equal(b_h.get(), b_h.get() + matrix_data_size, c_h.get(), c_h.get() + matrix_data_size);

	auto start_gpu_sh = std::chrono::high_resolution_clock::now();
	transpose(gpu_t{ block_size }, shared_t{}, a_h.get(), b_h.get(), a_d.get(), b_d.get(), n, m);
	auto end_gpu_sh = std::chrono::high_resolution_clock::now();
	bool shared_equal = std::equal(b_h.get(), b_h.get() + matrix_data_size, c_h.get(), c_h.get() + matrix_data_size);

	auto start_gpu_sh_p1 = std::chrono::high_resolution_clock::now();
	transpose(gpu_t{ block_size }, shared_p1_t{}, a_h.get(), b_h.get(), a_d.get(), b_d.get(), n, m);
	auto end_gpu_sh_p1 = std::chrono::high_resolution_clock::now();
	bool shared_p1_equal = std::equal(b_h.get(), b_h.get() + matrix_data_size, c_h.get(), c_h.get() + matrix_data_size);

	cout <<
		setw(6) << left << n << ", " <<
		setw(6) << left << m << ", " <<
		setw(12) << left << block_size << ", " <<
		setw(12) << left << thread_count << ", " <<
		setw(12) << left << duration_cast<seconds_double>(end_cpu - start_cpu).count() << ", " <<
		setw(18) << left << duration_cast<seconds_double>(end_gpu_dcs - start_gpu_dcs).count() << ", " <<
		setw(18) << left << dcs_equal << ", " <<
		setw(18) << left << duration_cast<seconds_double>(end_gpu_dcl - start_gpu_dcl).count() << ", " <<
		setw(18) << left << dcl_equal << ", " <<
		setw(18) << left << duration_cast<seconds_double>(end_gpu_sh - start_gpu_sh).count() << ", " <<
		setw(18) << left << shared_equal << ", " <<
		setw(18) << left << duration_cast<seconds_double>(end_gpu_sh_p1 - start_gpu_sh_p1).count() << ", " <<
		setw(18) << left << shared_p1_equal <<
		endl;
}


int main(int argc, char* argv[])
{
	using std::cout;
	using std::endl;

	try
	{
		auto device_count = get_devices_count();
		for (int32_t i = 0; i < device_count; i++)
		{
			auto p = get_device_properties(i);

			cout << "Device with ID=" << i << endl;
			print_properties(p);
		}

		auto current_device_i = get_current_device();
		auto current_device_p = get_device_properties(current_device_i);

		cout << endl;
		cout << "Currently selected device ID=" << current_device_i << endl;
		cout << endl;

		print_header();

		for (size_t n = 2; n < current_device_p.max_grid_size[1]; n *= 2)
		{
			for (size_t thread_count = 1; thread_count < current_device_p.max_threads_per_block; thread_count *= 2)
			{
				auto block_size = n / thread_count;
				if (block_size == 0 || block_size >= 64)
					continue;
				run(n, n, block_size);
			}
		}
	}
	catch (const std::runtime_error& error)
	{
		cout << "Error: " << error.what() << endl;
	}
	catch (...)
	{
		cout << "Unknown error!" << endl;
	}

	return 0;
}
