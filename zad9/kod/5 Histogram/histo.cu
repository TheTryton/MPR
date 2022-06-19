#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

int log2(int i)
{
    int r = 0;
    while (i >>= 1) r++;
    return r;
}

int bit_reverse(int w, int bits)
{
    int r = 0;
    for (int i = 0; i < bits; i++)
    {
        int bit = (w & (1 << i)) >> i;
        r |= bit << (bits - i - 1);
    }
    return r;
}

__global__ void naive_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = d_in[myId];
    int myBin = myItem % BIN_COUNT;
    d_bins[myBin]++;
}

__global__ void simple_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = d_in[myId];
    int myBin = myItem % BIN_COUNT;
    atomicAdd(&(d_bins[myBin]), 1);
}

int measure_histo(int histo_type, int size, int bin_count)
{
    const int ARRAY_SIZE = size;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    const int BIN_COUNT = bin_count;
    const int BIN_BYTES = BIN_COUNT * sizeof(int);

    // generate the input array on the host
    int* h_in = new int[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = bit_reverse(i, log2(ARRAY_SIZE));
    }
    int* h_bins = new int[BIN_COUNT];
    for (int i = 0; i < BIN_COUNT; i++) {
        h_bins[i] = 0;
    }

    // declare GPU memory pointers
    int* d_in;
    int* d_bins;

    // allocate GPU memory
    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_bins, BIN_BYTES);

    auto start = std::chrono::high_resolution_clock::now();

    // transfer the arrays to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice);

    int whichKernel = histo_type;

    // launch the kernel
    switch (whichKernel) {
    case 0:
        naive_histo << <ARRAY_SIZE / 64, 64 >> > (d_bins, d_in, BIN_COUNT);
        break;
    case 1:
        simple_histo << <ARRAY_SIZE / 64, 64 >> > (d_bins, d_in, BIN_COUNT);
        break;
    default:
        fprintf(stderr, "error: ran no kernel\n");
        exit(EXIT_FAILURE);
    }

    // copy back the sum from GPU
    cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    printf("%f", (end - start) / 1e6);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_bins);
    delete[] h_in;
    delete[] h_bins;
}


int main(int argc, char **argv)
{

    printf("size, bin_count, naive [ms], simple [ms]\n");
   
    for (int size = 1 << 16; size < (1 << 28); size *= 2)
    {
        for (int bin_count : {10, 50, 100, 500, 1000})
        {
            printf("%i,", size);
            printf("%i,", bin_count);
            measure_histo(0, size, bin_count);
            printf(",");
            measure_histo(1, size, bin_count);
            printf("\n");
        }
    }
        
    return 0;
}
