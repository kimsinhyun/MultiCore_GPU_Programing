#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>

#include "reduction.h"

#define SM_SIZE 1024

using namespace std;

void allocateDeviceMemory(void** M, int size)
{
    cudaError_t err = cudaMalloc(M, size);
    assert(err==cudaSuccess);
}


void deallocateDeviceMemory(void* M)
{
    cudaError_t err = cudaFree(M);
    assert(err==cudaSuccess);
}

void cudaMemcpyToDevice(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyHostToDevice);
    assert(err==cudaSuccess);
}

void cudaMemcpyToHost(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToHost);
    assert(err==cudaSuccess);
}

void reduce_ref(const int* const g_idata, int* const g_odata, const int n) {
    for (int i = 0; i < 2048; i++)
        g_odata[0] += g_idata[i];
}

__global__ 
void reduce1(const int* const g_idata, int* const g_odata)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        if ((tid % (2*s)) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__
void reduce2(const int* const d_idata, int* const d_odata){
    extern __shared__ int sdata[];
    // __shared__ int sdata[TILE_WIDTH];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_idata[i];
    __syncthreads();
    
    for (unsigned int s=1; s < blockDim.x ; s*=2){
        int index = 2 * s* tid;
        if(index < blockDim.x){
            sdata[index] += sdata[index+s];
        }
        __syncthreads();
    }
    if(tid == 0)
        d_odata[blockIdx.x] = sdata[0];
}

__global__
void reduce3(const int* const d_idata, int* const d_odata){
    extern __shared__ int sdata[];
    // __shared__ int sdata[TILE_WIDTH];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_idata[i];
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s > 0 ; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if(tid == 0)
        d_odata[blockIdx.x] = sdata[0];
}

__global__
void reduce4(const int* const d_idata, int* const d_odata){
    extern __shared__ int sdata[];
    // __shared__ int sdata[TILE_WIDTH];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    sdata[tid] = d_idata[i] + d_idata[i+blockDim.x];
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s > 0 ; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if(tid == 0)
        d_odata[blockIdx.x] = sdata[0];
}

__device__ void unroll(volatile int* sdata, int t) {
	sdata[t] += sdata[t + 32];
	sdata[t] += sdata[t + 16];
	sdata[t] += sdata[t + 8];
	sdata[t] += sdata[t + 4];
	sdata[t] += sdata[t + 2];
	sdata[t] += sdata[t + 1];
}


__global__ void reduce5(const int* const d_idata, int* const d_odata) {
	__shared__ int sdata[SM_SIZE];

	int tid = threadIdx.x;

	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[tid] = d_idata[i] + d_idata[i + blockDim.x];
	__syncthreads();
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid < 32) {
		unroll(sdata, tid);
	}

	if (tid == 0) {
		d_odata[blockIdx.x] = sdata[0];
	}
}


template <unsigned int blockSize>
__global__ void reduce6 (const int* const d_idata, int* const d_odata)
{
	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	sdata[threadIdx.x] = d_idata[i] + d_idata[i + blockDim.x];
	__syncthreads();
	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
    if (blockSize >= 512) {if (tid < 256) { sdata[tid] = (sdata[tid] + sdata[tid+256]);} __syncthreads(); } 
    if (blockSize >= 256) {if (tid < 128) { sdata[tid] = (sdata[tid] + sdata[tid+128]);} __syncthreads(); } 
    if (blockSize >= 128) {if (tid < 64) {  sdata[tid] = (sdata[tid] + sdata[tid+64]);}  __syncthreads(); } 

    if (tid < 32){
        if(blockSize >= 64) {
             sdata[tid] = (sdata[tid] + sdata[tid+32]);
        }
            __syncthreads();
        if(blockSize >= 32){
             sdata[tid] = (sdata[tid] + sdata[tid+16]);
        } 
            __syncthreads();
        if(blockSize >= 16){
             sdata[tid] = (sdata[tid] + sdata[tid+8]);
        } 
            __syncthreads();
        if (blockSize >= 8){
             sdata[tid] = (sdata[tid] + sdata[tid+4]);
        }
            __syncthreads();
        if(blockSize >= 4)  {
            sdata[tid] = (sdata[tid] + sdata[tid+2]);
        }
            __syncthreads();
        if (blockSize >= 2){
            sdata[tid] = (sdata[tid] + sdata[tid+1]);
        } 
            __syncthreads();
    }
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

void reduce6_switch(const int* const d_idata, int* const d_odata, int block_num, int blockdim){
    std::cout << "block_size: " << block_num << std::endl;
    switch(block_num){
        case 512:
            reduce6<512><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);          break;
        case 256:
            reduce6<256><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);          break;
        case 128:
            reduce6<128><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);          break;
        case 64:
            reduce6<64><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);           break;
        case 32:
            reduce6<32><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);           break;
        case 16:
            reduce6<16><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);           break;
        case 8:
            reduce6<8><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);            break;
        case 4:
            reduce6<4><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);            break;
        case 2:
            reduce6<2><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);            break;
        case 1:
            reduce6<1><<<blockdim, block_num, SM_SIZE>>>(d_idata, d_odata);            break;
    }
}

// void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, int* const d_odata, const int n) {
//     int size = n;
//     int block_size = 256;
//     dim3 block(block_size,1,1);
//     // int threads = ((size-1) / block.x)/2 +1; 
//     int threads = ((size-1) / block.x) +1; 
//     dim3 grid(threads, 1);
//     std::cout << "threads: " << threads << std::endl;
//     //=========== reduce 1~3 ===========
//     reduce1<<< grid, block, SM_SIZE >>>(d_idata, d_odata);
//     for (int i = size/block_size ; i / block_size > 1; i /= block_size){
//         reduce1<<< grid, block, SM_SIZE >>>(d_odata, d_odata);
//     }
//     reduce1<<< 1, block, SM_SIZE >>>(d_odata, d_odata);

//     //=========== reduce 4~5 ===========
//     // reduce5<<< grid, block, SM_SIZE >>>(d_idata, d_odata);
//     // for (int i = threads ; i / block_size / 2 > 1; i /= (block_size*2)){
//     //     reduce5<<< grid, block, SM_SIZE >>>(d_odata, d_odata);
//     // }
//     // reduce5<<< 1, block, SM_SIZE >>>(d_odata, d_odata);

//     //=========== reduce6 ===========
//     // reduce6_switch(d_idata, d_odata, block_size, threads, block);
//     // for (int i = threads ; i / block_size / 2 > 1; i /= (block_size*2)){
//     //     reduce6_switch(d_odata, d_odata, block_size, i, block);
//     // }
//     // reduce6_switch(d_odata, d_odata, block_size, 1, block);
// }


void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, int* const d_odata, const int n) {
    int size = n;
    int block_num = 256;
    int blockdim = ((size-1) / block_num) +1; 
    // int blockdim = ((size-1) / block_num)/2 +1; 
    // reduce5<<< blockdim, block_num, SM_SIZE >>>(d_idata, d_odata);
    // for (int i = blockdim/block_num ; i > 1 ; i /=block_num){
    //     if(i >= block_num)
    //         reduce5<<< i, block_num, SM_SIZE >>>(d_odata, d_odata);
    //     else
    //         reduce5<<< blockdim, block_num, SM_SIZE >>>(d_odata, d_odata);
    // }
    // reduce5<<< 1, block_num, SM_SIZE >>>(d_odata, d_odata);

    reduce1<<< blockdim, block_num, SM_SIZE >>>(d_idata, d_odata);
    for (int i = block_num ; i >= block_num; i /=2){
        reduce1<<< blockdim, i, SM_SIZE >>>(d_odata, d_odata);
    }
    reduce1<<< 1, block_num, SM_SIZE >>>(d_odata, d_odata);


    // reduce6_switch(d_idata, d_odata, block_num, blockdim);
    // for (int i = blockdim/block_num ; i > 1 ; i /=block_num){
    //     if(i >= block_num)
    //         reduce6_switch(d_odata, d_odata, block_num, blockdim);
    //     else
    //         reduce6_switch(d_odata, d_odata, block_num, blockdim);
    // }
    // reduce6_switch(d_odata, d_odata, block_num, blockdim);

}