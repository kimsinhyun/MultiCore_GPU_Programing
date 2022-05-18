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

void test_generate_list_of_randNum(int size, int * arr);
void test_print_list_of_randNum(int size, int * arr);


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

	sdata[tid] = d_idata[i] + d_idata[i + blockDim.x];
	__syncthreads();
	for (int s = blockDim.x / 2; s > blockSize; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
    if (blockSize >= 512) {if (tid < 256) { sdata[tid] = (sdata[tid] + sdata[tid+256]);} __syncthreads(); } 
    if (blockSize >= 256) {if (tid < 128) { sdata[tid] = (sdata[tid] + sdata[tid+128]);} __syncthreads(); } 
    if (blockSize >= 128) {if (tid < 64)  { sdata[tid] = (sdata[tid] + sdata[tid+64]); } __syncthreads(); } 
    // __syncthreads();
    if (tid < 32){
        if(blockSize >= 64) {
             sdata[tid] = (sdata[tid] + sdata[tid+32]);
            __syncthreads();
        }
        if(blockSize >= 32){
             sdata[tid] = (sdata[tid] + sdata[tid+16]);
            __syncthreads();
        } 
        if(blockSize >= 16){
             sdata[tid] = (sdata[tid] + sdata[tid+8]);
            __syncthreads();
        } 
        if (blockSize >= 8){
             sdata[tid] = (sdata[tid] + sdata[tid+4]);
            __syncthreads();
        }
        if(blockSize >= 4)  {
            sdata[tid] = (sdata[tid] + sdata[tid+2]);
            __syncthreads();
        }
        if (blockSize >= 2){
            sdata[tid] = (sdata[tid] + sdata[tid+1]);
            __syncthreads();
        } 
    }
    if (tid == 0) d_odata[blockIdx.x] = sdata[0];
}

void reduce6_switch(int block_num, int blockdim , const int* const d_idata, int* const d_odata){
    // std::cout << "cuda block_num: " << blockdim << std::endl;
    switch(blockdim){
        case 512:
            reduce6<512><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);          break;
        case 256:
            reduce6<256><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);          break;
        case 128:
            reduce6<128><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);          break;
        case 64:
            reduce6<64><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);           break;
        case 32:
            reduce6<32><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);           break;
        case 16:
            reduce6<16><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);           break;
        case 8:
            reduce6<8><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);            break;
        case 4:
            reduce6<4><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);            break;
        case 2:
            reduce6<2><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);            break;
        case 1:
            reduce6<1><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata);            break;
    }
}

template <unsigned int blockSize>
__global__ void reduce7(const int* const g_idata, int* const  g_odata, unsigned int n)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    // __syncthreads();
    if (tid < 32){
        if(blockSize >= 64) {
             sdata[tid] = (sdata[tid] + sdata[tid+32]);
            __syncthreads();
        }
        if(blockSize >= 32){
             sdata[tid] = (sdata[tid] + sdata[tid+16]);
            __syncthreads();
        } 
        if(blockSize >= 16){
             sdata[tid] = (sdata[tid] + sdata[tid+8]);
            __syncthreads();
        } 
        if (blockSize >= 8){
             sdata[tid] = (sdata[tid] + sdata[tid+4]);
            __syncthreads();
        }
        if(blockSize >= 4)  {
            sdata[tid] = (sdata[tid] + sdata[tid+2]);
            __syncthreads();
        }
        if (blockSize >= 2){
            sdata[tid] = (sdata[tid] + sdata[tid+1]);
            __syncthreads();
        } 
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void reduce7_switch(int block_num, int blockdim , const int* const d_idata, int* const d_odata, int size){
    // std::cout << "cuda block_num: " << blockdim << std::endl;
    switch(blockdim){
        case 512:
            reduce7<512><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);          break;
        case 256:
            reduce7<256><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);          break;
        case 128:
            reduce7<128><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);          break;
        case 64:
            reduce7<64><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);           break;
        case 32:
            reduce7<32><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);           break;
        case 16:
            reduce7<16><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);           break;
        case 8:
            reduce7<8><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);            break;
        case 4:
            reduce7<4><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);            break;
        case 2:
            reduce7<2><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);            break;
        case 1:
            reduce7<1><<<block_num, blockdim, SM_SIZE>>>(d_idata, d_odata, size);            break;
    }
}


void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, int* const d_odata, const int n) {
    //============================================================ 1th kernel ============================================================
    // int size = n;
    // int block_dim = 64;
    // int block_num = ((size-1) / block_dim) +1; 
    // int final_block_dim = block_dim;
    // if(block_num<=1){
    //     final_block_dim = size;
    //     reduce1<<< block_num, final_block_dim, SM_SIZE >>>(d_idata, d_odata);
    // }
    // else{
    //     reduce1<<< block_num, block_dim, SM_SIZE >>>(d_idata, d_odata);
    //     for (int i = block_num ; i >= block_dim; i /=block_dim){
    //         final_block_dim = i/block_dim;
    //         reduce1<<< i/block_dim, block_dim, SM_SIZE >>>(d_odata, d_odata);
    //     }
    //     reduce1<<< 1, final_block_dim, SM_SIZE >>>(d_odata, d_odata);
    // }
    
   //============================================================ 2th kernel ============================================================
    // int size = n;
    // int block_dim = 256;
    // int block_num = ((size-1) / block_dim) +1; 
    // int final_block_dim = block_dim;
    // if(block_num<=1){
    //     final_block_dim = size;
    //     reduce2<<< block_num, final_block_dim, SM_SIZE >>>(d_idata, d_odata);
    // }
    // else{
    //     reduce2<<< block_num, block_dim, SM_SIZE >>>(d_idata, d_odata);
    //     for (int i = block_num ; i >= block_dim; i /=block_dim){
    //         final_block_dim = i/block_dim;
    //         reduce2<<< i/block_dim, block_dim, SM_SIZE >>>(d_odata, d_odata);
    //     }
    //     reduce2<<< 1, final_block_dim, SM_SIZE >>>(d_odata, d_odata);
    // }

   //============================================================ 3th kernel ============================================================
    // int size = n;
    // int block_dim = 128;
    // int block_num = ((size-1) / block_dim) +1; 
    // int final_block_dim = block_dim;
    // if(block_num<=1){
    //     final_block_dim = size;
    //     reduce3<<< block_num, final_block_dim, SM_SIZE >>>(d_idata, d_odata);
    // }
    // else{
    //     reduce3<<< block_num, block_dim, SM_SIZE >>>(d_idata, d_odata);
    //     for (int i = block_num ; i >= block_dim; i /=block_dim){
    //         final_block_dim = i/block_dim;
    //         reduce3<<< i/block_dim, block_dim, SM_SIZE >>>(d_odata, d_odata);
    //     }
    //     reduce3<<< 1, final_block_dim, SM_SIZE >>>(d_odata, d_odata);
    // }
   //============================================================ 4th kernel ============================================================
    // int size = n;
    // int block_dim = 256;
    // int block_num = ((size-1) / block_dim) +1;
    // int final_block_dim = block_dim;
    // cout << "block_num: " << block_num << endl;
    // if(block_num/2 <= 1){
    //     final_block_dim = size/2;
    //     reduce4<<< 1, final_block_dim, SM_SIZE >>>(d_idata, d_odata);
    // }
    // else{
    //     reduce4<<< block_num/2, block_dim, SM_SIZE >>>(d_idata, d_odata);
    //     for (int i = block_num/2 ; i > block_dim; i /=(block_dim*2)){
    //         final_block_dim = i/block_dim/2;
    //         cout << "i/block_dim: " << i/block_dim << endl;
    //         cout << "i/block_dim*2: " << i/block_dim*2 << endl;
    //         cout << "final_block_dim: " << final_block_dim << endl;
    //         reduce4<<< i/block_dim, block_dim, SM_SIZE >>>(d_odata, d_odata);
    //     }
    //     reduce4<<< 1, final_block_dim, SM_SIZE >>>(d_odata, d_odata);
    // }
    
 
    //============================================================ 5th kernel ============================================================
    // 마지막 커널을 실행할 때 남아있는 element의 개수가 32보다 작으면 마지막 step은 4번 커널을 실행해야 됨.
    // int size = n;
    // int block_dim = 256;
    // int block_num = ((size-1) / block_dim) +1;
    // int final_block_dim = block_dim;
    // cout << "block_num: " << block_num << endl;
    // if(block_num/2 <= 1){
    //     final_block_dim = size/2;
    //     reduce5<<< 1, final_block_dim, SM_SIZE >>>(d_idata, d_odata);
    // }
    // else{
    //     reduce5<<< block_num/2, block_dim, SM_SIZE >>>(d_idata, d_odata);
    //     for (int i = block_num/2 ; i > block_dim; i /=(block_dim*2)){
    //         final_block_dim = i/block_dim/2;
    //         cout << "i/block_dim: " << i/block_dim << endl;
    //         cout << "i/block_dim*2: " << i/block_dim*2 << endl;
    //         cout << "final_block_dim: " << final_block_dim << endl;
    //         reduce5<<< i/block_dim, block_dim, SM_SIZE >>>(d_odata, d_odata);
    //     }
    //     if(final_block_dim < 32){
    //         reduce4<<< 1, final_block_dim, SM_SIZE >>>(d_odata, d_odata);
    //     }
    //     else{
    //         reduce5<<< 1, final_block_dim, SM_SIZE >>>(d_odata, d_odata);
    //     }
    // }

    //============================================================ 6th kernel ============================================================
    // int size = n;
    // int block_dim = 256;
    // int block_num = ((size-1) / block_dim)/2 +1;
    // reduce6_switch( block_num, block_dim, d_idata, d_odata);
    // for (int i = block_num ; i > block_dim; i /=(block_dim)){
    //     cout << "i: " << i << endl;
    //     reduce6_switch(size/i, block_dim, d_odata, d_odata);
    // }
    // reduce6_switch( 1, block_dim, d_odata, d_odata);
    
    
    //============================================================ 7th kernel ============================================================
    int size = n;
    int block_dim = 256;
    int block_num = ((size-1) / block_dim)/64 +1;
    cout << "block_num: " << block_num << endl;
    reduce7_switch( size/block_num, block_dim, d_idata, d_odata, size);
    reduce7_switch( 1, block_dim, d_odata, d_odata, size/block_num);

    //============================================================ test for more size of array ============================================================
    // long long int size = pow(2, 23);
    // cout << "size: " << size << endl;
    // int * randArray = new int [size];
    // for(long long int i=0;i<size;i++){
    //     randArray[i]=1;
    // }
    // int sum = 0;
    // for(int i = 0 ; i < size ; i++){
    //     sum += randArray[i];
    // }
    // cout << "sum: " << sum << endl; 
    // cudaMemcpyToDevice((void *)d_idata,(void *)randArray, sizeof(int)*size);
    
    // int block_dim = 256;
    // int block_num = ((size-1) / block_dim) +1; 
    // cout << "block_num: " << block_num << endl; 
    // int final_block_dim = block_dim;
    // if(block_num<=1){
    //     final_block_dim = size;
    //     reduce1<<< block_num, final_block_dim, SM_SIZE >>>(d_idata, d_odata);
    // }
    // else{
    //     reduce1<<< block_num, block_dim, SM_SIZE >>>(d_idata, d_odata);
    //     for (int i = block_num ; i >= block_dim; i /=block_dim){
    //         final_block_dim = i/block_dim;
    //         cout << "i: " << i << endl; 
    //         // cout << "i/block_dim: " << i/block_dim << endl; 
    //         reduce1<<< i/block_dim, block_dim, SM_SIZE >>>(d_odata, d_odata);
    //     }
    //     cout << "final_block_dim: " << final_block_dim << endl; 
    //     reduce1<<< 1, final_block_dim, SM_SIZE >>>(d_odata, d_odata);
    // }
    

}
