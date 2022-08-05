#include <stdio.h>
#include <iostream>
#include <chrono>
#include <assert.h>
#include "matmul.h"
using namespace std;

#define TILE_WIDTH 32


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

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        matrixC[i * n + j] += matrixA[i * n + k] * matrixB[k * n + j];
}

__global__ void gpu_matrix_mult(const int *a,const int *b, int* const c, int n)
// __device__ void gpu_matrix_mult(const int *a,const int *b, int* const c, int n)
{ 

    // __shared__ int subTileM[TILE_WIDTH][TILE_WIDTH];
    // __shared__ int subTileN[TILE_WIDTH][TILE_WIDTH];
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < n && row < n) 
    {
        int temp_sum = 0;
        for(int i = 0; i < n; i++) 
        {
            temp_sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = temp_sum;
    }
} 


__global__ void gpu_matrix_mult_shared_mem(const int *a,const int *b, int* const c, int width)
{ 
    __shared__ int subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ int subTileN[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int temp_sum = 0;
    for(int m = 0; m < width/TILE_WIDTH; ++m) 
    {
      subTileM[ty][tx] = a[Row*width + m*TILE_WIDTH+tx];
      subTileN[ty][tx] = b[(m*TILE_WIDTH+ty)*width+Col];
      __syncthreads();
      for(int k = 0 ; k <TILE_WIDTH ; ++k)
        temp_sum += subTileM[ty][k] * subTileN[k][tx];
      __syncthreads();
      
    }
    c[Row * width + Col] = temp_sum;
} 


void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int* d_A, const int* d_B,  int* const d_C, const int n) {
  // TODO: Implement your CUDA code
  // cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
  cudaMemcpy((void *)d_A, (const void *) matrixA, sizeof(int)*n*n, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_B, (const void *)matrixB, sizeof(int)*n*n, cudaMemcpyHostToDevice);
  int width = n;
  // int tile_width = 32;
  dim3 dimGrid(width/TILE_WIDTH, width/TILE_WIDTH,1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  // gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);  
  gpu_matrix_mult_shared_mem<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);  

  cudaMemcpy((void *)matrixC, (void *)d_C, sizeof(int)*n*n, cudaMemcpyDeviceToHost);  
  cudaDeviceSynchronize();
}
