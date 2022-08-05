#include "matmul.h"
#include <iostream>

void matmul_ref(const int* const matrixA, const int* const matrixB,
                int* const matrixC, const int n, const int m) {
  // You can assume matrixC is initialized with zero
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < m; k++)
        matrixC[i * n + j] += matrixA[i * m + k] * matrixB[k * n + j];
}

void matmul_optimized(const int* const matrixA, const int* const matrixB,
                      int* const matrixC, const int n, const int m) {
  // TODO: Implement your code
  int *temp_matrixB = new int[n*m];
  #pragma omp parallel for
  for(int i=0; i<n; i++){
    for(int j=0 ; j<m ; j++){
      temp_matrixB[i*m+j] = matrixB[j*n+i];
    }
  }
  int block = 256;

  
  // #pragma omp parallel for
  #pragma omp parallel
  {
    int temp=0;
    int i;
    int j;
    int k;
    int ii;
    int jj;
    int kk;
  
    #pragma omp for
    for (i = 0; i < n; i+=block){
      for (j = 0; j < n; j+=block){
        for (k = 0; k < m; k+=block){
          for(ii=0; ii < block ; ii++){
            for(jj=0; jj < block ; jj++){
              temp = 0;
              for(kk=0; kk < block ; kk++){
                temp = temp + matrixA[(i+ii) * m + (k+kk)] * temp_matrixB[(j+jj) * m + (k+kk)];
              }
              matrixC[(i+ii) * n + (j+jj)] = matrixC[(i+ii) * n + (j+jj)] + temp;
            }  
          }
        }
      }
    }
  }
}

