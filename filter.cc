#include <stdlib.h>
// #include <cstdio>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <future>

// void worker(int N,int FILTER_SIZE, int thread_num, float* array_in, float* array_out_serial,  const float* k);
void worker(int N, int FILTER_SIZE, float* array_in, float* array_out_parallel, const float* k, int index,int block_size);

int main(int argc, char** argv) 
{
  if(argc < 2) std::cout<<"Usage : ./filter num_items"<<std::endl;
  int N = atoi(argv[1]);      //1073741824  1072693248
  int NT=16; //Default value. change it as you like.

  //0. Initialize
  const int FILTER_SIZE=5;
  const float k[FILTER_SIZE] = {0.125, 0.25, 0.25, 0.25, 0.125};
  float *array_in = new float[N];
  float *array_out_serial = new float[N];
  float *array_out_parallel = new float[N];
  {
    std::chrono::duration<float> diff;
    auto start = std::chrono::steady_clock::now();
    for(int i=0;i<N;i++) {
      array_in[i] = i;
    }
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"init took "<<diff.count()<<" sec"<<std::endl;
  }

  {
    //1. Serial
    std::chrono::duration<float> diff;
    auto start = std::chrono::steady_clock::now();
    for(int i=0;i<N;i++) {
      for(int j=0;j<FILTER_SIZE;j++) {
        array_out_serial[i] += array_in[i+j] * k[j];
      }
    }
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"serial 1D filter took "<<diff.count()<<" sec"<<std::endl;
  }

  {
    //2. parallel 1D filter
    std::chrono::duration<float> diff;
    auto start = std::chrono::steady_clock::now();
    /* TODO: put your own parallelized 1D filter here */
    /****************/
    
    //N=1073741824;
    int split_num = 2;
    int *thread_start_index = new int[split_num];
    int block_size = N/split_num;     //1048576
    int block_size_thread = block_size/NT; 
    // std::cout << "block_size: " <<  block_size <<  std::endl;
    for (int i=0 ; i<split_num ; i++){
      thread_start_index[i] = i*block_size;
    }
    // for (int i=0 ; i<split_num ; i++){
    //   std::cout << i << "th item: " << thread_start_index[i] << std::endl;
    // }
    
    for(int i=0; i<split_num; i++){
      std::vector<std::thread> workers;
      for(int j = 0 ; j<NT; j++){
        std::cout << "i: " << i << ",  j:"<< j << std::endl;
        workers.push_back(std::thread(worker, N, FILTER_SIZE, array_in, array_out_parallel, k, thread_start_index[i]+block_size_thread*j, block_size_thread));
      }
      for(int i=0 ; i<NT ;i++){
        workers[i].join();
      }

    }
    
    

    /****************/
    /* TODO: put your own parallelized 1D filter here */
    auto end = std::chrono::steady_clock::now();
    diff = end-start;
    std::cout<<"parallel 1D filter took "<<diff.count()<<" sec"<<std::endl;
    int error_counts=0;
    const float epsilon = 0.01;
    for(int i=0;i<N;i++) {
      float err= std::abs(array_out_serial[i] - array_out_parallel[i]);
      if(err > epsilon) {
        error_counts++;
        if(error_counts < 5) {
          std::cout<<"ERROR at "<<i<<": Serial["<<i<<"] = "<<array_out_serial[i]<<" Parallel["<<i<<"] = "<<array_out_parallel[i]<<std::endl;
          std::cout<<"err: "<<err<<std::endl;
        }
      }
    }

  // for(int i=0;i<N;i++) {
  //   std::cout << array_out_parallel[i] << " ";
  // }

    if(error_counts==0) {
      std::cout<<"PASS"<<std::endl;
    } else {
      std::cout<<"There are "<<error_counts<<" errors"<<std::endl;
    }

  }
  return 0;
}




// for(int i=0;i<N;i++) {
//       for(int j=0;j<FILTER_SIZE;j++) {
//         array_out_serial[i] += array_in[i+j] * k[j];
//       }
//     }
void worker(int N, int FILTER_SIZE, float* array_in, float* array_out_parallel, const float* k, int index,int block_size_thread){
  std::cout << "test worker " << index << std::endl;
  for(int i=index; i<index+block_size_thread; i++){
    // std::cout << "i(index): " << i << std::endl;
    for(int j=0; j<FILTER_SIZE; j++){
        array_out_parallel[i] += array_in[i+j] * k[j];
    }
  }
}