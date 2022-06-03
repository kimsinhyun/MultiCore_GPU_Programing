#include "vgg16_cuda.h"
#include <iostream>
//batch size fixed by 128
#define BATCH_SIZE 128
#define SM_SIZE 1024
#define CHANNLE 3
#define INPUT_SIZE 32
#define PAD_SIZE 1

void test_print_sinhyun(float* test_input, int size){
    float* temp_input = new float[size];
    cudaMemcpy(temp_input, test_input, sizeof(float) * size, cudaMemcpyDeviceToHost);
    int temp_count = 0;
    for (int i = 0 ; i < size ; i ++){
        std::cout << " " << temp_input[i];
        temp_count ++;
    }
    std::cout << std::endl;
    std::cout << "temp_count: " << temp_count << std::endl;
}

__global__ void normalize_kernel(const uint8_t* const d_image, float* output,int batch, float max_int, float mean, float var){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    output[tid] = (d_image[tid]/max_int - mean) / var;
}

void normalize(const uint8_t* const d_image, float* output, int batch, int channel, int input_size) {
    float max_int = 255.0L;
    float mean = 0.5L;
    float var = 0.5L;
    int block_size = batch;
    int block_num = (batch*channel*input_size*input_size)/block_size;
    normalize_kernel<<<block_num,block_size, SM_SIZE>>>(d_image, output, batch, max_int, mean, var);
}
__global__ void padding_kernel(float* intput, float* output, int batch, int channel, int input_size, int pad_size, int W_grid, int tile_size) // input,in_size, output, padsize
{
    int b = blockIdx.x; 
    int c = blockIdx.y; 
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; 
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; 
    int H_OUT = input_size+2*pad_size;
    int W_OUT = input_size+2*pad_size;
    int input_index = b * (channel * input_size * input_size) + c * (input_size * input_size) + h * (input_size) + w;
    int output_index = b * (channel * H_OUT * W_OUT) + c * (H_OUT * W_OUT) + (h+pad_size) * W_OUT + w+pad_size;
    output[output_index] = intput[input_index];
}
void padding(float* input, float* output, int batch, int channel, int input_size, int pad_size, int W_grid, int tile_size){
    dim3 dimGrid(batch, channel,1);
    dim3 dimBlock(tile_size, tile_size, 1);
    padding_kernel<<<dimGrid, dimBlock>>>(input, output, batch, channel, input_size, pad_size, W_grid, tile_size);
}

__global__ void conv_kernel(float* input, float* output, float* weight, float* bias, int H, int W, int input_C, int output_C, int filter_size, int W_grid, int tile_size)
{
    int b = blockIdx.x;  // which image
    int oc = blockIdx.y; // output channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // which height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // which width 
    int H_OUT = H - (filter_size - 1); 
    int W_OUT = W - (filter_size - 1); 
    int output_index = b * (output_C * H_OUT * W_OUT) + oc * (H_OUT * W_OUT) + h * W_OUT + w;
    float val = 0;
    for (int ic=0; ic<input_C; ic++){
        int input_index = b * (input_C * H * W) + ic * (H * W) + h * (W) + w;
        int kernel_base = oc * (input_C * filter_size * filter_size) + ic * (filter_size * filter_size);
        val += input[input_index + 0] * weight[kernel_base + 0];
        val += input[input_index + 1] * weight[kernel_base + 1];
        val += input[input_index + 2] * weight[kernel_base + 2];
        val += input[input_index + 1*W + 0] * weight[kernel_base + 1*filter_size];
        val += input[input_index + 1*W + 1] * weight[kernel_base + 1*filter_size + 1];
        val += input[input_index + 1*W + 2] * weight[kernel_base + 1*filter_size + 2];
        val += input[input_index + 2*W + 0] * weight[kernel_base + 2*filter_size];
        val += input[input_index + 2*W + 1] * weight[kernel_base + 2*filter_size + 1];
        val += input[input_index + 2*W + 2] * weight[kernel_base + 2*filter_size + 2];
    }
    output[output_index] = bias[oc] + val;
}

void conv(float* input, float* output, float* conv_weight, float* conv_bias,int batch, int input_size, int conv_input_channel, int conv_output_channel, int filter_size, int W_grid, int tile_size){
    dim3 dimGrid(batch, conv_output_channel, 1); // batch*64*1
    dim3 dimBlock(tile_size, tile_size, 1);
    conv_kernel <<< dimGrid, dimBlock >>> (input, output, conv_weight, conv_bias, input_size,
                                    input_size, conv_input_channel, conv_output_channel, filter_size, W_grid, tile_size);
}
__global__ void relu_kernel(float* input, int C, int H, int W, int W_grid, int tile_size)
{
    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // input channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // which height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // which width 
    int input_index = b * (C * H * W) + c * (H * W) + h * (W) + w;
    input[input_index] = max(input[input_index], (float)(0.0));
}
void relu(float* input, int batch, int input_channel, int input_size, int W_grid, int tile_size){
    dim3 dimGrid(batch, input_channel, 1); // batch*64*1
    dim3 dimBlock(tile_size, tile_size, 1);
    relu_kernel<<<dimGrid, dimBlock>>> (input, input_channel, input_size, input_size, W_grid, tile_size);
}

__global__ void pool_kernel(float* input, float* output, int C, int H, int W, int W_grid, int tile_width)
{   
    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // output channel
    int h = (blockIdx.z / W_grid)*tile_width + threadIdx.y; // output height
    int w = (blockIdx.z % W_grid)*tile_width + threadIdx.x; // output width 

    int pool_kernel_size = 2;
    int H_pool = H*pool_kernel_size;
    int W_pool = W*pool_kernel_size;
    float max_val = -100000000000000000000000000000.0f;
    int input_index = b * (C * H_pool * W_pool) + c * (H_pool * W_pool) + h*pool_kernel_size * (W_pool) + w*pool_kernel_size;

    // Find maximum
    for (int sh = 0; sh < pool_kernel_size; sh++)
        for (int sw = 0; sw < pool_kernel_size; sw++) {
            float val = input[input_index + sh * (W_pool) + sw];
            if (val - max_val > max_val) max_val = val;
        }
    int output_index = b * (C * H * W) + c * (H * W) + h * (W) + w;
    output[output_index] = max_val;
}
void pool(float* input, float* output,int batch, int input_channel, int input_size, int W_grid, int tile_width){
    dim3 dimGrid(batch, input_channel, 1); // batch*64*1
    dim3 dimBlock(tile_width, tile_width, 1);
    pool_kernel<<<dimGrid, dimBlock>>>(input, output, input_channel, input_size,input_size, W_grid, tile_width);
}

void vgg16_cuda::predict(int batch) {
    normalize(d_image, d_input, batch, 3, 32);        // channel 3 ; input_size 32
    padding(d_input,d_input_padded, batch, 3, 32, 1, ceil(32/32), 32); // channel 3 ; input_size 32, pad_size 1, grid width 1, tile width 32
    conv(d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias, batch, 34, 3, 64, 3, ceil(32/32), 32); // input_size 34, input channel 3, output channel 64, filter size 3
    relu(d_C1_1_feature_map, batch, 64, 32, (32/32), 32);            // channel 64, input size 32
    padding(d_C1_1_feature_map, d_C1_1_feature_map_padded, batch, 64, 32, 1, ceil(32/32), 32); // input channel 64, input size 32, pad size 1
    conv(d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias, batch, 34, 64, 64, 3, ceil(32/32), 32); // input_size 34, input channel 64, output channel 64, filter size 3
    relu(d_C1_2_feature_map, batch, 64, 32, ceil(32/32), 32);            // channel 64, input size 32
    pool(d_C1_2_feature_map, d_S1_feature_map , batch, 64, 16, ceil(16/16), 16); // pooling 후 image size 32 -> 16
    // test_print_sinhyun(d_S1_feature_map, 128*64*16*16);


    std::cout << "C2_1_size: " << C2_1_size << std::endl;
    padding(d_S1_feature_map, d_S1_feature_map_padded, batch, 64, 16, 1, (16/16), 16);      // channel 64 ; input_size 32, pad_size 1, grid width 1, tile width 16
    conv(d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias, batch, 18, 64, 64, 3, ceil(16/16), 16); // input_size 18, input channel 64, output channel 64, filter size 3, tile size 16
    relu(d_C2_1_feature_map, batch, 64, 16, ceil(16/16), 16);                               // channel 64, input size 16, tile size 16
    padding(d_C2_1_feature_map, d_C2_1_feature_map_padded, batch, 64, 16, 1, (16/16), 16);  // channel 64 ; input_size 32, pad_size 1, grid width 1, tile width 16
    conv(d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias, batch, 18, 64, 64, 3, ceil(16/16), 16); // input_size 18, input channel 64, output channel 64, filter size 3, tile size 16
    relu(d_C2_2_feature_map, batch, 64, 16, ceil(16/16), 16);
    pool(d_C2_2_feature_map, d_S2_feature_map , batch, 128, 8, ceil(8/8), 8); // pooling 후 image size 32 -> 16
    std::cout << "S2_channel: " << S2_channel << std::endl;
    std::cout << "S2_size: " << S2_size << std::endl;
    test_print_sinhyun(d_C2_2_feature_map, 128*128*8*8);
    // pool(d_C2_2_feature_map, d_S2_feature_map , batch, 64, 16, ceil(16/16), 16); // pooling 후 image size 32 -> 16

    //////////BLOCK 1/////////////////////////////////
    // TODO: Implement pad
    // TODO: Implement conv1_1
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv1_2
    // TODO: Implement relu
    // TODO: Implement pool

    //////////BLOCK 2/////////////////////////////////
    // TODO: Implement pad
    // TODO: Implement conv2_1
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv2_2
    // TODO: Implement relu
    // TODO: Implement pool

    //////////BLOCK 3/////////////////////////////////
    // TODO: Implement pad
    // TODO: Implement conv3_1
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv3_2
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv3_3
    // TODO: Implement relu
    // TODO: Implement pool

    //////////BLOCK 4/////////////////////////////////
    // TODO: Implement pad
    // TODO: Implement conv4_1
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv4_2
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv4_3
    // TODO: Implement relu
    // TODO: Implement pool

    //////////BLOCK 5/////////////////////////////////
    // TODO: Implement pad
    // TODO: Implement conv5_1
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv5_2
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv5_3
    // TODO: Implement relu
    // TODO: Implement pool

    // TODO: Implement fc1
    // TODO: Implement relu

    /* NOTE: unless you want to make a major change to this class structure, 
    *  you need to write your output to the device memory d_output 
    *  so that classify() can handle the rest.
    */
}




void vgg16_cuda::prepare_device_memory(uint8_t* image) {
  // Alloc Model Parameters

  //////////BLOCK 1/////////////////////////////////
  cudaMalloc((void**)&d_conv1_1_weight,
             sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                 conv1_1_kernel_size * conv1_1_kernel_size);
  cudaMalloc((void**)&d_conv1_1_bias, sizeof(float) * conv1_1_out_channel);
  cudaMalloc((void**)&d_conv1_2_weight,
             sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                 conv1_2_kernel_size * conv1_2_kernel_size);
  cudaMalloc((void**)&d_conv1_2_bias, sizeof(float) * conv1_2_out_channel);

  //////////BLOCK 2/////////////////////////////////
  cudaMalloc((void**)&d_conv2_1_weight,
             sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                 conv2_1_kernel_size * conv2_1_kernel_size);
  cudaMalloc((void**)&d_conv2_1_bias, sizeof(float) * conv2_1_out_channel);
  cudaMalloc((void**)&d_conv2_2_weight,
             sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                 conv2_2_kernel_size * conv2_2_kernel_size);
  cudaMalloc((void**)&d_conv2_2_bias, sizeof(float) * conv2_2_out_channel);

  //////////BLOCK 3/////////////////////////////////
  cudaMalloc((void**)&d_conv3_1_weight,
             sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                 conv3_1_kernel_size * conv3_1_kernel_size);
  cudaMalloc((void**)&d_conv3_1_bias, sizeof(float) * conv3_1_out_channel);
  cudaMalloc((void**)&d_conv3_2_weight,
             sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                 conv3_2_kernel_size * conv3_2_kernel_size);
  cudaMalloc((void**)&d_conv3_2_bias, sizeof(float) * conv3_2_out_channel);
  cudaMalloc((void**)&d_conv3_3_weight,
             sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                 conv3_3_kernel_size * conv3_3_kernel_size);
  cudaMalloc((void**)&d_conv3_3_bias, sizeof(float) * conv3_3_out_channel);

  //////////BLOCK 4/////////////////////////////////
  cudaMalloc((void**)&d_conv4_1_weight,
             sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                 conv4_1_kernel_size * conv4_1_kernel_size);
  cudaMalloc((void**)&d_conv4_1_bias, sizeof(float) * conv4_1_out_channel);
  cudaMalloc((void**)&d_conv4_2_weight,
             sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                 conv4_2_kernel_size * conv4_2_kernel_size);
  cudaMalloc((void**)&d_conv4_2_bias, sizeof(float) * conv4_2_out_channel);
  cudaMalloc((void**)&d_conv4_3_weight,
             sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                 conv4_3_kernel_size * conv4_3_kernel_size);
  cudaMalloc((void**)&d_conv4_3_bias, sizeof(float) * conv4_3_out_channel);

  //////////BLOCK 5/////////////////////////////////
  cudaMalloc((void**)&d_conv5_1_weight,
             sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                 conv5_1_kernel_size * conv5_1_kernel_size);
  cudaMalloc((void**)&d_conv5_1_bias, sizeof(float) * conv5_1_out_channel);
  cudaMalloc((void**)&d_conv5_2_weight,
             sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                 conv5_2_kernel_size * conv5_2_kernel_size);
  cudaMalloc((void**)&d_conv5_2_bias, sizeof(float) * conv5_2_out_channel);
  cudaMalloc((void**)&d_conv5_3_weight,
             sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                 conv5_3_kernel_size * conv5_3_kernel_size);
  cudaMalloc((void**)&d_conv5_3_bias, sizeof(float) * conv5_3_out_channel);

  //////////FC 1////////////////////////////////////
  cudaMalloc((void**)&d_fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel);
  cudaMalloc((void**)&d_fc1_bias, sizeof(float) * fc1_out_channel);

  // Alloc Activations
  cudaMalloc((void**)&d_image,
             sizeof(uint8_t) * batch * input_size * input_size * input_channel);
  cudaMalloc((void**)&d_input,
             sizeof(float) * batch * input_channel * input_size * input_size);

  //////////BLOCK 1/////////////////////////////////
  cudaMalloc((void**)&d_input_padded,
             sizeof(float) * batch * input_channel * (input_size+2*conv1_1_padding_size) * (input_size+2*conv1_1_padding_size));
  cudaMalloc((void**)&d_C1_1_feature_map,
             sizeof(float) * batch * C1_1_channel * C1_1_size * C1_1_size);
  cudaMalloc((void**)&d_C1_1_feature_map_padded,
             sizeof(float) * batch * C1_1_channel * (C1_1_size+2*conv1_2_padding_size) * (C1_1_size+2*conv1_2_padding_size));
  cudaMalloc((void**)&d_C1_2_feature_map,
             sizeof(float) * batch * C1_2_channel * C1_2_size * C1_2_size);
  cudaMalloc((void**)&d_S1_feature_map,
             sizeof(float) * batch * S1_channel * S1_size * S1_size);

  //////////BLOCK 2/////////////////////////////////
  cudaMalloc((void**)&d_S1_feature_map_padded,
             sizeof(float) * batch * S1_channel * (S1_size+2*conv2_1_padding_size) * (S1_size+2*conv2_1_padding_size));
  cudaMalloc((void**)&d_C2_1_feature_map,
             sizeof(float) * batch * C2_1_channel * C2_1_size * C2_1_size);
  cudaMalloc((void**)&d_C2_1_feature_map_padded,
             sizeof(float) * batch * C2_1_channel * (C2_1_size+2*conv2_2_padding_size) * (C2_1_size+2*conv2_2_padding_size));
  cudaMalloc((void**)&d_C2_2_feature_map,
             sizeof(float) * batch * C2_2_channel * C2_2_size * C2_2_size);
  cudaMalloc((void**)&d_S2_feature_map,
             sizeof(float) * batch * S2_channel * S2_size * S2_size);

  //////////BLOCK 3/////////////////////////////////
  cudaMalloc((void**)&d_S2_feature_map_padded,
             sizeof(float) * batch * S2_channel * (S2_size+2*conv3_1_padding_size) * (S2_size+2*conv3_1_padding_size));
  cudaMalloc((void**)&d_C3_1_feature_map,
             sizeof(float) * batch * C3_1_channel * C3_1_size * C3_1_size);
  cudaMalloc((void**)&d_C3_1_feature_map_padded,
             sizeof(float) * batch * C3_1_channel * (C3_1_size+2*conv3_2_padding_size) * (C3_1_size+2*conv3_2_padding_size));
  cudaMalloc((void**)&d_C3_2_feature_map,
             sizeof(float) * batch * C3_2_channel * C3_2_size * C3_2_size);
  cudaMalloc((void**)&d_C3_2_feature_map_padded,
             sizeof(float) * batch * C3_2_channel * (C3_2_size+2*conv3_3_padding_size) * (C3_2_size+2*conv3_3_padding_size));
  cudaMalloc((void**)&d_C3_3_feature_map,
             sizeof(float) * batch * C3_3_channel * C3_3_size * C3_3_size);
  cudaMalloc((void**)&d_S3_feature_map,
             sizeof(float) * batch * S3_channel * S3_size * S3_size);

  //////////BLOCK 4/////////////////////////////////
  cudaMalloc((void**)&d_S3_feature_map_padded,
             sizeof(float) * batch * S3_channel * (S3_size+2*conv4_1_padding_size) * (S3_size+2*conv4_1_padding_size));
  cudaMalloc((void**)&d_C4_1_feature_map,
             sizeof(float) * batch * C4_1_channel * C4_1_size * C4_1_size);
  cudaMalloc((void**)&d_C4_1_feature_map_padded,
             sizeof(float) * batch * C4_1_channel * (C4_1_size+2*conv4_2_padding_size) * (C4_1_size+2*conv4_2_padding_size));
  cudaMalloc((void**)&d_C4_2_feature_map,
             sizeof(float) * batch * C4_2_channel * C4_2_size * C4_2_size);
  cudaMalloc((void**)&d_C4_2_feature_map_padded,
             sizeof(float) * batch * C4_2_channel * (C4_2_size+2*conv4_3_padding_size) * (C4_2_size+2*conv4_3_padding_size));
  cudaMalloc((void**)&d_C4_3_feature_map,
             sizeof(float) * batch * C4_3_channel * C4_3_size * C4_3_size);
  cudaMalloc((void**)&d_S4_feature_map,
             sizeof(float) * batch * S4_channel * S4_size * S4_size);

  //////////BLOCK 5/////////////////////////////////
  cudaMalloc((void**)&d_S4_feature_map_padded,
             sizeof(float) * batch * S4_channel * (S4_size+2*conv5_1_padding_size) * (S4_size+2*conv5_1_padding_size));
  cudaMalloc((void**)&d_C5_1_feature_map,
             sizeof(float) * batch * C5_1_channel * C5_1_size * C5_1_size);
  cudaMalloc((void**)&d_C5_1_feature_map_padded,
             sizeof(float) * batch * C5_1_channel * (C5_1_size+2*conv5_2_padding_size) * (C5_1_size+2*conv5_2_padding_size));
  cudaMalloc((void**)&d_C5_2_feature_map,
             sizeof(float) * batch * C5_2_channel * C5_2_size * C5_2_size);
  cudaMalloc((void**)&d_C5_2_feature_map_padded,
             sizeof(float) * batch * C5_2_channel * (C5_2_size+2*conv5_3_padding_size) * (C5_2_size+2*conv5_3_padding_size));
  cudaMalloc((void**)&d_C5_3_feature_map,
             sizeof(float) * batch * C5_3_channel * C5_3_size * C5_3_size);
  cudaMalloc((void**)&d_S5_feature_map,
             sizeof(float) * batch * S5_channel * S5_size * S5_size);


  cudaMalloc((void**)&d_output, sizeof(float) * batch * output_size);

  // Copy Parameters
  //////////BLOCK 1/////////////////////////////////
  cudaMemcpy(d_conv1_1_weight, conv1_1_weight,
             sizeof(float) * conv1_1_in_channel * conv1_1_out_channel *
                 conv1_1_kernel_size * conv1_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_1_bias, conv1_1_bias, sizeof(float) * conv1_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv1_2_weight, conv1_2_weight,
              sizeof(float) * conv1_2_in_channel * conv1_2_out_channel *
                  conv1_2_kernel_size * conv1_2_kernel_size,
              cudaMemcpyHostToDevice);
   cudaMemcpy(d_conv1_2_bias, conv1_2_bias, sizeof(float) * conv1_2_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 2/////////////////////////////////
  cudaMemcpy(d_conv2_1_weight, conv2_1_weight,
             sizeof(float) * conv2_1_in_channel * conv2_1_out_channel *
                 conv2_1_kernel_size * conv2_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_1_bias, conv2_1_bias, sizeof(float) * conv2_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_2_weight, conv2_2_weight,
              sizeof(float) * conv2_2_in_channel * conv2_2_out_channel *
                  conv2_2_kernel_size * conv2_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv2_2_bias, conv2_2_bias, sizeof(float) * conv2_2_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 3/////////////////////////////////
  cudaMemcpy(d_conv3_1_weight, conv3_1_weight,
             sizeof(float) * conv3_1_in_channel * conv3_1_out_channel *
                 conv3_1_kernel_size * conv3_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_1_bias, conv3_1_bias, sizeof(float) * conv3_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_2_weight, conv3_2_weight,
              sizeof(float) * conv3_2_in_channel * conv3_2_out_channel *
                  conv3_2_kernel_size * conv3_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_2_bias, conv3_2_bias, sizeof(float) * conv3_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_3_weight, conv3_3_weight,
              sizeof(float) * conv3_3_in_channel * conv3_3_out_channel *
                  conv3_3_kernel_size * conv3_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv3_3_bias, conv3_3_bias, sizeof(float) * conv3_3_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 4/////////////////////////////////
  cudaMemcpy(d_conv4_1_weight, conv4_1_weight,
             sizeof(float) * conv4_1_in_channel * conv4_1_out_channel *
                 conv4_1_kernel_size * conv4_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_1_bias, conv4_1_bias, sizeof(float) * conv4_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_2_weight, conv4_2_weight,
              sizeof(float) * conv4_2_in_channel * conv4_2_out_channel *
                  conv4_2_kernel_size * conv4_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_2_bias, conv4_2_bias, sizeof(float) * conv4_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_3_weight, conv4_3_weight,
              sizeof(float) * conv4_3_in_channel * conv4_3_out_channel *
                  conv4_3_kernel_size * conv4_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv4_3_bias, conv4_3_bias, sizeof(float) * conv4_3_out_channel,
              cudaMemcpyHostToDevice);

  //////////BLOCK 5/////////////////////////////////
  cudaMemcpy(d_conv5_1_weight, conv5_1_weight,
             sizeof(float) * conv5_1_in_channel * conv5_1_out_channel *
                 conv5_1_kernel_size * conv5_1_kernel_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_1_bias, conv5_1_bias, sizeof(float) * conv5_1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_2_weight, conv5_2_weight,
              sizeof(float) * conv5_2_in_channel * conv5_2_out_channel *
                  conv5_2_kernel_size * conv5_2_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_2_bias, conv5_2_bias, sizeof(float) * conv5_2_out_channel,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_3_weight, conv5_3_weight,
              sizeof(float) * conv5_3_in_channel * conv5_3_out_channel *
                  conv5_3_kernel_size * conv5_3_kernel_size,
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_conv5_3_bias, conv5_3_bias, sizeof(float) * conv5_3_out_channel,
              cudaMemcpyHostToDevice);


  cudaMemcpy(d_fc1_weight, fc1_weight,
             sizeof(float) * fc1_in_channel * fc1_out_channel,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_bias, fc1_bias, sizeof(float) * fc1_out_channel,
             cudaMemcpyHostToDevice);

  // copy input image
  size_t image_size = batch * input_size * input_size * input_channel;
  cudaMemcpy(d_image, image, image_size * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
}

void vgg16_cuda::classify(int* predict, int batch) {
  // read logits back to cpu
  cudaMemcpy(output, d_output, sizeof(float) * output_size * batch,
             cudaMemcpyDeviceToHost);
  // Softmax
  softmax(output, predict, batch, output_size);
}

vgg16_cuda::~vgg16_cuda() {
  cudaFree(d_conv1_1_weight);   
  cudaFree(d_conv1_2_weight);   
  cudaFree(d_conv2_1_weight);   
  cudaFree(d_conv2_2_weight);  
  cudaFree(d_conv3_1_weight);   
  cudaFree(d_conv3_2_weight);   
  cudaFree(d_conv3_3_weight);   
  cudaFree(d_conv4_1_weight);   
  cudaFree(d_conv4_2_weight);   
  cudaFree(d_conv4_3_weight); 
  cudaFree(d_conv5_1_weight);   
  cudaFree(d_conv5_2_weight);   
  cudaFree(d_conv5_3_weight);   
 
  cudaFree(d_conv1_1_bias);   
  cudaFree(d_conv1_2_bias);   
  cudaFree(d_conv2_1_bias);   
  cudaFree(d_conv2_2_bias);  
  cudaFree(d_conv3_1_bias);   
  cudaFree(d_conv3_2_bias);   
  cudaFree(d_conv3_3_bias);   
  cudaFree(d_conv4_1_bias);   
  cudaFree(d_conv4_2_bias);   
  cudaFree(d_conv4_3_bias); 
  cudaFree(d_conv5_1_bias);   
  cudaFree(d_conv5_2_bias);   
  cudaFree(d_conv5_3_bias);   
   
  cudaFree(d_fc1_weight);     
  cudaFree(d_fc1_bias);        

  cudaFree(d_image);          
  cudaFree(d_input); 

  cudaFree(d_input_padded);          
  cudaFree(d_C1_1_feature_map); 
  cudaFree(d_C1_1_feature_map_padded); 
  cudaFree(d_C1_2_feature_map); 
  cudaFree(d_S1_feature_map); 

  cudaFree(d_S1_feature_map_padded); 
  cudaFree(d_C2_1_feature_map); 
  cudaFree(d_C2_1_feature_map_padded); 
  cudaFree(d_C2_2_feature_map); 
  cudaFree(d_S2_feature_map); 

  cudaFree(d_S2_feature_map_padded); 
  cudaFree(d_C3_1_feature_map); 
  cudaFree(d_C3_1_feature_map_padded); 
  cudaFree(d_C3_2_feature_map); 
  cudaFree(d_C3_2_feature_map_padded); 
  cudaFree(d_C3_3_feature_map); 
  cudaFree(d_S3_feature_map); 

  cudaFree(d_S3_feature_map_padded); 
  cudaFree(d_C4_1_feature_map); 
  cudaFree(d_C4_1_feature_map_padded); 
  cudaFree(d_C4_2_feature_map); 
  cudaFree(d_C4_2_feature_map_padded); 
  cudaFree(d_C4_3_feature_map); 
  cudaFree(d_S4_feature_map); 

  cudaFree(d_S4_feature_map_padded); 
  cudaFree(d_C5_1_feature_map); 
  cudaFree(d_C5_1_feature_map_padded); 
  cudaFree(d_C5_2_feature_map); 
  cudaFree(d_C5_2_feature_map_padded); 
  cudaFree(d_C5_3_feature_map); 
  cudaFree(d_S5_feature_map); 
 
  cudaFree(d_output);       
  cudaFree(d_predict_cuda);   
}
