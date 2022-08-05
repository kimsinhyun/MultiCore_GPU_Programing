#include "vgg16_cuda.h"
#include <iostream>

void test_print_sinhyun(float* test_input, int size){
    float* temp_input = new float[size];
    cudaMemcpy(temp_input, test_input, sizeof(float) * size, cudaMemcpyDeviceToHost);
    int temp_count = 0;
    for (int i = 0 ; i < size ; i ++){
        // if(i%10==0) std::cout << std::endl;
        std::cout << " " << temp_input[i];
        temp_count ++;
    }
    std::cout << std::endl;
    std::cout << "temp_count: " << temp_count << std::endl;
}
void test_file_write_sinhyun(float* test_input, int size){
    float* temp_input = new float[size];
    cudaMemcpy(temp_input, test_input, sizeof(float) * size, cudaMemcpyDeviceToHost);
    std::ofstream ofile("test_print.txt");
    int temp_count = 0;
    if (ofile.is_open()) {
        for (int i = 0 ; i < size ; i ++){
            ofile << " " << temp_input[i];
            temp_count ++;
        }
        ofile.close();
    }
}

__global__ void normalize_kernel(const uint8_t* const image, float* output, int channel, int input_size, int W_grid, int tile_size) {
  
    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // input channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // input height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // input width 
    float max = 255.0L;    float mean = 0.5L;    float var = 0.5L;
    int base = b * (channel * input_size * input_size) + c * (input_size * input_size) + h * input_size + w;
    output[base] = (image[base]/max - mean) / var;
}

void normalize(const uint8_t* const d_image, float* output, int batch, int channel, int input_size, int W_grid, int tile_size) {
    dim3 dimGrid(batch, channel,1);
    dim3 dimBlock(tile_size, tile_size, 1);
    normalize_kernel<<<dimGrid,dimBlock>>>(d_image, output, channel, input_size, W_grid, tile_size);
}

__global__ void padding_kernel(float* intput, float* output, int batch, int channel, int input_size, int pad_size, int W_grid, int tile_size) // input,in_size, output, padsize
{
    int b = blockIdx.x; 
    int c = blockIdx.y; 
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; 
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; 
    int h_out = input_size+2*pad_size;
    int w_out = input_size+2*pad_size;
    int input_index = b * (channel * input_size * input_size) + c * (input_size * input_size) + h * (input_size) + w;
    int output_index = b * (channel * h_out * w_out) + c * (h_out * w_out) + (h+pad_size) * w_out + w+pad_size;
    output[output_index] = intput[input_index];
}

void padding(float* input, float* output, int batch, int channel, int input_size, int pad_size, int W_grid, int tile_size){
    dim3 dimGrid(batch, channel,1);
    dim3 dimBlock(tile_size, tile_size, 1);
    padding_kernel<<<dimGrid, dimBlock>>>(input, output, batch, channel, input_size, pad_size, W_grid, tile_size);
}

__global__ void conv_kernel(float* input, float* output, float* weight, float* bias, int input_size, int input_C, int output_C, int filter_size, int W_grid, int tile_size)
{
    int b = blockIdx.x;  // which image
    int c = blockIdx.y; // output channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // which height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // which width 
    int h_out = input_size - (filter_size - 1); 
    int w_out = input_size - (filter_size - 1); 
    int output_index = b * (output_C * h_out * w_out) + c * (h_out * w_out) + h * w_out + w;
    float temp = 0;
    for (int ic=0; ic<input_C; ic++){
        int input_index = b * (input_C * input_size * input_size) + ic * (input_size * input_size) + h * (input_size) + w;
        int kernel_index = c * (input_C * filter_size * filter_size) + ic * (filter_size * filter_size);
        temp += input[input_index + 0] * weight[kernel_index + 0];
        temp += input[input_index + 1] * weight[kernel_index + 1];
        temp += input[input_index + 2] * weight[kernel_index + 2];
        temp += input[input_index + 1 * input_size + 0] * weight[kernel_index + 1 * filter_size];
        temp += input[input_index + 1 * input_size + 1] * weight[kernel_index + 1 * filter_size + 1];
        temp += input[input_index + 1 * input_size + 2] * weight[kernel_index + 1 * filter_size + 2];
        temp += input[input_index + 2 * input_size + 0] * weight[kernel_index + 2 * filter_size];
        temp += input[input_index + 2 * input_size + 1] * weight[kernel_index + 2 * filter_size + 1];
        temp += input[input_index + 2 * input_size + 2] * weight[kernel_index + 2 * filter_size + 2];
    }
    output[output_index] = bias[c] + temp;
}


void conv(float* input, float* output, float* conv_weight, float* conv_bias,int batch, int input_size, int conv_input_channel, int conv_output_channel, int filter_size, int W_grid, int tile_size){
    dim3 dimGrid(batch, conv_output_channel, 1); // batch*channel*1
    dim3 dimBlock(tile_size, tile_size, 1);
    conv_kernel <<< dimGrid, dimBlock >>> (input, output, conv_weight, conv_bias, input_size, conv_input_channel, conv_output_channel, filter_size, W_grid, tile_size);
}
__global__ void relu_kernel(float* input, int output_C, int input_size, int W_grid, int tile_size)
{
    int b = blockIdx.x; // mini_batch
    int c = blockIdx.y; // input channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // which height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // which width 
    int input_index = b * (output_C * input_size * input_size) + c * (input_size * input_size) + h * (input_size) + w;
    input[input_index] = max(input[input_index], (float)(0.0));
}
void relu(float* input, int batch, int input_channel, int input_size, int W_grid, int tile_size){
    dim3 dimGrid(batch, input_channel, 1); 
    dim3 dimBlock(tile_size, tile_size, 1);
    relu_kernel<<<dimGrid, dimBlock>>> (input, input_channel, input_size, W_grid, tile_size);
}

__global__ void pool_kernel(float* input, float* output, int output_C, int input_size, int W_grid, int tile_size)
{   
    int b = blockIdx.x; // per batch
    int c = blockIdx.y; // output channel
    int h = (blockIdx.z / W_grid)*tile_size + threadIdx.y; // output height
    int w = (blockIdx.z % W_grid)*tile_size + threadIdx.x; // output width 
    int pool_kernel_size = 2;
    int h_pool = input_size*pool_kernel_size;
    int w_pool = input_size*pool_kernel_size;
    float max = -3.4e+38;           //std::numeric_limits<float>::lowest()
    int epsilon = 1.19209e-07;      //std::numeric_limits<float>::epsilon()

    int input_index = b * (output_C * h_pool * w_pool) + c * (h_pool * w_pool) + h*pool_kernel_size * (w_pool) + w*pool_kernel_size;

    for (int kh = 0; kh < pool_kernel_size; kh++)
        for (int kw = 0; kw < pool_kernel_size; kw++) {
            float temp = input[input_index + kh * (w_pool) + kw];
            if (temp - max > epsilon ) max = temp;
        }
    int output_index = b * (output_C * input_size * input_size) + c * (input_size * input_size) + h * (input_size) + w;
    output[output_index] = max;
}
void pool(float* input, float* output,int batch, int input_channel, int input_size, int W_grid, int tile_size){
    dim3 dimGrid(batch, input_channel, 1); 
    dim3 dimBlock(tile_size, tile_size, 1);
    pool_kernel<<<dimGrid, dimBlock>>>(input, output, input_channel, input_size, W_grid, tile_size);
}

__global__ void fc_kernel(float* input, float* output, float* weight, float* bias, int input_C, int output_C)
{
    int b = blockIdx.x; // mini batch
    int c = blockIdx.y; // output channel index

    float temp = 0;
    for (int ic=0; ic<input_C; ic++)
        temp += weight[c * input_C + ic] * input[b * input_C + ic];

    output[b * output_C + c] = bias[c] + temp;
}

void fc(float* input, float* output, float* weight, float* bias, int batch, int input_C, int output_C){
    dim3 dimGrid(batch, output_C, 1); 
    dim3 dimBlock(1,1,1);
    fc_kernel<<<dimGrid, dimBlock>>>(input, output, weight, bias,input_C, output_C);
}


void vgg16_cuda::predict(int batch) {
    normalize(d_image, d_input, batch, 3, 32, ceil(32/32), 32);        // channel 3 ; input_size 32
    //////////BLOCK 1/////////////////////////////////
    // TODO: Implement pad
    // TODO: Implement conv1_1
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv1_2
    // TODO: Implement relu
    // TODO: Implement pool
    padding(d_input,d_input_padded, batch, 3, 32, 1, ceil(32/32), 32); // channel 3 ; input_size 32, pad_size 1, grid width 1, tile width 32
    conv(d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias, batch, 34, 3, 64, 3, ceil(32/32), 32); // input_size 34, input channel 3, output channel 64, filter size 3
    // conv_sm(d_input_padded, d_C1_1_feature_map, d_conv1_1_weight, d_conv1_1_bias, batch, 34, 3, 64, 3, ceil(32/32), 32); // input_size 34, input channel 3, output channel 64, filter size 3
    relu(d_C1_1_feature_map, batch, 64, 32, ceil(32/32), 32);            // channel 64, input size 32
    padding(d_C1_1_feature_map, d_C1_1_feature_map_padded, batch, 64, 32, 1, ceil(32/32), 32); // input channel 64, input size 32, pad size 1
    conv(d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias, batch, 34, 64, 64, 3, ceil(32/32), 32); // input_size 34, input channel 64, output channel 64, filter size 3
    // conv_sm(d_C1_1_feature_map_padded, d_C1_2_feature_map, d_conv1_2_weight, d_conv1_2_bias, batch, 34, 64, 64, 3, ceil(32/32), 32); // input_size 34, input channel 64, output channel 64, filter size 3
    relu(d_C1_2_feature_map, batch, 64, 32, ceil(32/32), 32);            // channel 64, input size 32
    pool(d_C1_2_feature_map, d_S1_feature_map , batch, 64, 16, ceil(16/16), 16); // input channel 64, output size 16       pooling 후 image size 32 -> 16

    //////////BLOCK 2/////////////////////////////////
    // TODO: Implement pad
    // TODO: Implement conv2_1
    // TODO: Implement relu
    // TODO: Implement pad
    // TODO: Implement conv2_2
    // TODO: Implement relu
    // TODO: Implement pool
    padding(d_S1_feature_map, d_S1_feature_map_padded, batch, 64, 16, 1, ceil(16/16), 16);      // channel 64 ; input_size 16, pad_size 1, grid width 1, tile width 16
    conv(d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias, batch, 18, 64, 128, 3, ceil(16/16), 16);    // input_size 18, input channel 64,  output channel 128, filter size 3, tile size 16
    // conv_sm(d_S1_feature_map_padded, d_C2_1_feature_map, d_conv2_1_weight, d_conv2_1_bias, batch, 18, 64, 128, 3, ceil(16/16), 16);    // input_size 18, input channel 64,  output channel 128, filter size 3, tile size 16
    relu(d_C2_1_feature_map, batch, 128, 16, ceil(16/16), 16);                               // channel 128, input size 16, tile size 16
    padding(d_C2_1_feature_map, d_C2_1_feature_map_padded, batch, 128, 16, 1, ceil(16/16), 16);  // channel 128, input_size 16, pad_size 1, grid width 1, tile width 16
    conv(d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias, batch, 18, 128, 128, 3, ceil(16/16), 16); // input_size 18, input channel 128, output channel 128, filter size 3, tile size 16
    // conv_sm(d_C2_1_feature_map_padded, d_C2_2_feature_map, d_conv2_2_weight, d_conv2_2_bias, batch, 18, 128, 128, 3, ceil(16/16), 16); // input_size 18, input channel 128, output channel 128, filter size 3, tile size 16
    relu(d_C2_2_feature_map, batch, 128, 16, ceil(16/16), 16);           // channel 128, input size 16
    pool(d_C2_2_feature_map, d_S2_feature_map , batch, 128, 8, ceil(8/8), 8);   //  input channel 128, output size 8          pooling 후 image size 16 -> 8
    // test_print_sinhyun(d_S2_feature_map, 128*128*8*8);

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
    padding(d_S2_feature_map, d_S2_feature_map_padded, batch, 128, 8, 1, ceil(8/8), 8);      // channel 128 ; input_size 8, pad_size 1, grid width 1, tile width 8
    conv(d_S2_feature_map_padded, d_C3_1_feature_map, d_conv3_1_weight, d_conv3_1_bias, batch, 10, 128, 256, 3, ceil(8/8), 8);    // input_size 10, input channel 128,  output channel 256, filter size 3, tile size 8
    // conv_sm(d_S2_feature_map_padded, d_C3_1_feature_map, d_conv3_1_weight, d_conv3_1_bias, batch, 10, 128, 256, 3, ceil(8/8), 8);    // input_size 10, input channel 128,  output channel 256, filter size 3, tile size 8
    relu(d_C3_1_feature_map, batch, 256, 8, ceil(8/8), 8);                               // channel 256, input size 8, tile size 8
    padding(d_C3_1_feature_map, d_C3_1_feature_map_padded, batch, 256, 8, 1, ceil(8/8), 8);  // channel 256, input_size 8, pad_size 1, grid width 1, tile width 8
    conv(d_C3_1_feature_map_padded, d_C3_2_feature_map, d_conv3_2_weight, d_conv3_2_bias, batch, 10, 256, 256, 3, ceil(8/8), 8); // input_size 10, input channel 256, output channel 256, filter size 3, tile size 8
    // conv_sm(d_C3_1_feature_map_padded, d_C3_2_feature_map, d_conv3_2_weight, d_conv3_2_bias, batch, 10, 256, 256, 3, ceil(8/8), 8); // input_size 10, input channel 256, output channel 256, filter size 3, tile size 8
    relu(d_C3_2_feature_map, batch, 256, 8, ceil(8/8), 8);                               // channel 256, input size 8
    padding(d_C3_2_feature_map, d_C3_2_feature_map_padded, batch, 256, 8, 1, ceil(8/8), 8);  // channel 256, input_size 8, pad_size 1, grid width 1, tile width 8
    conv(d_C3_2_feature_map_padded, d_C3_3_feature_map, d_conv3_3_weight, d_conv3_3_bias, batch, 10, 256, 256, 3, ceil(8/8), 8); // input_size 10, input channel 256, output channel 256, filter size 3, tile size 8
    // conv_sm(d_C3_2_feature_map_padded, d_C3_3_feature_map, d_conv3_3_weight, d_conv3_3_bias, batch, 10, 256, 256, 3, ceil(8/8), 8); // input_size 10, input channel 256, output channel 256, filter size 3, tile size 8
    relu(d_C3_3_feature_map, batch, 256, 8, ceil(8/8), 8);                               // channel 256, input size 8
    pool(d_C3_3_feature_map, d_S3_feature_map , batch, 256, 4, ceil(4/4), 4);   //  input channel 256, output size 4          pooling 후 image size 8 -> 4
    // test_print_sinhyun(d_S3_feature_map, 128*256*4*4);

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
    // std::cout << "conv4_1_in_channel: " << conv4_1_in_channel << std::endl;
    // std::cout << "conv4_1_out_channel: " << conv4_1_out_channel << std::endl;
    padding(d_S3_feature_map, d_S3_feature_map_padded, batch, 256, 4, 1, ceil(4/4), 4);      // channel 256 ; input_size 4, pad_size 1, grid width 1, tile width 4
    conv(d_S3_feature_map_padded, d_C4_1_feature_map, d_conv4_1_weight, d_conv4_1_bias, batch, 6, 256, 512, 3, ceil(4/4), 4);    // input_size 6, input channel 256,  output channel 512, filter size 3, tile size 4
    // conv_sm(d_S3_feature_map_padded, d_C4_1_feature_map, d_conv4_1_weight, d_conv4_1_bias, batch, 6, 256, 512, 3, ceil(4/4), 4);    // input_size 6, input channel 256,  output channel 512, filter size 3, tile size 4
    relu(d_C4_1_feature_map, batch, 512, 4, ceil(4/4), 4);                               // channel 512, input size 4, tile size 4
    padding(d_C4_1_feature_map, d_C4_1_feature_map_padded, batch, 512, 4, 1, ceil(4/4), 4);  // channel 512, input_size 4, pad_size 1, grid width 1, tile width 4
    conv(d_C4_1_feature_map_padded, d_C4_2_feature_map, d_conv4_2_weight, d_conv4_2_bias, batch, 6, 512, 512, 3, ceil(4/4), 4); // input_size 6, input channel 512, output channel 512, filter size 3, tile size 4
    // conv_sm(d_C4_1_feature_map_padded, d_C4_2_feature_map, d_conv4_2_weight, d_conv4_2_bias, batch, 6, 512, 512, 3, ceil(4/4), 4); // input_size 6, input channel 512, output channel 512, filter size 3, tile size 4
    relu(d_C4_2_feature_map, batch, 512, 4, ceil(4/4), 4);                               // channel 512, input size 4
    padding(d_C4_2_feature_map, d_C4_2_feature_map_padded, batch, 512, 4, 1, ceil(4/4), 4);  // channel 512, input_size 4, pad_size 1, grid width 1, tile width 4
    conv(d_C4_2_feature_map_padded, d_C4_3_feature_map, d_conv4_3_weight, d_conv4_3_bias, batch, 6, 512, 512, 3, ceil(4/4), 4); // input_size 6, input channel 512, output channel 512, filter size 3, tile size 4
    // conv_sm(d_C4_2_feature_map_padded, d_C4_3_feature_map, d_conv4_3_weight, d_conv4_3_bias, batch, 6, 512, 512, 3, ceil(4/4), 4); // input_size 6, input channel 512, output channel 512, filter size 3, tile size 4
    relu(d_C4_3_feature_map, batch, 512, 4, ceil(4/4), 4);                               // channel 512, input size 4
    pool(d_C4_3_feature_map, d_S4_feature_map , batch, 512, 2, ceil(2/2), 2);   //  input channel 512, output size 2          pooling 후 image size 8 -> 4
    // test_print_sinhyun(d_S4_feature_map, 128*512*2*2);

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
    // std::cout << "conv5_1_in_channel: " << conv5_1_in_channel << std::endl;
    // std::cout << "conv5_1_out_channel: " << conv5_1_out_channel << std::endl;
    // std::cout << "conv5_1_padding_size: " << conv5_1_padding_size << std::endl;
    // std::cout << "conv5_1_kernel_size: " << conv5_1_kernel_size << std::endl;
    // std::cout << "S5_channel: " << S5_channel << std::endl;
    // std::cout << "S5_size: " << S5_size << std::endl;
    padding(d_S4_feature_map, d_S4_feature_map_padded, batch, 512, 2, 1, ceil(2/2), 2);      // channel 512 ; input_size 2, pad_size 1, grid width 1, tile width 2
    conv(d_S4_feature_map_padded, d_C5_1_feature_map, d_conv5_1_weight, d_conv5_1_bias, batch, 4, 512, 512, 3, ceil(2/2), 2);    // input_size 4, input channel 512,  output channel 512, filter size 3, tile size 2
    relu(d_C5_1_feature_map, batch, 512, 2, ceil(2/2), 2);                               // channel 512, input size 2, tile size 2
    padding(d_C5_1_feature_map, d_C5_1_feature_map_padded, batch, 512, 2, 1, ceil(2/2), 2);  // channel 512, input_size 2, pad_size 1, grid width 1, tile width 2
    conv(d_C5_1_feature_map_padded, d_C5_2_feature_map, d_conv5_2_weight, d_conv5_2_bias, batch, 4, 512, 512, 3, ceil(2/2), 2); // input_size 4, input channel 512, output channel 512, filter size 3, tile size 2
    relu(d_C5_2_feature_map, batch, 512, 2, ceil(2/2), 2);                               // channel 512, input size 2
    padding(d_C5_2_feature_map, d_C5_2_feature_map_padded, batch, 512, 2, 1, ceil(2/2), 2);  // channel 512, input_size 2, pad_size 1, grid width 1, tile width 2
    conv(d_C5_2_feature_map_padded, d_C5_3_feature_map, d_conv5_3_weight, d_conv5_3_bias, batch, 4, 512, 512, 3, ceil(2/2), 2); // input_size 4, input channel 512, output channel 512, filter size 3, tile size 2
    relu(d_C5_3_feature_map, batch, 512, 2, ceil(2/2), 2);                               // channel 512, input size 2
    pool(d_C5_3_feature_map, d_S5_feature_map , batch, 512, 1, ceil(1/1), 1);   //  input channel 512, output size 2          pooling 후 image size 2 -> 1
    // test_print_sinhyun(d_S5_feature_map, 128*512*1*1);

    // std::cout << "fc1_in_channel: " << fc1_in_channel << std::endl;
    fc(d_S5_feature_map, d_output, d_fc1_weight, d_fc1_bias, batch, fc1_in_channel, fc1_out_channel);
    // test_file_write_sinhyun(d_S5_feature_map, batch*512*1*1);
    // relu(d_output, batch, 10, 1, 1, 1);                               // 각 128장 image마다 10개씩 결과가 나옴.
    // test_print_sinhyun(d_output, 128*10);

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
