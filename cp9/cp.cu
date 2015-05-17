#include "cp.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t result_ = (call); \
    if (result_ != cudaSuccess) { \
        fprintf(stderr, #call " failed: %s\n", \
                cudaGetErrorString(result_)); \
        exit(1); \
    } \
} while(0)

#define block_size 29 //25 or 29
#define thread_rows 7 //7 or 6
#define debug 0


__global__ void my_kernel(int size_x, int size_y, const double* input, double* output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= size_x || y >= size_y)
        return;
    output[x + size_x * y] = 2.0 * input[x + size_x * y];
}


__global__ void dot_product(int size_x, int size_y, int o_size_y, const float* input, float* output)
{

    int large_square_size = block_size * thread_rows;

    int x = threadIdx.x*thread_rows + blockIdx.x * large_square_size;
    int y = threadIdx.y*thread_rows + blockIdx.y * large_square_size;
    if (blockIdx.y > blockIdx.x)
        return;



    float buffer[thread_rows][thread_rows];
    memset(buffer, 0, thread_rows*thread_rows*sizeof(float));
    //Allocate shared memory
    __shared__ float block1[block_size][block_size * thread_rows];
    __shared__ float block2[block_size][block_size * thread_rows];

    int block1_info_y = threadIdx.y*thread_rows + blockIdx.y*large_square_size;
    int block2_info_y = threadIdx.y*thread_rows + blockIdx.x*large_square_size;
    //loop over blocks of input matrix
    for (int b = 0; b < (size_x + block_size - 1)/block_size; ++b)
    {

        //One thread loads two value of each of the input matrix.
        int block1_info_x = threadIdx.x + b*block_size;        
        int block2_info_x = threadIdx.x + b*block_size;
        
        for (int row = 0; row < thread_rows; ++row)
            block1[threadIdx.x][threadIdx.y*thread_rows + row] = input[block1_info_x*size_y + block1_info_y + row];

        for (int row = 0; row < thread_rows; ++row)
            block2[threadIdx.x][threadIdx.y*thread_rows + row] = input[block2_info_x*size_y + block2_info_y + row];
        
        __syncthreads();

        if (!(x > o_size_y || y > o_size_y))
        {
    
        for (int i=0; i < block_size; ++i)
        {     
            for (int i_row = 0; i_row < thread_rows; ++i_row)
            {
                for (int j_row = 0; j_row < thread_rows; ++j_row)
                {

                    buffer[i_row][j_row] += block1[i][threadIdx.y*thread_rows + i_row] * block2[i][threadIdx.x*thread_rows + j_row];
                }
            }
        }
        }
    __syncthreads();

    }

    for (int i_row = 0; i_row < thread_rows; ++i_row)
    {
        for (int j_row = 0; j_row < thread_rows; ++j_row)
        {
            if (x + j_row< o_size_y && y + i_row < o_size_y)
            {
                output[x + j_row + (y+i_row)*o_size_y] = buffer[i_row][j_row];
            }
        }
    }

}


void correlate(int ny, int nx, const float* data, float* result) {

    int x_padding = (block_size - nx % block_size) % block_size;
    int y_padding = (block_size*thread_rows - ny % (block_size*thread_rows)) % (block_size*thread_rows);
    float *normalized = new float[(ny+y_padding)*(nx+x_padding)];

    for (int i=0; i<ny; ++i)
    {
        double mean = 0;
        double sum = 0;
        double product = 0;
        double var = 0;
        for (int k=0; k<nx; ++k)
        {
            double x = data[i*nx + k];
            sum += x;
            product += x*x;
        }
        mean = sum/nx;
        var = product -nx*mean*mean;

        for (int k=0; k<nx; ++k)
        {
            normalized[(ny+y_padding)*k + i] = (data[i*nx + k] - mean)/sqrt(var);
        }
        for (int k=nx; k<nx+x_padding; ++k)
        {
            normalized[(ny+y_padding)*k + i] = 0;
        }

    }

    for (int i=ny; i<ny+y_padding; ++i)
    {
        for (int k=0; k<nx+x_padding; ++k)
        {
            normalized[(ny+y_padding)*k + i] = 0;
        }
    }

    float *dev_input;
    float *dev_output;

    //allocates memory
    CHECK_CUDA_ERROR(cudaMalloc((void **) &dev_input, (nx+x_padding)*(ny+y_padding) * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &dev_output, ny*ny * sizeof(float)));
    //copy array to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(dev_input, normalized, (nx+x_padding)*(ny+y_padding)*sizeof(float), cudaMemcpyHostToDevice));

    dim3 szBlock(block_size, block_size);
    dim3 szGrid((ny + szBlock.x*thread_rows - 1) / (szBlock.x*thread_rows), (ny + szBlock.y*thread_rows - 1) / (szBlock.y*thread_rows));
    dot_product <<< szGrid, szBlock >>> (nx + x_padding, ny + y_padding, ny, dev_input, dev_output);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(result, dev_output, ny*ny*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(dev_input));
    CHECK_CUDA_ERROR(cudaFree(dev_output));
    delete [] normalized;

}


