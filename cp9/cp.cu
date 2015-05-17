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

#define BLOCK_SIZE 27
#define THREAD_ROWS 6 
#define debug 0


__global__ void correlate_call(int size_x, int size_y, int ny, const float* input, float* output)
{

    int large_square_size = BLOCK_SIZE * THREAD_ROWS;

    int x = threadIdx.x*THREAD_ROWS + blockIdx.x * large_square_size;
    int y = threadIdx.y*THREAD_ROWS + blockIdx.y * large_square_size;
    if (blockIdx.y > blockIdx.x)
        return;



    float buffer[THREAD_ROWS][THREAD_ROWS] = {0.0};
    //Allocate shared memory
    __shared__ float block1[BLOCK_SIZE][BLOCK_SIZE * THREAD_ROWS];
    __shared__ float block2[BLOCK_SIZE][BLOCK_SIZE * THREAD_ROWS];

    int block1_info_y = threadIdx.y*THREAD_ROWS + blockIdx.y*large_square_size;
    int block2_info_y = threadIdx.y*THREAD_ROWS + blockIdx.x*large_square_size;
    //loop over blocks of input matrix
    for (int b = 0; b < (size_x + BLOCK_SIZE - 1)/BLOCK_SIZE; ++b)
    {

        //One thread loads two value of each of the input matrix.
        int block1_info_x = threadIdx.x + b*BLOCK_SIZE;        
        int block2_info_x = threadIdx.x + b*BLOCK_SIZE;
        
        for (int row = 0; row < THREAD_ROWS; ++row)
            block1[threadIdx.x][threadIdx.y*THREAD_ROWS + row] = input[block1_info_x*size_y + block1_info_y + row];

        for (int row = 0; row < THREAD_ROWS; ++row)
            block2[threadIdx.x][threadIdx.y*THREAD_ROWS + row] = input[block2_info_x*size_y + block2_info_y + row];
        
        __syncthreads();

        if (!(x > ny || y > ny))
        {
    
        for (int i=0; i < BLOCK_SIZE; ++i)
        {     
            for (int veci_r = 0; veci_r < THREAD_ROWS; veci_r++)
            {
                for (int vecj_r = 0; vecj_r < THREAD_ROWS; vecj_r++)
                {

                    buffer[veci_r][vecj_r] += block1[i][threadIdx.y*THREAD_ROWS + veci_r] * block2[i][threadIdx.x*THREAD_ROWS + vecj_r];
                }
            }
        }
        }
    __syncthreads();

    }

    for (int veci_r = 0; veci_r < THREAD_ROWS; veci_r++)
    {
        for (int vecj_r = 0; vecj_r < THREAD_ROWS; vecj_r++)
        {
            if (x + vecj_r< ny && y + veci_r < ny)
            {
                output[x + vecj_r + (y+veci_r)*ny] = buffer[veci_r][vecj_r];
            }
        }
    }

}

void normaliseInput(int ny,int nx, float* normalised,const float* data,int x_se, int y_se){
    for (int i = 0; i < ny; i++)
    {
        double mean = 0;
        double sum = 0;
        double sumSq = 0;
        double var = 0;
        for (int k = 0; k < nx; k++)
        {
            double x = data[k + i*nx];
            sum += x;
            sumSq += x*x;
        }
        mean = sum/nx;
        var = sumSq -nx*mean*mean;
        
        for (int k = 0; k < nx; k++)
        {
            normalised[i + (ny+y_se)*k] = (data[k + i*nx] - mean)/sqrt(var);
        }
        for (int k = nx; k < nx+x_se; k++)
        {
            normalised[i + (ny+y_se)*k] = 0;
        }

    }

    for (int i = ny; i < ny + y_se; i++)
    {
        for (int k = 0; k < nx+x_se; k++)
        {
            normalised[i + (ny+y_se)*k] = 0;
        }
    }
}

void correlate(int ny, int nx, const float* data, float* result) {
   //required blocks 
    int x_se = (BLOCK_SIZE - nx % BLOCK_SIZE) % BLOCK_SIZE;
    int y_se = (BLOCK_SIZE*THREAD_ROWS - ny % (BLOCK_SIZE*THREAD_ROWS)) % (BLOCK_SIZE*THREAD_ROWS);
    const int ARRAY_BYTES_FLOAT_IN = (nx+x_se)*(ny+y_se) * sizeof(float);
    const int ARRAY_BYTES_FLOAT_OUT = ny*ny * sizeof(float);
    float *normalised = new float[(ny + y_se)*(nx + x_se)];
    normaliseInput(ny,nx,normalised,data,x_se,y_se);
    float *d_input;
    float *d_output;

    //Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_input,ARRAY_BYTES_FLOAT_IN));
    CHECK_CUDA_ERROR(cudaMalloc((void **) &d_output,ARRAY_BYTES_FLOAT_OUT));
    //Copy from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, normalised, ARRAY_BYTES_FLOAT_IN, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((ny + blockSize.x*THREAD_ROWS - 1) / (blockSize.x*THREAD_ROWS), (ny + blockSize.y*THREAD_ROWS - 1) / (blockSize.y*THREAD_ROWS));
    //Execute Kernel
    correlate_call <<< gridSize, blockSize >>> (nx + x_se, ny + y_se, ny, d_input, d_output);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_output, ARRAY_BYTES_FLOAT_OUT, cudaMemcpyDeviceToHost));
    //Free memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    delete [] normalised;

}


