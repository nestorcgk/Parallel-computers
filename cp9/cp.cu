#include "cp.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#define BLOCK_SIZE 32
#define ROWS_PER_THREAD 6 
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t result_ = (call); \
    if (result_ != cudaSuccess) { \
        fprintf(stderr, #call " failed: %s\n", \
                cudaGetErrorString(result_)); \
        exit(1); \
    } \
} while(0)




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

__global__ void correlate_call(int size_x, int size_y, int ny, const float* input, float* output)
{
    if(blockIdx.y > blockIdx.x)
	return;
    int s_block = BLOCK_SIZE * ROWS_PER_THREAD;
    int x = threadIdx.x*ROWS_PER_THREAD + blockIdx.x * s_block;
    int y = threadIdx.y*ROWS_PER_THREAD + blockIdx.y * s_block;
    float temp[ROWS_PER_THREAD][ROWS_PER_THREAD] = {0.0};

    //Allocate shared memory SM
    __shared__ float chunk_1[BLOCK_SIZE][BLOCK_SIZE * ROWS_PER_THREAD];
    __shared__ float chunk_2[BLOCK_SIZE][BLOCK_SIZE * ROWS_PER_THREAD];

    int chunk_1_y = threadIdx.y*ROWS_PER_THREAD + blockIdx.y*s_block;
    int chunk_2_y = threadIdx.y*ROWS_PER_THREAD + blockIdx.x*s_block;
    int cont = (size_x + BLOCK_SIZE - 1)/BLOCK_SIZE;
    //Iterates over blocks
    for (int blockIdx = 0; blockIdx < cont; blockIdx++)
    {
        int chunk_1_x = threadIdx.x + blockIdx*BLOCK_SIZE;        
        int chunk_2_x = threadIdx.x + blockIdx*BLOCK_SIZE;
        //Copy data to shared memory
        for (int row = 0; row < ROWS_PER_THREAD; ++row)
            chunk_1[threadIdx.x][threadIdx.y*ROWS_PER_THREAD + row] = input[chunk_1_x*size_y + chunk_1_y + row];

        for (int row = 0; row < ROWS_PER_THREAD; ++row)
            chunk_2[threadIdx.x][threadIdx.y*ROWS_PER_THREAD + row] = input[chunk_2_x*size_y + chunk_2_y + row];
        //ensure completed wirte to share memory 
        __syncthreads();

        if (y <= ny || x <= ny)
        {
    
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {     
            for (int veci_r = 0; veci_r < ROWS_PER_THREAD; veci_r++)
            {
                for (int vecj_r = 0; vecj_r < ROWS_PER_THREAD; vecj_r++)
                {

                    temp[veci_r][vecj_r] += chunk_1[i][threadIdx.y*ROWS_PER_THREAD + veci_r] * chunk_2[i][threadIdx.x*ROWS_PER_THREAD + vecj_r];
                }
            }
        }
        }
    __syncthreads();

    }

    for (int veci_r = 0; veci_r < ROWS_PER_THREAD; veci_r++)
    {
        for (int vecj_r = 0; vecj_r < ROWS_PER_THREAD; vecj_r++)
        {
            if (x + vecj_r< ny && y + veci_r < ny)
            {
                output[x + vecj_r + (y+veci_r)*ny] = temp[veci_r][vecj_r];
            }
        }
    }

}



void correlate(int ny, int nx, const float* data, float* result) {
   //required blocks 
    int x_se = (BLOCK_SIZE - nx % BLOCK_SIZE) % BLOCK_SIZE;
    int y_se = (BLOCK_SIZE*ROWS_PER_THREAD - ny % (BLOCK_SIZE*ROWS_PER_THREAD)) % (BLOCK_SIZE*ROWS_PER_THREAD);
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
    dim3 gridSize((ny + blockSize.x*ROWS_PER_THREAD - 1) / (blockSize.x*ROWS_PER_THREAD), (ny + blockSize.y*ROWS_PER_THREAD - 1) / (blockSize.y*ROWS_PER_THREAD));
    //Execute Kernel
    correlate_call <<< gridSize, blockSize >>> (nx + x_se, ny + y_se, ny, d_input, d_output);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_output, ARRAY_BYTES_FLOAT_OUT, cudaMemcpyDeviceToHost));
    //Free memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    delete [] normalised;

}


