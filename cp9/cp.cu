#include "cp.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CHECK_CUDA_ERROR(call) do { \
        cudaError_t result_ = (call); \
        if (result_ != cudaSuccess) { \
            fprintf(stderr, #call " failed: %s\n", \
                    cudaGetErrorString(result_)); \
            exit(1); \
        } \
    } while(0)



void normaliseInput(int ny, int nx, float* normalised, const float* data){

    for (int rowj = 0; rowj < ny; rowj++)
    {
        double sumSqRow = 0.0;
        double mean = 0.0;
        //substract mean
        for (int i = rowj*nx; i < rowj*nx + nx; i++)
        {
            mean += (double)data[i];
        }
        mean /= (double)nx;
    
        for (int i = rowj*nx; i < rowj*nx + nx; i++)
        {
            double value = (double) data[i] - mean;
            normalised[i] = value;
            sumSqRow += pow(value,2);
        }
        double value2 = sqrt(sumSqRow);
        for (int i = rowj*nx; i < rowj*nx + nx; i++)
        {
            normalised[i] /= value2;
        }
    }
}

//calculates correlation of two rows given a normalised matrix
__global__ void correlateCall(int ny, int nx, float* normalised, float * d_result, const int BLOCK_SIZE){
    float res = 0.0;
    int i = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int j = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    __shared__ float blockMem
    if(j <= i && i < ny)
    {
    for(int k = 0; k < nx ; k++){
        res += normalised[k + i*nx] * normalised[k + j*nx];
    }
    d_result[i + j*ny] = res;
    }
}



void correlate(int ny, int nx, const float* data, float* result) {
    const int DATA_SIZE = ny*nx;
    const int RESULT_SIZE = ny*ny;
    const int BLOCK_SIZE = 8;
    const int ARRAY_BYTES_RESULT = RESULT_SIZE * sizeof(float);
    const int ARRAY_BYTES_DATA = DATA_SIZE * sizeof(float);
    //Create GPU pointers
    float * d_data;
    float * d_result;

    float *normalised = new float[ny*nx];
    normaliseInput(ny,nx,normalised,data);

    //Allocate GPU memory
    cudaMalloc((void**) &d_data, ARRAY_BYTES_DATA);
    cudaMalloc((void**) &d_result, ARRAY_BYTES_RESULT);
    //Copy from host to device
    cudaMemcpy(d_data,normalised, ARRAY_BYTES_DATA, cudaMemcpyHostToDevice);
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);  
    const dim3 gridSize(ceil(ny/ (double) BLOCK_SIZE), ceil(ny/(double) BLOCK_SIZE), 1);
    //Kernel call
    correlateCall<<<gridSize, blockSize>>>(ny,nx,d_data,d_result,BLOCK_SIZE);
    //Copy results from host to device      
    cudaMemcpy(result, d_result, ARRAY_BYTES_RESULT, cudaMemcpyDeviceToHost);
    //free Memory
    delete [] normalised;
    cudaFree(d_data);
    cudaFree(d_result);
}
