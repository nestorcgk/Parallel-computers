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



void normaliseInput(int ny, int nx, double* normalised, const float* data){

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
__global__ void correlateCall(int ny, int nx, double* normalised, float * d_result, const int BLOCK_SIZE){
    double res = 0.0;
    int i = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int j = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    __shared__ float blockMemi[nx*BLOCK_SIZE];
    __shared__ float blockMemj[nx*BLOCK_SIZE];
    
    if(thread.Idy == 0)
    {
        for (int idx = 0; idx < nx; ++idx)
        {
            blockMemi[nx*threadIdx.x + threadIdx.x + idx] = normalised[idx + i*nx];
        }
    }
    if(thread.idx == 0){
        for (int idy = 0; idx < nx; ++idy)
        {
            blockMemj[nx*threadIdx.y + threadIdx.y + idy] = normalised[idy + j*nx];
        }
    }
    __syncthreads(); //ensure completed writed to shared memory

    if(j <= i && i < ny)
    {
    for(int k = 0; k < nx ; k++){
        res += blockMemi[nx*threadIdx.x + threadIdx.x + k] * blockMemj[nx*threadIdx.y + threadIdx.y + k];
    }
    d_result[i + j*ny] = res;
    }
}



void correlate(int ny, int nx, const float* data, float* result) {
    const int DATA_SIZE = ny*nx;
	const int RESULT_SIZE = ny*ny;
    const int BLOCK_SIZE = 8;
	const int ARRAY_BYTES_FLOAT = RESULT_SIZE * sizeof(float);
	const int ARRAY_BYTES_DOUBLE = DATA_SIZE * sizeof(double);
	//Create GPU pointers
    double * d_data;
	float * d_result;

	double *normalised = new double[ny*nx];
    normaliseInput(ny,nx,normalised,data);

    //Allocate GPU memory
	cudaMalloc((void**) &d_data, ARRAY_BYTES_DOUBLE);
	cudaMalloc((void**) &d_result, ARRAY_BYTES_FLOAT);
    //Copy from host to device
	cudaMemcpy(d_data,normalised, ARRAY_BYTES_DOUBLE, cudaMemcpyHostToDevice);
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);  
  	const dim3 gridSize(ceil(ny/ (double) BLOCK_SIZE), ceil(ny/(double) BLOCK_SIZE), 1);
    //Kernel call
    correlateCall<<<gridSize, blockSize>>>(ny,nx,d_data,d_result,BLOCK_SIZE);
    //Copy results from host to device 		
	cudaMemcpy(result, d_result, ARRAY_BYTES_FLOAT, cudaMemcpyDeviceToHost);
	//free Memory
    delete [] normalised;
    cudaFree(d_data);
	cudaFree(d_result);
}
