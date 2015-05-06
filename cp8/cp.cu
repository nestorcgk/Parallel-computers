#include "cp.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


__global__ void cube(float * d_out, float * d_in){
	int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f*f;
}

void normaliseInput(int ny, int nx, double* normalised, const float* data){
    //#pragma omp parallel for
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
__global__ void matProduct(int ny, int nx,int vec1,int vec2, double* normalised){
    double res = 0.0;
    //matrix[x + y*nx]
    
    for (int i = 0; i < nx; i++) {
        res += normalised[i + vec1*nx]*normalised[i + vec2*nx];
    }
    result[vec1+vec2*ny] = res;
}



void correlate(int ny, int nx, const float* data, float* result) {
    
    //for (int i = 0; i < ny * ny; ++i) {
    //    result[i] = 0.0f;
    //}
    const int ARRAY_SIZE = ny*nx;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// declare GPU memory pointers
	float * d_data;
	float * d_result;

	//Normalize input
	double *normalised = new double[ny*nx];
    normaliseInput(ny,nx,normalised,data);

	// allocate GPU memory
	cudaMalloc((void**) &d_data, ARRAY_BYTES);
	cudaMalloc((void**) &d_result, ARRAY_BYTES);

	// transfer the array to the GPU
	cudaMemcpy(d_data, data, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	const dim3 blockSize(128, 1, 1);  
  	const dim3 gridSize(ceil(nx/128.0), 1, 1);




    #pragma omp parallel for
    for (int i = 0; i < ny; i++)//(int i = 0; i < ny; i++)
    {
        for (int j = 0; j <= i; j++)
        {
        	matProduct<<<gridSize, blockSize>>>(d_result, d_data);
            //result[i+j*ny] = matProduct(ny, nx, i, j, normalised);

        }
    }
	

	// copy back the result array to the CPU
	cudaMemcpy(result, d_result, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaFree(d_data);
	cudaFree(d_result);
}
