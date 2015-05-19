#include "mf.h"
#include <cuda_runtime.h>
#define BLOCK_SIZE 8


__devive__ float median(float * med, int iSize) {
    for (int i = iSize - 1; i > 0; --i) {
        for (int j = 0; j < i; ++j) {
            if (med[j] > med[j+1]) {
                float dTemp = med[j];
                med[j] = med[j+1];
                med[j+1] = dTemp;
            }
        }
    }
    float dMedian = 0.0;
    if ((iSize % 2) == 0) {
        dMedian = (med[iSize/2] + med[(iSize/2) - 1])/2.0;
    } else {
        dMedian = med[iSize/2];
    }
    return dMedian;
}


__global__ void mfCall(int ny, int nx, int hy, int hx, const float* in, float* d_result){
    int nhx = 2*hx+1;
    int nhy = 2*hy+1;
    int edgex = nhx/2;
    int edgey = nhy/2;
    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    float *window = new float[nhy*nhx];
    int k = 0;

    if(x >= nx || y >= ny)
    	return;

    for (int wx = 0; wx < nhx; wx++)
    {
        for(int wy = 0; wy <nhy ; wy++)
        {
        	int xwind = x + wx - edgex;
            int ywind = y + wy - edgey;
            if(xwind >= 0 && xwind <nx && ywind >= 0 && ywind < ny)
            {
                window[k] = in[xwind + nx*ywind];
                k++;
            }    
        }
    }
    d_result[x + nx*y] = median(window,k);
        
    

}

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
    const int DATA_SIZE = ny*nx;
    const int ARRAY_BYTES_DR = DATA_SIZE * sizeof(float);

    //Create GPU pointers
    float * d_data;
    float * d_result;

    //Allocate GPU memory
    cudaMalloc((void**) &d_data, ARRAY_BYTES_DR);
    cudaMalloc((void**) &d_result, ARRAY_BYTES_DR);
    //Copy from host to device
    cudaMemcpy(d_data,in, ARRAY_BYTES_DR, cudaMemcpyHostToDevice);
    const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);  
    const dim3 gridSize(ceil(nx/ (double) BLOCK_SIZE), ceil(ny/(double) BLOCK_SIZE), 1);
    //Kernel call
    mfCall<<<gridSize, blockSize>>>(ny,nx,hy,hx,d_data,d_result);
    //Copy results from host to device      
    cudaMemcpy(out, d_result, ARRAY_BYTES_DR, cudaMemcpyDeviceToHost);
    //free Memory
    cudaFree(d_data);
    cudaFree(d_result);
}
