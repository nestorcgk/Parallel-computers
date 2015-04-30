#include <iostream>
#include <vector>
#include <random>
#include "error.h"
#include "timer.h"
#include "cp.h"
#ifdef CUDA_EXERCISE
#include <cuda_runtime.h>
#endif

static void benchmark(int ny, int nx) {
    float* data = NULL;
    float* result = NULL;
#ifdef CUDA_EXERCISE
    cudaMallocHost((void**)&data, nx * ny * sizeof(float));
    cudaMallocHost((void**)&result, ny * ny * sizeof(float));
#else
    data = new float[ny * nx];
    result = new float[ny * ny];
#endif
    std::mt19937 rng;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            float v = u(rng);
            data[x + nx * y] = v;
        }
    }
    std::cout << "cp\t" << ny << "\t" << nx << "\t" << std::flush;
    { Timer t; correlate(ny, nx, data, result); }
    std::cout << std::endl;
#ifdef CUDA_EXERCISE
    cudaFreeHost(data);
    cudaFreeHost(result);
#else
    delete[] data;
    delete[] result;
#endif
}

int main(int argc, const char** argv) {
    if (argc != 3) {
        error("usage: cp-benchmark Y X");
    }
    int ny = std::stoi(argv[1]);
    int nx = std::stoi(argv[2]);
    benchmark(ny, nx);
}
