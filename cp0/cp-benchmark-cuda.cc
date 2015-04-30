#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "error.h"
#include "timer.h"
#include "cp.h"

static void benchmark(int ny, int nx) {
    std::mt19937 rng;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float* data = NULL;
    float* result = NULL;
    cudaMallocHost((void**)&data, nx * ny * sizeof(float));
    cudaMallocHost((void**)&result, ny * ny * sizeof(float));
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            float v = u(rng);
            data[x + nx * y] = v;
        }
    }
    std::cout << "cp\t" << ny << "\t" << nx << "\t" << std::flush;
    { Timer t; correlate(ny, nx, data, result); }
    std::cout << std::endl;
    cudaFreeHost(data);
    cudaFreeHost(result);
}

int main(int argc, const char** argv) {
    if (argc != 3) {
        error("usage: cp-benchmark Y X");
    }
    int ny = std::stoi(argv[1]);
    int nx = std::stoi(argv[2]);
    benchmark(ny, nx);
}
