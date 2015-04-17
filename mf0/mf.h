#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <omp.h>
#ifndef MF_H
#define MF_H
using namespace std;

// nx, ny: image dimensions, width x pixels and height y pixels.
// hx, hy: window radius in x and y directions.
// in: input image.
// out: output image.
//
// Pixel (x,y) for 0 <= x < nx and 0 <= y < ny is located at
// in[x + y*nx] and out[x + y*nx].

void mf(int ny, int nx, int hy, int hx, const float* in, float* out);
float median(vector<float> med);

#endif