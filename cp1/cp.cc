#include <iostream>
#include <math.h>
using namespace std;

//Sum of Squares (x) = N Var (X)
double calculateSSxx(int ny, int nx, int rowj, const float* data){
    double sumSquares = 0.0;
    double mean = 0.0;
    
    for (int i = rowj*nx; i < rowj*nx + nx; i++) {
        sumSquares += pow(data[i],2);
        mean += data[i];
    }
    mean = mean / (double) nx;
    
    return sumSquares - nx * (pow(mean,2)) ;
}


//Sum of squares (x,y) = N * cov (X,Y)
double calculateSSxy(int ny, int nx,int rowi, int rowj, const float* data){
    double sum = 0.0;
    double meanx = 0.0;
    double meany = 0.0;


    int j = rowj*nx;
    for (int i = rowi*nx; i < rowi*nx + nx; i++) {
        meanx += data[i];
        meany += data[j];
        j++;
    }
    meanx = meanx / nx;
    meany = meany / nx;

    j = rowj*nx;
    for (int i = rowi*nx; i < rowi*nx + nx; i++) {
        sum += data[i]*data[j];
        j++;
        
    }
    return sum -(nx*meanx*meany);
}

float calculateCorrelation(int ny, int nx,int rowi, int rowj, const float* data){
    double ssxy = calculateSSxy(ny,nx,rowi,rowj,data);
    double ssxx = calculateSSxx(ny,nx,rowi,data);
    double ssyy = calculateSSxx(ny,nx,rowj,data);
    
    return ssxy / (sqrt(ssxx*ssyy));
}

void correlate(int ny, int nx, const float* data, float* result){
    
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j <= i; j++)
        {
                result[i+j*ny] = calculateCorrelation(ny, nx, i, j, data);
        }
    }
}

