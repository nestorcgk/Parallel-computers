#include "cp.h"
void normaliseInput(int ny, int nx, double* normalised, const float* data){
    #pragma omp parallel for
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
double matProduct(int ny, int nx,int vec1,int vec2, double* normalised){
    double res = 0.0;
    //matrix[x + y*nx]
    for (int i = 0; i < nx; i++) {
        res += normalised[i + vec1*nx]*normalised[i + vec2*nx];
    }
    return res;
}

//for all i and j with 0 <= j <= i < ny
void correlate(int ny, int nx, const float* data, float* result){
    double *normalised = new double[ny*nx];
    normaliseInput(ny,nx,normalised,data);
    
    #pragma omp parallel for
    for (int i = 0; i < ny; i++)//(int i = 0; i < ny; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            result[i+j*ny] = matProduct(ny, nx, i, j, normalised);
        }
    }
}