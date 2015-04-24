#include "cp.h"
#include "vector.h"
typedef double double4_t __attribute__
    ((__vector_size__ (4*sizeof(double))));

//creates a vector matrix in which each element is double4_t with row  mean 0 and  sum of squares = 1
void normaliseInputVec(int ny, int nx, double4_t* normalised, const float* data,int xdim){
    #pragma omp parallel for schedule(static,1)
    for (int rowj = 0; rowj < ny; rowj++)
    {
        double sumSqRow = 0.0;
        double mean = 0.0;
        //substract mean
        for (int i = rowj * nx; i < rowj*nx + nx; i++)
        {
            mean += (double)data[i];
        }
        mean /= (double)nx;

    
        for (int col = 0; col < nx; col++)
        {
            double value = (double) data[rowj * nx + col] - mean;
            normalised[rowj * xdim + col/4][col % 4] = value;
            sumSqRow += pow(value,2);
        }

        double value2 = sqrt(sumSqRow);
        for (int col = 0; col < nx; col++)
        {
            normalised[rowj * xdim + col/4][col % 4] /= value2;
        }
    }
}

//Adds the accumulator that contains 4 double4_t vectors
double sumVecAcc(double4_t vec[]){
    double4_t res = {0.0};
    double sum = 0.0;

    for (int i = 0; i < 4; i++)
    {
        res += vec[i];
    }

    for (int j = 0; j < 4; j++)
    {
        sum += res[j];
    }
    return sum;
}


//computes the correlation coefficient for two vectors (rows)
double matProductVec(int ny, int nx,int vec1,int vec2, double4_t* normalised,int xdim){
    double4_t temp[4] = {0.0};
    int accParam = 0;

    for (int i=0; i < xdim; i += 4)
    {
        if((xdim - i) > 4)
        {
            accParam = 4;
        }else{
            accParam = (xdim - i);
        }
        for (int k = 0; k < accParam; k++)
        {
            temp[k] += normalised[i + k + vec1*xdim] * normalised[i + k + vec2*xdim];
        }

    }

    return sumVecAcc(temp);
}

//creates all the correlated pais
void correlate(int ny, int nx, const float* data, float* result){
    int xdim = (int) ceil(nx / (double) 4.0);
    double4_t *normalised = double4_alloc(ny * xdim);
    normaliseInputVec(ny ,nx, normalised ,data, xdim);

    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            result[i + j*ny] = matProductVec(ny, nx, i, j, normalised, xdim);
        }
    }
    free(normalised);
}