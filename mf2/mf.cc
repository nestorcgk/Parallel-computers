#include "mf.h"

float median(vector<float> med)
{
    float median = 0.0;
    nth_element(med.begin(), med.begin() + med.size()/2, med.end());

    if(med.size() % 2 == 0)
    {
        median = med[med.size()/2];
        nth_element(med.begin(), med.begin() + med.size()/2 -1, med.end());
        median = (median + med[(med.size()/2) -1])/2.0;

    }else
    {
        median = med[med.size()/2];
    }
    return median;
}

void mf(int ny, int nx, int hy, int hx, const float* in, float* out)
{
    int nhx = 2*hx+1;
    int nhy = 2*hy+1;
    int edgex = nhx/2;
    int edgey = nhy/2;
    int xwind = 0;
    int ywind = 0;
    vector<float> window;
    window.reserve(hx*nhy);
    #pragma omp parallel for  
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {
            vector<float> window;
            for (int wx = 0; wx < nhx; wx++)
            {
                for(int wy = 0; wy <nhy ; wy++)
                {
                    xwind = x + wx - edgex;
                    ywind = y + wy - edgey;
                    if(xwind >= 0 && xwind <nx && ywind >= 0 && ywind < ny)
                    {
                        window.push_back(in[xwind + nx*ywind]);
                    }
                    
                }
            }
            out[x + nx*y] = median(window);
        }
    }
}