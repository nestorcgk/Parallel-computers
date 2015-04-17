#include "mf.h"

float median(float * med,int k)
{
    float median = 0.0;
    //nth_element(med.begin(), med.begin() + med.size()/2, med.end());
    nth_element(med + 0, med + k/2, med +k);

    if(k % 2 == 0)
    {
        median = med[k/2];
        //nth_element(med.begin(), med.begin() + med.size()/2 -1, med.end());
        nth_element(med + 0, med + k/2 -1, med +k);
        median = (median + med[(k/2) -1])/2.0;

    }else
    {
        median = med[k/2];
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
    
    #pragma omp parallel for  
    for (int y = 0; y < ny; y++)
    {
        //vector<float> window;
        //window.reserve(hx*nhy);
        float *window = new float[nhy*nhx];

        for (int x = 0; x < nx; x++)
        {
            int k = 0;
            //window.clear();
            for (int wx = 0; wx < nhx; wx++)
            {
                for(int wy = 0; wy <nhy ; wy++)
                {
                    xwind = x + wx - edgex;
                    ywind = y + wy - edgey;
                    if(xwind >= 0 && xwind <nx && ywind >= 0 && ywind < ny)
                    {
                        //window.push_back(in[xwind + nx*ywind]);
                        //window.insert(window.begin()+k,in[xwind + nx*ywind]);
                        window[k] = in[xwind + nx*ywind];
                        k++;
                    }
                    
                }
            }
            out[x + nx*y] = median(window,k);
        }
    }
}