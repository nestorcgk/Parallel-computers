#include "mf.h"





void clear(vector<float> &q)
{
    vector<float>  empty;
    swap( q, empty );
}

void mf(int ny, int nx, int hy, int hx, const float* in, float* out)
{
	int edgex = hx/2;
	int edgey = hy/2;
	int xwind = 0;
	int ywind = 0;
	
    #pragma omp parallel for  
    for (int y = edgey; y < ny-edgey; y++)
    {
    	for (int x = edgex; x < nx-edgex; x++)
    	{ 
            vector<float> window;  
    		for (int wx = 0; wx < hx; wx++)
    		{
    			for(int wy = 0; wy <hy ; wy++)
    			{
    				xwind = x + wx - edgex;
    				ywind = y + wy - edgey;
    				window.push_back(in[xwind + nx*ywind]);
    			}
    		}
    		nth_element(window.begin(), window.begin() + window.size()/2, window.end());
    		out[x + nx*y] = window[window.size()/2];//median 
            //#pragma omp critical
    		//clear(window);
    	}
    }





    //for (int i = edgex; i < ny * nx; ++i) {
    //    out[i] = in[i];
    //}
}
