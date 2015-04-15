#include "mf.h"



void mf(int ny, int nx, int hy, int hx, const float* in, float* out)
{
    // FIXME
	int edgex = hx/2;
	int edgey = hy/2;
	int xwind = 0;
	int ywind = 0;
	vector<float> window;
    
    for (int y = 0; y < ny; y++)
    {
    	for (int x = 0; x < nx; x++)
    	{
            vector<float> window;  
    		for (int wx = 0; wx < hx; wx++)
    		{
    			for(int wy = 0; wy <hy ; wy++)
    			{
    				xwind = x + wx - edgex;
    				ywind = y + wy - edgey;
                    if(xwind>= 0 && xwind <nx && ywind>= 0 && ywind <ny)
                    {
                        window.push_back(in[xwind + nx*ywind]);
                    }
    				
    			}
    		}
            nth_element(window.begin(), window.begin() + window.size()/2, window.end());
            if(window.size() % 2 == 0)
            {
                out[x + nx*y] = (window[window.size()/2] + window[window.size()/2+1])/2.0;
                
            }else
            {
                out[x + nx*y] = window[window.size()/2];
            }
    		
    		
    	}
    }
}
