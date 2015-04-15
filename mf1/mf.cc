#include "mf.h"
#include <iostream>
#include <queue>
using namespace std;

void mf(int ny, int nx, int hy, int hx, const float* in, float* out)
{
    // FIXME
	int edgex = hx/2;
	int edgey = hy/2;
	int limit = (ny*nx) - (ny*nx);
	int xwind = 0;
	int ywind = 0;
    
    for (int y = edgey; i < ny-edgey; i++)
    {
    	for (int x = edgex; y < nx-edgex; y++)
    	{
    		for (int wx = 0; wx < hx; wx++)
    		{
    			for(int wy = 0; wy <hy ; wy++)
    			{
    				xwind = x + wx - edgex;
    				ywind = y + wy - edgey;
    				window.push(in[xwind + nx*ywind]);
    			}
    		}
    		out[x + nx*y] = median(window);//median 
    	}
    }

void clear(priority_queue<float, std::vector<float>, std::greater<float>> &q)
{
    priority_queue<float, std::vector<float>, std::greater<float> >  empty;
    swap( q, empty );
}

float median(priority_queue<float, std::vector<float>, std::greater<float>> &q){
    float res = 0.0;
    float aux = 0.0;
    long initSize = q.size();
    while (q.size() > initSize/2 +1) {
        q.pop();
    }
    if (initSize % 2 == 0) {
        aux = q.top();
        q.pop();
        res = (aux + q.top())/2.0;
    }else{
        //q.pop();
        res = q.top();
    }
    clear(q);
    
    return res;
}

    //for (int i = edgex; i < ny * nx; ++i) {
    //    out[i] = in[i];
    //}
}
