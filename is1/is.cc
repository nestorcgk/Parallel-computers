#include "is.h"
#include "vector.h"
#include <iostream>
typedef double double3_t __attribute__
    ((__vector_size__ (4*sizeof(double))));

void toVector(int ny, int nx, double4_t* vecData, const float* data){

	for (int i = 0; i < ny*nx*3; i++)
	{

		vecData[i/3][i % 3] = data[i];
		if(i % 3 == 0)
		{
			vecData[i/3][3] = 0.0;
		}	
	}	
}

void preCompSums(int ny, int nx, double4_t* vecData, double4_t* sums){
	for (int y = 0; y < ny; ++y)
	{
		for (int x = 0; x < nx; ++x)
		{
			double4_t sum = {0.0};
			for (int k = 0; k < y; ++k)
			{
				for (int i = 0; i < x; ++i)
				{
					//[x + nx * y]
					sum += vecData[i + k*nx];
					std::cout << "VecData[i] = "<<vecData[i + k*nx][0] <<" "<<vecData[i + k*nx][1]<<" "<<vecData[i + k*nx][2]<< "\n";
				std::cout << "Entra"<<"\n";
				}
				//std::cout << "Entra"<<"\n";
			}
			sums[x + y*nx] = sum;
			std::cout << "x = " << x <<" y = "<<y<<" x+y*nx: "<<x+y*nx<< "\n";
			std::cout << "Sums[i] = "<< sums[x+y*nx][0] <<" "<<sums[x+y*nx][1]<<" "<<sums[x+y*nx][2]<< "\n";
		}
	}
}



Result segment(int ny, int nx, const float* data) {
	std::cout << "ny = " << ny <<" nx "<<nx<<"\n";
	double4_t *vecData = double4_alloc(ny*nx);
	double4_t *sums = double4_alloc(ny*nx);
    toVector(ny,nx,vecData,data);
	preCompSums(ny,nx,vecData,sums);
	std::cout << "Sums[2] = " << sums[2][0] <<" "<<sums[2][1]<<" "<<sums[2][2]<< "\n";
	for (int i = 0; i < 9; ++i)
	{
		std::cout << "data[i] = " << data[i]<< "\n";
	}
    
    Result result { ny/3, nx/3, 2*ny/3, 2*nx/3, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f} };
    return result;
}
