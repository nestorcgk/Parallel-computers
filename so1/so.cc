#include "so.h"
#include <algorithm>
#include <omp.h>
using namespace std;


void psort(int n, data_t* data) {
    
    int numBlocks = omp_get_max_threads();
    int blockSize = n/numBlocks;
    
    #pragma omp parallel for 
    for (int i = 0; i < numBlocks; ++i)
    {
        if(i == numBlocks -1){
            sort(data + i * blockSize,data + n);
        }else{
            sort(data + i * blockSize,data + (i + 1) * blockSize);
        }
    }

    while (numBlocks/2 >= 1){
        #pragma omp parallel for 
        for (int j = 0; j < numBlocks / 2; j ++) {           
            int first = 2 * j * blockSize;
            int last = first + blockSize;
            int lastIndex = last + blockSize;
            if(j == numBlocks/2 -1)
            {
                inplace_merge(data + first,data + last,data + n);
            }else{
                inplace_merge(data + first,data + last,data + lastIndex);
            }
        }
        numBlocks /= 2;
        blockSize *= 2;
    }
}

