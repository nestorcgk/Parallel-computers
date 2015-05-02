#include "so.h"
#include <algorithm>
#include <omp.h>
using namespace std;


void psort(int n, data_t* data) {
    
    int numBlocks = omp_get_max_threads();
    int blockSize = n/numBlocks;
    
    //#pragma omp parallel num_threads(numBlocks)
    //{
    #pragma omp parallel for
    for (int i = 0; i < numBlocks; ++i)
    {
        sort(data + i * blockSize,data + (i+1)*blockSize);
    }

	//}
    
    while (numBlocks/2 >= 1){
        //#pragma omp parallel num_threads(numBlocks/2)
        //{
        #pragma omp parallel for
        for (int j = 0; j < numBlocks/2; j ++) {
            data_t *temp = new data_t[blockSize*2];            
            int first = 2 * j * blockSize;
            int last = first + blockSize;
            int lastIndex = min(last + blockSize,n);
            merge(data + first,data + first + blockSize,data + last,data + lastIndex,temp);
            copy(temp,temp+(blockSize*2),data + first);
        }
       	//} 
        numBlocks /= 2;
        blockSize *= 2;
    }
}

