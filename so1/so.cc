#include "so.h"
#include <algorithm>
#include <omp.h>
using namespace std;


void psort(int n, data_t* data) {
    
    int numBlocks = omp_get_max_threads();
    //int numThreads = 0;
    int blockSize = n/numBlocks;
    bool adeqThreads = 1;
    
    //checks that the threads are a power of two
    
        #pragma omp parallel for 
        for (int i = 0; i < numBlocks; ++i)
        {
            if(i == numBlocks -1){
                sort(data + i * blockSize,data + n);
            }else{
                sort(data + i * blockSize,data + (i + 1) * blockSize);
            }
            int numThreads = omp_get_num_threads();
            //checks if the number of threads is a power of two
            if(omp_get_thread_num() == 0){
            adeqThreads = !(numThreads == 0) && !(numThreads & (numThreads - 1));
            }
        }
     
    if (adeqThreads)
    {

    while (numBlocks/2 >= 1){
        int cont = numBlocks/2;
        #pragma omp parallel for 
        for (int j = 0; j < cont; j ++) {           
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
    }else
    {
        sort(data + 0,data + n);
    }
}

