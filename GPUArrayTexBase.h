#include "globalDefs.h"

#ifndef GPUARRAYTEXBASE_H
#define GPUARRAYTEXBASE_H

#define SAFEHOSTCAPACITY(n, SIZE) (n <= (PERLINE/sizeof(SIZE)) ? n : (PERLINE/sizeof(SIZE)) * ceil(n/(float)(PERLINE/sizeof(SIZE))))
//inheriting as a template is behaving strangely.  Going to make it deal with voidssss
class GPUArrayTexBase {
    protected:
        void copyToDeviceArrayInternal(void *dest, cudaArray *src, int numBytes) {
            //MAKE SURE THIS WORKS FOR COPYING FROM 2d ARRAYS
            CUCHECK(cudaMemcpyFromArray(dest, src, 0, 0, numBytes, cudaMemcpyDeviceToDevice));
        }

    public:
        virtual void copyToDeviceArray(void *dest){};
        cudaChannelFormatDesc channelDesc;
};
#endif
