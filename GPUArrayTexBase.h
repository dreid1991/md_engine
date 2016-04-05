#ifndef GPUARRAYTEXBASE_H
#define GPUARRAYTEXBASE_H

#include "globalDefs.h"

class GPUArrayTexBase {
    protected:
        void copyToDeviceArrayInternal(void *dest,
                                       cudaArray *src,
                                       int numBytes)
        {
            //! \todo Make sure this works for copying from 2d arrays
            CUCHECK(cudaMemcpyFromArray(dest, src, 0, 0, numBytes,
                                                    cudaMemcpyDeviceToDevice));
        }

    public:
        virtual void copyToDeviceArray(void *dest){};
        cudaChannelFormatDesc channelDesc;
};
#endif
