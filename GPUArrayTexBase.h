#pragma once
#ifndef GPUARRAYTEXBASE_H
#define GPUARRAYTEXBASE_H

#include "globalDefs.h"

/*! \brief Base class for GPU Texture Arrays
 *
 * This class is a base class to work with Texture arrays.
 */
class GPUArrayTexBase {
protected:
    /*! \brief Default constructor */
    GPUArrayTexBase() {}

    /*! \brief Constructor
     *
     * \param cDesc Channel descriptor for the texture device
     */
    explicit GPUArrayTexBase(cudaChannelFormatDesc cDesc)
                : channelDesc(cDesc) {}

public:
    /*! \brief Copy data from GPU device to texture
     *
     * \param dest Destination: Pointer to the GPU memory
     */
    virtual void copyToDeviceArray(void *dest) = 0;

protected:
    /*! \brief Copy data from host device to the GPU texture
     *
     * \param dest Destination: Pointer to the GPU memory
     * \param src Source: Pointer to cudaArray
     * \param numBytes Number of bytes to be copied
     *
     * This function copies data from the GPU device onto a GPU texture
     * array.
     */
    void copyToDeviceArrayInternal(void *dest,
                                   cudaArray *src,
                                   int numBytes)
    {
        //! \todo Make sure this works for copying from 2d arrays
        CUCHECK(cudaMemcpyFromArray(dest, src, 0, 0, numBytes,
                                                cudaMemcpyDeviceToDevice));
    }

protected:
    cudaChannelFormatDesc channelDesc; //!< Descriptor for the texture
};
#endif
