#pragma once
#ifndef GPUARRAY_H
#define GPUARRAY_H

/*! \brief Base class for a GPUArray */
class GPUArray {
    protected:
        /*! \brief Constructor */
        GPUArray() {};

    public:
        /*! \brief Destructor */
        virtual ~GPUArray() {};

        /*! \brief Send data from host to GPU device */
        virtual void dataToDevice() = 0;

        /*! \brief Send data from GPU device to host */
        virtual void dataToHost() = 0;

        /*! \brief Return number of elements of array
         *
         * This function returns the number of elements in the array. Note,
         * that this is not the size in bytes. For this, use size()*sizeof(T),
         * where T is the class used in the GPUArray.
         */
        virtual size_t size() const = 0;
};

#endif
