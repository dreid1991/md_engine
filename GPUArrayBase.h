#ifndef GPUARRAYBASE_H
#define GPUARRAYBASE_H

/*! \class GPUArrayBase
 * \brief Base class for a GPUArray
 */
class GPUArrayBase {
    protected:
        /*! \brief Constructor */
        GPUArrayBase() {};

    public:
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
        virtual int size() const = 0;
};

#endif
