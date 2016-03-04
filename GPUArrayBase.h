#ifndef GPUARRAYBASE_H
#define GPUARRAYBASE_H

#include <vector>

using namespace std;

/*! \class GPUArrayBase
 * \brief Base class for a GPUArray
 */
class GPUArrayBase {
    protected:
        /*! \brief Constructor */
        GPUArrayBase() : size(0) {};

    public:
        /*! \brief Send data from host to GPU device */
        virtual void dataToDevice() = 0;

        /*! \brief Send data from GPU device to host */
        virtual void dataToHost() = 0;

        int size; //!< Size of the array
};

#endif
