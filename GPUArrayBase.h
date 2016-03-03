#ifndef GPUARRAYBASE_H
#define GPUARRAYBASE_H

#include <vector>

using namespace std;

/*! \class GPUArrayBase
 * \brief Base class for a GPUArray
 */
class GPUArrayBase {
    public:
        /*! \brief Constructor */
        GPUArrayBase() : size(0) {};

        /*! \brief Send data from host to GPU device */
        virtual void dataToDevice(){};

        /*! \brief Send data from GPU device to host */
        virtual void dataToHost(){};

        int size; //!< Size of the array
};
#endif
