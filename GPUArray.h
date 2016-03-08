#ifndef GPUARRAY_H
#define GPUARRAY_H

#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "Python.h"
#include "cutils_math.h"
#include "globalDefs.h"
#include "GPUArrayBase.h"
#include "GPUArrayDevice.h"

using namespace std;

/* \class GPUArray
 * \brief Array storing data on the CPU and the GPU
 *
 * \tparam T Data type stored in the array
 *
 * GPU array stores data on the host CPU and the GPU device and is able to
 * move the data from the CPU to the GPU and back again.
 */
template <typename T>
class GPUArray : public GPUArrayBase {

public:
    /*! \brief Constructor
     *
     * Constructor creating empty host CPU data vector.
     */
    GPUArray() {}

    /*! \brief Constructor
     *
     * \param size_ Size (number of elements) on the CPU and GPU data array
     *
     * Constructor creating empty arrays of the specified size on the CPU
     * and the GPU.
     */
    GPUArray(int size_)
        : h_data(vector<T>(size_,T())), d_data(GPUArrayDevice<T>(size_)) {}

    /*! \brief Constructor
     *
     * \param vals Vector to be passed to the CPU array
     *
     * Constructor setting the CPU data array with the specified vector.
     */
    GPUArray(vector<T> &vals) {
        set(vals);
        if (!vals.size()) {
            d_data = GPUArrayDevice<T>(vals.size());
        }
    }

    /*! \brief Set CPU data
     *
     * \param other Vector containing new data
     *
     * Set the CPU data to to data specified in the given vector.
     */
    bool set(vector<T> &other) {
        if (other.size() < size()) {
            setHost(other);
            return true;
        } else {
            d_data = GPUArrayDevice<T>(other.size());
            setHost(other);
        }
        return false;

    }

    /*! \brief Return size of data array */
    int size() const { return h_data.size(); }

    /*! \brief Ensure that the GPU data array is large enough */
    void ensureSize() {
        if (h_data.size() > d_data.n) { 
            d_data = GPUArrayDevice<T>(size());
        }
    }

    /*! \brief Send data from CPU to GPU */
    void dataToDevice() {
        d_data.set(h_data.data());

    }

    /*! \brief Send data from GPU to CPU asynchronously */
    void dataToHostAsync(cudaStream_t stream) {
        d_data.getAsync(h_data.data(), stream);
    }

    /*! \brief Send data from GPU to CPU synchronously */
    void dataToHost() {
        //eeh, want to deal with the case where data originates on the device,
        //which is a real case, so removed checked on whether data is on device
        //or not
        d_data.get(h_data.data());
    }

    /*! \brief Copy data to GPU array */
    void copyToDeviceArray(void *dest) {
        d_data.copyToDeviceArray(dest);
    }

    /*! \brief Return pointer to GPU data array */
    T *getDevData() {
        return d_data.ptr;
    }

    /*! \brief Set Memory by value */
    void memsetByVal(T val) {
        d_data.memsetByVal(val);
    }

private:
    /*! \brief Set the host data
     *
     * \param vals Vector with the data to be stored in the CPU data vector.
     *
     * Set the host data.
     */
    void setHost(vector<T> &vals) {
        h_data = vals;
    }

public:

    GPUArrayDevice<T> d_data; //!< Array storing data on the GPU
    vector<T> h_data; //!< Array storing data on the CPU
};

#endif
