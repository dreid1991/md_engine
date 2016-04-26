#pragma once
#ifndef GPUARRAYGLOBAL_H
#define GPUARRAYGLOBAL_H

#include <vector>

#include "GPUArray.h"
#include "GPUArrayDeviceGlobal.h"

//! Array storing data on the CPU and the GPU
/*!
 * \tparam T Data type stored in the array
 *
 * GPU array stores data on the host CPU and the GPU device and is able to
 * move the data from the CPU to the GPU and back again.
 */
template <typename T>
class GPUArrayGlobal : public GPUArray {

public:
    //! Constructor
    /*!
     * Constructor creating empty host CPU data vector.
     */
    GPUArrayGlobal() {}

    //! Constructor
    /*!
     * \param size_ Size (number of elements) on the CPU and GPU data array
     *
     * Constructor creating empty arrays of the specified size on the CPU
     * and the GPU.
     */
    explicit GPUArrayGlobal(int size_)
        : h_data(std::vector<T>(size_,T())), d_data(GPUArrayDeviceGlobal<T>(size_)) {}

    //! Copy from vector constructor
    /*!
     * \param vals Vector to be passed to the CPU array
     *
     * Constructor setting the CPU data array with the specified vector.
     */
    explicit GPUArrayGlobal(std::vector<T> const &vals) {
        h_data = vals;
        d_data = GPUArrayDeviceGlobal<T>(h_data.size());
    }

    //! Set CPU data
    /*!
     * \param other Vector containing new data
     *
     * Set the CPU data to to data specified in the given vector.
     */
    void set(std::vector<T> const &other) {
        if (other.size() > size()) {
            d_data = GPUArrayDeviceGlobal<T>(other.size());
        }
        h_data = other;

    }

    //! Return number of elements stored in the array
    /*!
     * \return Number of elements in the array
     */
    size_t size() const { return h_data.size(); }

    //! Ensure that the GPU data array is large enough to store data
    void ensureSize() {
        if (h_data.size() > d_data.size()) {
            d_data = GPUArrayDeviceGlobal<T>(size());
        }
    }

    //! Send data from CPU to GPU
    void dataToDevice() {
        d_data.set(h_data.data());
    }

    //! Send data from GPU to CPU asynchronously
    void dataToHostAsync(cudaStream_t stream) {
        d_data.get(h_data.data(), stream);
    }

    //! Send data from GPU to CPU synchronously
    void dataToHost() {
        //eeh, want to deal with the case where data originates on the device,
        //which is a real case, so removed checked on whether data is on device
        //or not
        d_data.get(h_data.data());
    }

    //! Copy data to GPU array
    void copyToDeviceArray(void *dest) {
        d_data.copyToDeviceArray(dest);
    }

    //! Return pointer to GPU data array
    T *getDevData() {
        return d_data.data();
    }

    //! Set Memory by value
    void memsetByVal(T val) {
        d_data.memsetByVal(val);
    }

public:

    std::vector<T> h_data; //!< Array storing data on the CPU
    GPUArrayDeviceGlobal<T> d_data; //!< Array storing data on the GPU
};

#endif
