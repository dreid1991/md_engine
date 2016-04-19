#pragma once
#ifndef GPUARRAYDEVICEGLOBAL_H
#define GPUARRAYDEVICEGLOBAL_H

#include <cuda_runtime.h>

#include "globalDefs.h"
#include "GPUArrayDevice.h"
#include "Logging.h"

//! Global function to set the device memory
/*!
 * \param ptr Pointer to the memory
 * \param val Pointer to the value that will be stored into all elements
 * \param n Number of elements that will be set to val
 * \param Tsize Size of the value type
 */
void MEMSETFUNC(void *ptr, const void *val, size_t n, size_t Tsize);

//!Array on the GPU device
/*!
 * \tparam T Data type stored in the array
 *
 * Array storing data on the GPU device.
 */
template <typename T>
class GPUArrayDeviceGlobal : public GPUArrayDevice {
public:
    //! Constructor
    /*!
     * \param size Size of the array (number of elements)
     *
     * This constructor creates the array on the GPU device and allocates
     * enough memory to store n_ elements.
     */
    explicit GPUArrayDeviceGlobal(size_t size = 0)
        : GPUArrayDevice(size) { allocate(); }

    //! Copy constructor
    GPUArrayDeviceGlobal(const GPUArrayDeviceGlobal<T> &other)
        : GPUArrayDevice(other.n)
    {
        allocate();
        CUCHECK(cudaMemcpy(ptr, other.ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToDevice));
    }

    //! Move constructor
    GPUArrayDeviceGlobal(GPUArrayDeviceGlobal<T> &&other)
        : GPUArrayDevice(other.n)
    {
        ptr = other.ptr;
        other.n = 0;
        other.cap = 0;
        other.ptr = nullptr;
    }

    //! Destructor
    ~GPUArrayDeviceGlobal() {
        deallocate();
    }

    //! Assignment operator
    /*!
     * \param other GPUArrayDeviceGlobal from which data is copied
     * \return This object
     */
    GPUArrayDeviceGlobal<T> &operator=(const GPUArrayDeviceGlobal<T> &other) {
        if (n != other.n) {
            //! \todo Think about if it would be better not to force
            //!       reallocation here
            resize(other.n, true); // Force resizing
        }
        CUCHECK(cudaMemcpy(ptr, other.ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToDevice));
        return *this;
    }

    //! Move assignment operator
    GPUArrayDeviceGlobal<T> &operator=(GPUArrayDeviceGlobal<T> &&other) {
        deallocate();
        n = other.n;
        ptr = other.ptr;
        other.n = 0;
        other.cap = 0;
        other.ptr = nullptr;
        return *this;
    }

    //! Access pointer to data
    /*!
     * \return Pointer to memory location
     */
    T *data() { return (T*)ptr; }

    //! Const access pointer to data
    /*!
     * \return Const pointer to memory location
     */
    T const *data() const { return (T const*)ptr; }

    //! Copy data to given pointer
    /*!
     * \param copyTo Pointer to the memory to where the data will be copied
     *
     * This function copies the data stored in the GPU array to the
     * position specified by the pointer *copyTo using cudaMemcpy
     */
    void get(T *copyTo) const {
        if (copyTo == nullptr) {
            return;
        }
        CUCHECK(cudaMemcpy(copyTo, ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToHost));
    }

    //! Copy a given amount of data from a specific position of the array
    /*!
     * \param copyTo CPU Memory location where the data is copied to
     * \param offset Initial array index to be copied (0 is first element)
     * \param num Number of elements to be copied
     *
     * \return Pointer to memory location where data was copied to
     */
    T *getWithOffset(T *copyTo, int offset, int num) {
        if (copyTo == (T *) NULL) {
            copyTo = (T *) malloc(n*sizeof(T));
        }
        T *pointer = data();
        CUCHECK(cudaMemcpy(copyTo, pointer+offset, num*sizeof(T),
                                                cudaMemcpyDeviceToHost));
        return copyTo;
    }

    //! Copy data to pointer asynchronously
    /*!
     * \param copyTo Pointer to the memory where the data will be copied to
     * \param stream cudaStream_t object used for asynchronous copy
     *
     * \return Pointer to CPU memory location where data was copied to
     *
     * Copy data stored in the GPU array to the address specified by the
     * pointer copyTo using cudaMemcpyAsync.
     */
    T *getAsync(T *copyTo, cudaStream_t stream) const {
        if (copyTo == (T *) NULL) {
            copyTo = (T *) malloc(n*sizeof(T));
        }
        CUCHECK(cudaMemcpyAsync(copyTo, ptr, n*sizeof(T),
                                        cudaMemcpyDeviceToHost, stream));
        return copyTo;
    }

    //! Copy data from pointer
    /*!
     * \param copyFrom Pointer to address from which the data will be
     *                 copied
     *
     * Copy data from a given adress specified by the copyFrom pointer to
     * the GPU array. The number of bytes copied from memory is the size of
     * the the GPUArrayDeviceGlobal.
     */
    void set(const T *copyFrom) {
        CUCHECK(cudaMemcpy(ptr, copyFrom, size()*sizeof(T),
                                                cudaMemcpyHostToDevice));
    }

    //! Copy data to pointer
    /*!
     * \param dest Pointer to the memory to which the data should be copied
     *
     * Copy data from the device to the memory specified by the pointer
     * dest.
     *
     * \todo This function is essential identical to get(). Can we have one
     * function that covers both cases (void* and T* pointers)?
     */
    void copyToDeviceArray(void *dest) const {
        CUCHECK(cudaMemcpy(dest, ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToDevice));
    }

    //! Copy data to pointer asynchronously
    /*!
     * \param dest Pointer to the memory to which the data should be copied
     * \param stream cudaStream_t object for asynchronous copy
     *
     * Copy data from the device to the memory specified by the dest
     * pointer using cudaMemcpyAsync.
     *
     * \todo This function is essentially identical to getAsync(). Can we
     * have one function that covers both cases (void* and T* pointers)?
     */
    void copyToDeviceArrayAsync(void *dest, cudaStream_t stream) const {
        CUCHECK(cudaMemcpyAsync(dest, ptr, n*sizeof(T),
                                        cudaMemcpyDeviceToDevice, stream));
    }

    //! Set all bytes in the array to a specific value
    /*!
     * \param val Value written to all bytes in the array
     *
     * WARNING: val must be a one byte value
     *
     * Set each byte in the array to the value specified by val.
     *
     * \todo For this function val needs to be converted to unsigned char
     * and this value is used.
     */
    void memset(int val) {
        CUCHECK(cudaMemset(ptr, val, n*sizeof(T)));
    }

    //! Set array elements to a specific value
    /*!
     * \param val Value the elements are set to
     *
     * Set all array elements to the value specified by the parameter val
     */
    void memsetByVal(const T &val) {
        mdAssert(sizeof(T) == 4  || sizeof(T) == 8 ||
                 sizeof(T) == 12 || sizeof(T) == 16,
                 "Type parameter incompatible size");
        MEMSETFUNC(ptr, &val, n, sizeof(T));
    }

private:
    //! Allocate memory
    void allocate() { CUCHECK(cudaMalloc(&ptr, n * sizeof(T))); cap = size(); }

    //! Deallocate memory
    void deallocate() {
        CUCHECK(cudaFree(ptr));
        n = 0;
        cap = 0;
        ptr = nullptr;
    }
};

#endif
