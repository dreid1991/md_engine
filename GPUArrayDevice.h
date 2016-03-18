#ifndef GPUARRAYDEVICE_H
#define GPUARRAYDEVICE_H

#include "cutils_math.h"
#include "globalDefs.h"
#include "memset_defs.h"

/*! \brief Global function to set the device memory */
void MEMSETFUNC(void *, const void *, size_t, size_t);

/*! \class GPUArrayDevice
 * \brief Array on the GPU device
 *
 * \tparam T Data type stored in the array
 *
 * Array storing data on the GPU device.
 */
template <typename T>
class GPUArrayDevice {
public:
    /*! \brief Constructor
     *
     * \param n_ Size of the array (number of elements)
     *
     * This constructor creates the array on the GPU device and allocates
     * enough memory to store n_ elements.
     */
    explicit GPUArrayDevice(size_t n_ = 0)
        : n(n_) { allocate(); }

    /*! \brief Copy constructor */
    GPUArrayDevice(const GPUArrayDevice<T> &other)
        : n(other.n)
    {
        allocate();
        CUCHECK(cudaMemcpy(ptr, other.ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToDevice));
    }

    /*! \brief Move constructor */
    GPUArrayDevice(GPUArrayDevice<T> &&other)
        : ptr(other.ptr), n(other.n)
    {
        other.n = 0;
        other.ptr = (T *) NULL;
    }

    /*! \brief Destructor */
    ~GPUArrayDevice() {
        deallocate();
    }

    /*! \brief Assignment operator */
    GPUArrayDevice<T> &operator=(const GPUArrayDevice<T> &other) {
        if (n != other.n) {
            deallocate();
            n = other.n;
            allocate();
        }
        CUCHECK(cudaMemcpy(ptr, other.ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToDevice));
        return *this;
    }

    /*! \brief Move assignment operator */
    GPUArrayDevice<T> &operator=(GPUArrayDevice<T> &&other) {
        deallocate();
        n = other.n;
        ptr = other.ptr;
        other.n = 0;
        other.ptr = (T *) NULL;
        return *this;
    }

    /*! \brief Get size (number of elements) of array */
    size_t size() const { return n; }

    /*! \brief Access pointer to data */
    T *data() { return ptr; }
    const T *data() const { return ptr; }

    /*! \brief Copy data to given pointer
     *
     * \param copyTo Pointer to the memory to where the data will be copied
     *
     * This function copies the data stored in the GPU array to the
     * position specified by the pointer *copyTo using cudaMemcpy
     */
    T *get(T *copyTo) const {
        if (copyTo == (T *) NULL) {
            copyTo = (T *) malloc(n*sizeof(T));
        }
        CUCHECK(cudaMemcpy(copyTo, ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToHost));
        return copyTo;
    }

    /*! \brief Copy data to pointer asynchronously
     *
     * \param copyTo Pointer to the memory where the data will be copied to
     * \param stream cudaStream_t object used for asynchronous copy
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

    /*! \brief Copy data from pointer
     *
     * \param copyFrom Pointer to address from which the data will be
     *                 copied
     *
     * Copy data from a given adress specified by the copyFrom pointer to
     * the GPU array. The number of bytes copied from memory is the size of
     * the the GPUArrayDevice.
     */
    void set(const T *copyFrom) {
        CUCHECK(cudaMemcpy(ptr, copyFrom, n*sizeof(T),
                                                cudaMemcpyHostToDevice));
    }

    /*! \brief Copy data to pointer
     *
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

    /*! \brief Copy data to pointer asynchronously
     *
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

    /*! \brief Set all bytes in the array to a specific value
     *
     * \param val Value written to all bytes in the array
     *
     * WARNING: val must be a one byte value
     *
     * Set each byte in the array to the value specified by val.
     *
     * \todo For this function val needs to be converted to unsigned char
     * and this value is used.
     */
    void memset(const T &val) {
        CUCHECK(cudaMemset(ptr, val, n*sizeof(T)));
    }

    /*! \brief Set array elements to a specific value
     *
     * \param val Value the elements are set to
     *
     * Set all array elements to the value specified by the parameter val
     */
    void memsetByVal(const T &val) {
        assert(sizeof(T) == 4  || sizeof(T) == 8 ||
               sizeof(T) == 12 || sizeof(T) == 16);
        MEMSETFUNC((void *) ptr, &val, n, sizeof(T));
    }

private:
    /*! \brief Allocate memory */
    void allocate() { CUCHECK(cudaMalloc(&ptr, n * sizeof(T))); }

    /*! \brief Deallocate memory */
    void deallocate() {
        CUCHECK(cudaFree(ptr));
        ptr = (T *) NULL;
    }

private:
    T *ptr; //!< Pointer to the data
    size_t n; //!< Number of entries stored in the device
};

#endif
