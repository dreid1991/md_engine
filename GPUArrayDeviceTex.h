#pragma once
#ifndef GPUARRAYDEVICETEX_H
#define GPUARRAYDEVICETEX_H

#include <cuda_runtime.h>

#include "globalDefs.h"
#include "GPUArrayDevice.h"
#include "Logging.h"

void MEMSETFUNC(cudaSurfaceObject_t, void *, int, int);

/*! \brief Manager for data on a GPU Texture
 *
 * \tparam T type of data stored in the Texture
 *
 * This class manages data stored in a GPU Texture device. This type of memory
 * is often faster than the standard global or shared memory on the GPU. This
 * type of storage should be used only for runtime loop, not for general
 * storage.
 */
template <class T>
class GPUArrayDeviceTex : public GPUArrayDevice {
public:

    /*! \brief Default constructor */
    GPUArrayDeviceTex()
        : GPUArrayDevice(0), texObject(0), surfObject(0) {}

    /*! \brief Constructor
     *
     * \param desc_ Channel descriptor
     */
    GPUArrayDeviceTex(cudaChannelFormatDesc desc_)
        : GPUArrayDevice(0), texObject(0), surfObject(0), channelDesc(desc_)
    {
        initializeDescriptions();
    }

    /*! \brief Constructor
     *
     * \param size Size of the array (number of elements)
     * \param desc Channel descriptor
     */
    GPUArrayDeviceTex(size_t size, cudaChannelFormatDesc desc)
        : GPUArrayDevice(size), texObject(0), surfObject(0), channelDesc(desc)
    {
        initializeDescriptions();
        allocate();
    }

    /*! \brief Copy constructor
     *
     * \param other GPUArrayDeviceTex to copy from
     */
    GPUArrayDeviceTex(const GPUArrayDeviceTex<T> &other)
        : GPUArrayDevice(other.size()), texObject(0), surfObject(0),
          channelDesc(other.channelDesc)
    {
        initializeDescriptions();
        allocate();
        CUCHECK(cudaMemcpy2DArrayToArray(data(), 0, 0, other.data(), 0, 0,
                                         nX() * sizeof(T), nY(),
                                         cudaMemcpyDeviceToDevice));
    }

    /*! \brief Move constructor
     *
     * \param other GPUArrayDeviceTex containing the data to move
     */
    GPUArrayDeviceTex(GPUArrayDeviceTex<T> &&other) {
        // Take the values of the other object
        texObject = other.texObject;
        surfObject = other.surfObject;
        resDesc = other.resDesc;
        texDesc = other.texDesc;
        n = other.n;
        cap = other.cap;
        ptr = other.ptr;
        // Set the other object to zero
        other.texObject = 0;
        other.surfObject = 0;
        other.n = 0;
        other.cap = 0;
        other.ptr = nullptr;
    }

    /*! \brief Desctructor */
    ~GPUArrayDeviceTex() {
        deallocate();
    }

    /*! \brief Assignment operator
     *
     * \param other Right hand side of assignment operator
     *
     * \return This object
     */
    GPUArrayDeviceTex<T> &operator=(const GPUArrayDeviceTex<T> &other) {
        channelDesc = other.channelDesc;
        if (other.size()) {
            resize(other.size());
        }
        int x = nX();
        int y = nY();
        CUCHECK(cudaMemcpy2DArrayToArray(data(), 0, 0, other.data(), 0, 0,
                                         x*sizeof(T), y,
                                         cudaMemcpyDeviceToDevice));
        return *this;
    }

    /*! \brief Move assignment operator
     *
     * \param other Right hand side of assignment operator
     *
     * \return This object
     */
    GPUArrayDeviceTex<T> &operator=(GPUArrayDeviceTex<T> &&other) {
        // Clear this object
        deallocate();

        // Take the values of the other object
        texObject = other.texObject;
        surfObject = other.surfObject;
        resDesc = other.resDesc;
        texDesc = other.texDesc;
        n = other.n;
        cap = other.cap;
        ptr = other.ptr;

        // Set other object to zero
        other.texObject = 0;
        other.surfObject = 0;
        other.n = 0;
        other.cap = 0;
        other.ptr = nullptr;

        // Return this object
        return *this;
    }

    /*! \brief Initialize descriptors
     *
     * The default values are cudaResourceTypeArray for the resource type of
     * the resource descriptor and cudaReadModeElementType for the read mode
     * of the texture descriptor. All other values of the resource and texture
     * descriptors are set to zero.
     */
    void initializeDescriptions() {
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        //.res.array.array is unset.  Set when allocing on device
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
    }

    /*! \brief Create Texture and Surface Objects
     *
     * Objects are only created if they don't exist yet.
     */
    void createTexSurfObjs() {
        if (tex() == 0) {
            cudaCreateTextureObject(&texObject, &resDesc, &texDesc, NULL);
        }
        if (surf() == 0) {
            cudaCreateSurfaceObject(&surfObject, &resDesc);
        }
    }

    /*! \brief Resize the Texture Array
     *
     * \param newSize New size of the array
     * \param force Force reallocation of memory
     * \return True if memory was reallocated
     *
     * Resize the Texture array. If the new size is larger than capacity,
     * new memory is allocated. This function can destroy the data on the
     * GPU texture device.
     */
    virtual bool resize(size_t newSize, bool force = false) {
        bool memoryReallocated = GPUArrayDevice::resize(newSize, force);
        return memoryReallocated;
    }

    /*! \brief Access data pointer
     *
     * \return Pointer to device memory
     */
    cudaArray *data() { return (cudaArray *)ptr; }

    /*! \brief Const access to data pointer
     *
     * \return Pointer to const device memory
     */
    cudaArray const* data() const { return (cudaArray const*)ptr; }

    /*! \brief Access Texture Object
     *
     * \return Texture object to access current memory
     */
    cudaTextureObject_t tex() { createTexSurfObjs(); return texObject; }

    /*! \brief Access Surface Object
     *
     * \return Surface object to access current memory
     */
    cudaSurfaceObject_t surf() { createTexSurfObjs(); return surfObject; }

    /*! \brief Copy data from device to a given memory
     *
     * \param copyTo Pointer pointing to the memory taking the data
     *
     * \return Pointer to the position data is copied to
     */
    T *get(T *copyTo) {
        int x = nX();
        int y = nY();

        if (copyTo == (T *) NULL) {
            copyTo = (T *) malloc(x*y*sizeof(T));
        }
        CUCHECK(cudaMemcpy2DFromArray(copyTo, x * sizeof(T), data(), 0, 0,
                                      x * sizeof(T), y,
                                      cudaMemcpyDeviceToHost));
        return copyTo;
    }

    /*! \brief Copy data from pointer to device
     *
     * \param copyFrom Pointer to memory where to copy from
     */
    void set(T *copyFrom) {
        int x = nX();
        int y = nY();
        cudaMemcpy2DToArray(data(), 0, 0, copyFrom, x*sizeof(T),
                            x * sizeof(T), y, cudaMemcpyHostToDevice );
    }

    /*! \brief Copy data from device asynchronously
     *
     * \param copyTo Pointer where to copy data to
     * \param stream Cuda Stream object for asynchronous copying
     *
     * \return Pointer to memory where data was copied to
     */
    T *getAsync(T *copyTo, cudaStream_t stream) {
        int x = nX();
        int y = nY();

        if (copyTo == (T *) NULL) {
            copyTo = (T *) malloc(x*y*sizeof(T));
        }
        CUCHECK(cudaMemcpy2DFromArrayAsync(copyTo, x * sizeof(T), data(), 0, 0,
                                           x * sizeof(T), y,
                                           cudaMemcpyDeviceToHost, stream));
        return copyTo;

    }

    /*! \brief Copy data to GPU device
     *
     * \param dest Pointer to GPU memory
     */
    void copyToDeviceArray(void *dest) { //DEST HAD BETTER BE ALLOCATED
        int numBytes = size() * sizeof(T);
        //! \todo Make sure this works for copying from 2d arrays
        CUCHECK(cudaMemcpyFromArray(dest, data(), 0, 0, numBytes,
                                                cudaMemcpyDeviceToDevice));
    }

    /*! \brief Set all elements of GPUArrayDeviceTex to specific value
     *
     * \param val_ Value to set data to
     */
    void memsetByVal(T val_) {
        mdAssert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16,
                 "Type T has incompatible size");
        createTexSurfObjs();
        MEMSETFUNC(surfObject, &val_, size(), sizeof(T));
    }

private:
    /*! \brief Allocate memory on the Texture device */
    void allocate() {
        int x = nX();
        int y = nY();
        CUCHECK(cudaMallocArray((cudaArray_t *)(&ptr), &channelDesc, x, y) );
        cap = x*y;
        //assuming address gets set in blocking manner
        resDesc.res.array.array = data();
    }

    /*! \brief Destroy Texture and Surface objects, deallocate memory */
    void deallocate() {
        if (tex() != 0) {
            CUCHECK(cudaDestroyTextureObject(texObject));
            texObject = 0;
        }
        if (surf() != 0) {
            CUCHECK(cudaDestroySurfaceObject(surfObject));
            surfObject = 0;
        }
        if (data() != (cudaArray *) NULL) {
            CUCHECK(cudaFreeArray(data()));
        }
    }

    /*! \brief Get size in x-dimension of Texture Array
     *
     * \return Size in x-dimension
     */
    int nX() {
        return std::fmin((int) (PERLINE/sizeof(T)), (int) size());
    }

    /*! \brief Get size in y-dimension of Texture Array
     *
     * \return Size in y-dimension
     */
    int nY() {
        return std::ceil(size() / (float) (PERLINE/sizeof(T)));
    }

private:
    cudaTextureObject_t texObject; //!< Texture object
    cudaSurfaceObject_t surfObject; //!< Texture surface
    cudaResourceDesc resDesc; //!< Resource descriptor
    cudaTextureDesc texDesc; //!< Texture descriptor
    cudaChannelFormatDesc channelDesc; //!< Descriptor for the texture
};

#endif
