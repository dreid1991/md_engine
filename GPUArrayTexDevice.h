#ifndef GPUARRAYTEXDEVICE_H
#define GPUARRAYTEXDEVICE_H

#include <cuda_runtime.h>
#include <cassert>

#include "GPUArrayTexBase.h"

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
class GPUArrayTexDevice : public GPUArrayTexBase {
public:

    cudaArray *d_data; //!< Pointer to the data
    int size; //!< Number of elements currently stored
    int capacity; //!< Number of elements fitting into the currently
                  //!< allocated memory
    cudaTextureObject_t tex; //!< Texture object
    cudaSurfaceObject_t surf; //!< Texture surface
    cudaResourceDesc resDesc; //!< Resource descriptor
    cudaTextureDesc texDesc; //!< Texture descriptor
    bool madeTex; //!< True if texture has been created.

    /*! \brief Destroy Texture and Surface objects, deallocate memory */
    void destroyDevice() {
        if (madeTex) {
            CUCHECK(cudaDestroyTextureObject(tex));
            CUCHECK(cudaDestroySurfaceObject(surf));
        }
        if (d_data != (cudaArray *) NULL) {
            CUCHECK(cudaFreeArray(d_data));
        }
        madeTex = false;
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

    /*! \brief Default constructor */
    GPUArrayTexDevice() : madeTex(false) {
        d_data = (cudaArray *) NULL;
        size = 0;
        capacity = 0;
    }

    /*! \brief Constructor
     *
     * \param desc_ Channel descriptor
     */
    GPUArrayTexDevice(cudaChannelFormatDesc desc_) : madeTex(false) {
        d_data = (cudaArray *) NULL;
        channelDesc = desc_;
        initializeDescriptions();
        size = 0;
        capacity = 0;
    }

    /*! \brief Constructor
     *
     * \param size_ Size of the array (number of elements)
     * \param desc_ Channel descriptor
     */
    GPUArrayTexDevice(int size_, cudaChannelFormatDesc desc_)
        : madeTex(false)
    {
        size = size_;
        channelDesc = desc_;
        initializeDescriptions();
        allocDevice();
        createTexSurfObjs();
    }

    /*! \brief Desctructor */
    ~GPUArrayTexDevice() {
        destroyDevice();
    }

    /*! \brief Copy constructor
     *
     * \param other GPUArrayTexDevice to copy from
     */
    GPUArrayTexDevice(const GPUArrayTexDevice<T> &other) {
        channelDesc = other.channelDesc;
        size = other.size;
        capacity = other.capacity;
        initializeDescriptions();
        allocDevice();
        CUCHECK(cudaMemcpy2DArrayToArray(d_data, 0, 0, other.d_data, 0, 0,
                                         NX() * sizeof(T), NY(),
                                         cudaMemcpyDeviceToDevice));
        createTexSurfObjs();
    }

    /*! \brief Assignment operator
     *
     * \param other Right hand side of assignment operator
     *
     * \return This object
     */
    GPUArrayTexDevice<T> &operator=(const GPUArrayTexDevice<T> &other) {
        channelDesc = other.channelDesc;
        if (other.size) {
            resize(other.size); //creates tex surf objs
        }
        int x = NX();
        int y = NY();
        CUCHECK(cudaMemcpy2DArrayToArray(d_data, 0, 0, other.d_data, 0, 0,
                                         x*sizeof(T), y,
                                         cudaMemcpyDeviceToDevice));
        return *this;
    }

    /*! \brief Custom copy operator
     *
     * \param other GPUArrayTexDevice to copy from
     */
    void copyFromOther(const GPUArrayTexDevice<T> &other) {
        //I should own no pointers at this point, am just copying other's
        channelDesc = other.channelDesc;
        size = other.size;
        capacity = other.capacity;
        d_data = other.d_data;


    }

    /*! \brief Set other GPUArrayTexDevice to NULL state
     *
     * \param other GPUArrayTexDevice to set to NULL
     */
    void nullOther(GPUArrayTexDevice<T> &other) {
        other.d_data = (cudaArray *) NULL;
        other.size = 0;
        other.capacity = 0;

    }

    /*! \brief Move constructor
     *
     * \param other GPUArrayTexDevice containing the data to move
     */
    GPUArrayTexDevice(GPUArrayTexDevice<T> &&other) {
        copyFromOther(other);
        d_data = other.d_data;
        initializeDescriptions();
        resDesc.res.array.array = d_data;
        if (other.madeTex) {
            createTexSurfObjs();
        }
        nullOther(other);
    }

    /*! \brief Move assignment operator
     *
     * \param other Right hand side of assignment operator
     *
     * \return This object
     */
    GPUArrayTexDevice<T> &operator=(GPUArrayTexDevice<T> &&other) {
        destroyDevice();
        copyFromOther(other);
        initializeDescriptions();
        resDesc.res.array.array = d_data;
        if (other.madeTex) {
            createTexSurfObjs();

        }
        nullOther(other);
        return *this;
    }

    /*! \brief Create Texture and Surface Objects */
    void createTexSurfObjs() {

        tex = 0;
        surf = 0;
        cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
        cudaCreateSurfaceObject(&surf, &resDesc);
        madeTex = true;
    }

    /*! \brief Get size in x-dimension of Texture Array
     *
     * \return Size in x-dimension
     */
    int NX() {
        return std::fmin((int) (PERLINE/sizeof(T)), (int) size);
    }

    /*! \brief Get size in y-dimension of Texture Array
     *
     * \return Size in y-dimension
     */
    int NY() {
        return std::ceil(size / (float) (PERLINE/sizeof(T)));
    }

    /*! \brief Resize the Texture Array
     *
     * \param n_ New size
     *
     * Resize the Texture array. If the new size is larger than capacity,
     * new memory is allocated. This function can destroy the data on the
     * GPU texture device.
     */
    void resize(int n_) {
        if (n_ > capacity) {
            destroyDevice();
            size = n_;
            allocDevice();
            createTexSurfObjs();
        } else {
            size = n_;
        }

    }

    /*! \brief Allocate memory on the Texture device */
    void allocDevice() {
        int x = NX();
        int y = NY();
        CUCHECK(cudaMallocArray(&d_data, &channelDesc, x, y) );
        capacity = x*y;
        //assuming address gets set in blocking manner
        resDesc.res.array.array = d_data;
    }

    /*! \brief Copy data from device to a given memory
     *
     * \param copyTo Pointer pointing to the memory taking the data
     *
     * \return Pointer to the position data is copied to
     */
    T *get(T *copyTo) {
        int x = NX();
        int y = NY();

        if (copyTo == (T *) NULL) {
            copyTo = (T *) malloc(x*y*sizeof(T));
        }
        CUCHECK(cudaMemcpy2DFromArray(copyTo, x * sizeof(T), d_data, 0, 0,
                                      x * sizeof(T), y,
                                      cudaMemcpyDeviceToHost));
        return copyTo;
    }

    /*! \brief Copy data from pointer to device
     *
     * \param copyFrom Pointer to memory where to copy from
     */
    void set(T *copyFrom) {
        int x = NX();
        int y = NY();
        cudaMemcpy2DToArray(d_data, 0, 0, copyFrom, x*sizeof(T),
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
        int x = NX();
        int y = NY();

        if (copyTo == (T *) NULL) {
            copyTo = (T *) malloc(x*y*sizeof(T));
        }
        CUCHECK(cudaMemcpy2DFromArrayAsync(copyTo, x * sizeof(T), d_data, 0, 0,
                                           x * sizeof(T), y,
                                           cudaMemcpyDeviceToHost, stream));
        return copyTo;

    }

    /*! \brief Copy data to GPU device
     *
     * \param dest Pointer to GPU memory
     */
    void copyToDeviceArray(void *dest) { //DEST HAD BETTER BE ALLOCATED
        int numBytes = size * sizeof(T);
        copyToDeviceArrayInternal(dest, d_data, numBytes);

    }

    /*! \brief Set all elements of GPUArrayTexDevice to specific value
     *
     * \param val_ Value to set data to
     */
    void memsetByVal(T val_) {
        assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16);
        MEMSETFUNC(surf, &val_, size, sizeof(T));
    }
};

#endif
