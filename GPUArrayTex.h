#pragma once
#ifndef GPUARRAYTEX_H
#define GPUARRAYTEX_H

#include <vector>

#include "GPUArrayBase.h"
#include "GPUArrayTexBase.h"
#include "GPUArrayTexDevice.h"

/*! \brief Manage data on the CPU and a GPU Texture
 *
 * \tparam T Type of data stored on the CPU and GPU
 *
 * This class manages data stored on the CPU and a GPU texture device. The
 * class allocates memory both on the CPU and the GPU and transfers data
 * between the two. This class is designed only for use in the runtime loop,
 * not for general storage.
 */
template <class T>
class GPUArrayTex : public GPUArrayBase, public GPUArrayTexBase {
    public:
        GPUArrayTexDevice<T> d_data; //!< Array storing data on the GPU
        std::vector<T> h_data; //!< Array storing data on the CPU

        /*! \brief Default constructor */
        GPUArrayTex() {
        }

        /*! \brief Constructor
         *
         * \param desc_ Cuda channel descriptor for asynchronous data transfer
         */
        GPUArrayTex(cudaChannelFormatDesc desc_) : d_data(desc_) {
        }

        /*! \brief Constructor
         *
         * \param vals Vector containing data
         * \param desc_ Cuda channel descriptor for asynchronous data transfer
         *
         * This constructor allocates memory on the CPU and GPU large enough to
         * fit the data given in the vector. Then, it copies the data to the
         * CPU memory. The GPU memory remains unset.
         */
        GPUArrayTex(std::vector<T> vals, cudaChannelFormatDesc desc_)
            : d_data(vals.size(), desc_)
        {
            set(vals);
        }

        /*! \brief Set the CPU memory
         *
         * \param other Vector containing data
         * \return True always
         *
         * Copy data from vector to the CPU memory.
         */
        bool set(std::vector<T> &other) {
            d_data.resize(other.size());
            h_data = other;
            h_data.reserve(d_data.capacity());
            return true;
        }

        /*! \brief Send data from CPU to GPU */
        void dataToDevice() {
            d_data.set(h_data.data());
        }

        /*! \brief Send data from GPU to CPU */
        void dataToHost() {
            d_data.get(h_data.data());
        }

        /*! \brief Return number of elements stored in array
         *
         * \return Number of elements
         */
        size_t size() const { return h_data.size(); }

        /*! \brief Resize the GPU array to be large enough to contain CPU data
         * \todo This function should not be necessary
         */
        void ensureSize() {
            d_data.resize(h_data.size());
        }

        /*! \brief Copy data from GPU to CPU asynchronously
         *
         * \todo It would be nicer to call dataToHost and have it copy
         *       asynchronously if a stream is passed and synchronously
         *       otherwise
         */
        void dataToHostAsync(cudaStream_t stream) {
            d_data.getAsync(h_data.data(), stream);
        }

        /*! \brief Copy data from GPU texture to other GPU memory
         *
         * \param dest Pointer to GPU memory, destination for copy
         */
        void copyToDeviceArray(void *dest) { //DEST HAD BETTER BE ALLOCATED
            int numBytes = size() * sizeof(T);
            copyToDeviceArrayInternal(dest, d_data.data(), numBytes);

        }

        /*! \brief Return texture object from GPUArrayTexDevice
         *
         * \return Cuda Texture Object used for GPU memory storage
         */
        cudaTextureObject_t getTex() {
            return d_data.tex;
        }

        /*! \brief Return surface object from GPUArrayTexDevice
         *
         * \return Cuda Surface Object used to write to GPU texture memory
         */
        cudaSurfaceObject_t getSurf() {
            return d_data.surf;
        }

        /*! \brief
         *
         * \param
         */
        void memsetByVal(T val) {
            d_data.memsetByVal(val);
        }
};
#endif
