#pragma once
#ifndef GPUARRAYDEVICE_H
#define GPUARRAYDEVICE_H

#include <cstddef>

//! Base class for GPUArrayDevices
/*!
 * A GPUArrayDevice is a memory-managed pointer for storgage on the GPU. It is
 * mainly used by the GPUArray which manages the GPU memory and takes care of
 * sending data from CPU to GPU and back.
 *
 * This class is a Base class defining the function common to all memory
 * operations on the GPU. The child classes differ in which type of memory
 * they store the data: Global memory or Texture memory. Not yet
 * used/implemented is memory stored to Constant memory or Local memory.
 */
class GPUArrayDevice {
protected:
    //! Constructor
    /*!
     * \param size Size of the array
     */
    GPUArrayDevice(size_t size = 0) : n(size), cap(0), ptr(nullptr) {}

public:
    //! Destructor
    virtual ~GPUArrayDevice() = default;

    //! Get the size of the array
    /*!
     * \return Number of elements stored in the array
     */
    size_t size() const { return n; }

    //! Get the capacity of the array
    /*!
     * \return Capacity
     *
     * The capacity is the number of elements that can be stored in the
     * currently allocated memory.
     */
    size_t capacity() const { return cap; }

    //! Change size of the array
    /*!
     * \param newSize New size of the array
     * \param force Force reallocation of memory
     * \return True if memory is reallocated. Else return false.
     *
     * Resize the array. If newSize is larger than the capacity, the current
     * memory is deallocated and new memory is allocated. In this case, the
     * stored memory is lost and the function returns false.
     */
    virtual bool resize(size_t newSize, bool force = false);

private:
    //! Allocate memory for the array
    virtual void allocate() = 0;

    //! Deallocate memory
    virtual void deallocate() = 0;

protected:
    size_t n; //!< Number of elements stored in the array
    size_t cap; //!< Capacity of allocated memory
    void *ptr; //!< Pointer to memory location
};

#endif

