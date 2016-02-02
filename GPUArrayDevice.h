#ifndef GPUARRAYDEVICE_H
#define GPUARRAYDEVICE_H
#include "globalDefs.h"
#include "cutils_math.h"
#include "memset_defs.h"
/*
template <class T>
__global__ void memsetKernal(T *ptr, int n, T val) {
    int idx = GETIDX();
    if (idx < n) {
        ptr[n] = val;
    }

}
*/

void MEMSETFUNC(void *, void *, int, int);

template <class T>
class GPUArrayDevice {
    public:
        T *ptr;
        int n;
        int Tsize;
        GPUArrayDevice() {
            ptr = (T *) NULL;
            n = 0;
            Tsize = sizeof(T);
        }
        GPUArrayDevice(int n_) : n(n_) {
            allocate();
            Tsize = sizeof(T);
        }
        void allocate() {
            CUCHECK(cudaMalloc(&ptr, n * sizeof(T)));
        }
        void deallocate() {
            if (ptr != (T *) NULL) {
                CUCHECK(cudaFree(ptr));
                ptr = (T *) NULL;
            }
        }

        ~GPUArrayDevice() {
            deallocate();
        }
        GPUArrayDevice(const GPUArrayDevice<T> &other) { //copy constructor
            n = other.n;
            allocate();
            CUCHECK(cudaMemcpy(ptr, other.ptr, n*sizeof(T), cudaMemcpyDeviceToDevice));
            Tsize = sizeof(T);
        }
        GPUArrayDevice<T> &operator=(const GPUArrayDevice<T> &other) { //copy assignment
            if (n != other.n) {
                deallocate();
                n = other.n;
                allocate();
            }
            CUCHECK(cudaMemcpy(ptr, other.ptr, n*sizeof(T), cudaMemcpyDeviceToDevice));
            return *this;
        }
        GPUArrayDevice(GPUArrayDevice<T> &&other) { //move constructor;
            n = other.n;
            ptr = other.ptr;
            other.n = 0;
            other.ptr = (T *) NULL;
            Tsize = sizeof(T);
        }
        GPUArrayDevice<T> &operator=(GPUArrayDevice<T> &&other) { //move assignment
            deallocate();
            n = other.n;
            ptr = other.ptr;
            other.n = 0;
            other.ptr = (T *) NULL;
            return *this;
        }
        T *get(T *copyTo) {
            if (copyTo == (T *) NULL) {
                copyTo = (T *) malloc(n*sizeof(T));
            }
            CUCHECK(cudaMemcpy(copyTo, ptr, n*sizeof(T), cudaMemcpyDeviceToHost));
            return copyTo;
        }
        T *getAsync(T *copyTo, cudaStream_t stream) {
            if (copyTo == (T *) NULL) {
                copyTo = (T *) malloc(n*sizeof(T));
            }
            CUCHECK(cudaMemcpyAsync(copyTo, ptr, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
            return copyTo;
        }
        void set(T *copyFrom) {
            CUCHECK(cudaMemcpy(ptr, copyFrom, n*sizeof(T), cudaMemcpyHostToDevice));
        }
        void copyToDeviceArray(void *dest) {
            CUCHECK(cudaMemcpy(dest, ptr, n*sizeof(T), cudaMemcpyDeviceToDevice));
        }
        void memset(T val) {
            CUCHECK(cudaMemset(ptr, val, n*sizeof(T)));
        }
        void memsetByVal(T val_) {
            assert(Tsize==4 or Tsize==8 or Tsize==12 or Tsize==16);
            MEMSETFUNC((void *) ptr, &val_, n, Tsize);
        }

};
#endif
