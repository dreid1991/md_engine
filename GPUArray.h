#ifndef GPUARRAY_H
#define GPUARRAY_H
#include "Python.h"
#include <vector>
#include <cuda_runtime.h>
#include "cutils_math.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "globalDefs.h"
#include "GPUArrayBase.h"
#include "GPUArrayDevice.h"
using namespace std;







template <class T>
class GPUArray : public GPUArrayBase {
    void setHost(vector<T> &vals) {
        h_data = vals;
        size = vals.size();
    }
    void setHostCanFit(vector<T> &vals) {

        memcpy(h_data.data(), vals.data(), sizeof(T) * vals.size());
        size = vals.size();
        //okay... may be some excess on host, device.  can deal with later if problem
    }
    public:

        GPUArrayDevice<T> d_data;
        vector<T> h_data;
        GPUArray() {
            h_data = vector<T>();
        }
        GPUArray(int size_) {
            T fillVal = T();
            size = size_;
            h_data = vector<T>(size, fillVal);
            d_data = GPUArrayDevice<T>(size);

        }
        GPUArray(vector<T> &vals) {
            set(vals);
            if (!vals.size()) {
                d_data = GPUArrayDevice<T>(vals.size());
            }
        }

        bool set(vector<T> &other) {
            if (other.size() < size) {
                setHostCanFit(other);
                return true;
            } else {
                d_data = GPUArrayDevice<T>(other.size());
                setHost(other);
            }
            return false;

        }
        void ensureSize() {
            if (h_data.size() > d_data.n) { 
                d_data = GPUArrayDevice<T>(h_data.size());
            }
        }
        void dataToDevice() {
            d_data.set(h_data.data());

        }
        void dataToHostAsync(cudaStream_t stream) {
            d_data.getAsync(h_data.data(), stream);
        }
        void dataToHost() {
            //eeh, want to deal with the case where data originates on the device, which is a real case, so removed checked on whether data is on device or not
            d_data.get(h_data.data());
        }
        void copyToDeviceArray(void *dest) {
            d_data.copyToDeviceArray(dest);
        }
        T *getDevData() {
            return d_data.ptr;
        }
        void memsetByVal(T val) {
            d_data.memsetByVal(val);
        }

};
#endif
