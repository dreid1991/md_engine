#ifndef GPUARRAYTEX_H
#define GPUARRAYTEX_H
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
#include "GPUArrayTexBase.h"
#include "GPUArrayTexDevice.h"
using namespace std;



//for use in runtime loop, not general storage



template <class T>
class GPUArrayTex : public GPUArrayBase, public GPUArrayTexBase {
    public:
        GPUArrayTexDevice<T> d_data;
        vector<T> h_data;
        GPUArrayTex() {
        }
        GPUArrayTex(cudaChannelFormatDesc desc_) : d_data(desc_) {
        }
        GPUArrayTex(vector<T> vals, cudaChannelFormatDesc desc_) : d_data(vals.size(), desc_) {
            set(vals);
        }
        bool set(vector<T> &other) {
            d_data.resize(other.size());
            h_data = other;
            h_data.reserve(d_data.capacity);
            return true;
        }
        void dataToDevice() {
            d_data.set(h_data.data());
        }
        void dataToHost() {
            d_data.get(h_data.data());
        }

        int size() const { return h_data.size(); }

        void ensureSize() {
            d_data.resize(h_data.size());
        }
        void dataToHostAsync(cudaStream_t stream) {
            d_data.getAsync(h_data.data(), stream);
        }
        void copyToDeviceArray(void *dest) { //DEST HAD BETTER BE ALLOCATED
            int numBytes = size() * sizeof(T);
            copyToDeviceArrayInternal(dest, d_data.d_data, numBytes);

        }
        cudaTextureObject_t getTex() {
            return d_data.tex;
        }
        cudaSurfaceObject_t getSurf() {
            return d_data.surf;
        }
        void memsetByVal(T val) {
            d_data.memsetByVal(val);
        }
};
#endif
