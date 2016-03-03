#ifndef GPUARRAYTEXPAIR_H
#define GPUARRAYTEXPAIR_H
#include "Python.h"
#include <vector>
#include <cuda_runtime.h>
#include "cutils_math.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "globalDefs.h"
#include "GPUArrayBasePair.h"
#include "GPUArrayTexBase.h"
using namespace std;



//for use in runtime loop, not general storage



template <class T>
class GPUArrayTexPair : public GPUArrayBasePair, public GPUArrayTexBase {
    public:
        GPUArrayTexDevice<T> d_data[2];
        vector<T> h_data;
        GPUArrayTexPair() {
        }
        GPUArrayTexPair(cudaChannelFormatDesc desc_) {
            for (int i=0; i<2; i++) {
                d_data[i] = GPUArrayTexDevice<T>(desc_);
            }
        }
        GPUArrayTexPair(vector<T> vals, cudaChannelFormatDesc desc_) {
            for (int i=0; i<2; i++) {
                d_data[i] = GPUArrayTexDevice<T>(vals.size(), desc_);
            }
            set(vals);
        }
        void set(vector<T> &other) {
            size = other.size();
            for (int i=0; i<2; i++) {
                d_data[i].resize(size);
            }
            h_data = other;
            h_data.reserve(d_data[0].capacity);

        }
        void dataToDevice() {
            d_data[activeIdx].set(h_data.data());
        }
        void dataToHost() {
            dataToHost(activeIdx);
        }
        void dataToHost(int idx) {
            d_data[idx].get(h_data.data());
        }
        void ensureSize() {
            for (int i=0; i<2; i++) {
                d_data[i].resize(h_data.size());
            }
        }
        void dataToHostAsync(cudaStream_t stream) {
            d_data[activeIdx].getAsync(h_data.data(), stream);
        }
        void copyToDeviceArray(void *dest) { //DEST HAD BETTER BE ALLOCATED
            int numBytes = size * sizeof(T);
            copyToDeviceArrayInternal(dest, d_data[activeIdx].d_data, numBytes);

        }
        cudaTextureObject_t getTex(int idx) {
            return d_data[idx].tex;
        }
        cudaSurfaceObject_t getSurf(int idx) {
            return d_data[idx].surf;
        }
        cudaTextureObject_t getTex() {
            return getTex(activeIdx);
        }
        cudaSurfaceObject_t getSurf() {
            return getSurf(activeIdx);
        }
        void memsetByVal(T val, int idx) {
            d_data[idx].memsetByVal(val);
        }
        void memsetByVal(T val) {
            memsetByVal(val, activeIdx);
        }
};



#endif
