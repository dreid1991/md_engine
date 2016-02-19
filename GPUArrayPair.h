#ifndef GPUARRAYPAIR_H
#define GPUARRAYPAIR_H
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
#include "GPUArrayDevice.h"
using namespace std;




//okay, going to test this in isloation first, then make it take texture / surface as an argument



template <class T>
class GPUArrayPair : public GPUArrayBasePair {

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
        vector<T> h_data;
        GPUArrayDevice<T> d_data[2];
        GPUArrayPair() : GPUArrayBasePair() {
            size = 0;
        }
        GPUArrayPair(vector<T> &vals) {
            set(vals);
            if (!vals.size()) { //should alloc anyway
                for (int i=0; i<2; i++) {
                    d_data[i] = GPUArrayDevice<T>(vals.size());
                }

            }
        }
        T *getDevData(int n) {
            return d_data[n].ptr;
        }
        T *getDevData() {
            return getDevData(activeIdx);
        }
        bool set(vector<T> &other) {
            if (other.size() < size) {
                setHostCanFit(other);
                return true;
            } else {
                for (int i=0; i<2; i++) {
                    d_data[i] = GPUArrayDevice<T>(other.size());
                }
                setHost(other);
            }
            return false;

        }
        T *operator ()(int n) {
            return getDevData(n);
        }
        void dataToDevice() {
            CUCHECK(cudaMemcpy(d_data[activeIdx].ptr, h_data.data(), size*sizeof(T), cudaMemcpyHostToDevice ));

        }
        void dataToHost() {
            dataToHost(activeIdx);
        }      
        void dataToHost(int idx) {
            CUCHECK(cudaMemcpy(h_data.data(), d_data[idx].ptr, size*sizeof(T), cudaMemcpyDeviceToHost));
        }      
        void copyToDeviceArray(void *dest) {
            CUCHECK(cudaMemcpy(dest, d_data[activeIdx].ptr, size*sizeof(T), cudaMemcpyDeviceToDevice));
        }
        void memsetByVal(T val, int idx) {
            d_data[idx].memsetByVal(val);
        }
        void memsetByVal(T val) {
            memsetByVal(val, activeIdx);
        }


};
#endif
