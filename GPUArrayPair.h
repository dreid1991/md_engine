#pragma once
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
    }
    public:
        vector<T> h_data;
        GPUArrayDevice<T> d_data[2];
        GPUArrayPair() : GPUArrayBasePair() {
        }
        GPUArrayPair(vector<T> &vals) {
            set(vals);
            for (int i=0; i<2; i++) {
                d_data[i] = GPUArrayDevice<T>(vals.size());
            }
        }
        T *getDevData(int n) {
            return d_data[n].data();
        }
        T *getDevData() {
            return getDevData(activeIdx);
        }
        bool set(vector<T> &other) {
            if (other.size() < size()) {
                setHost(other);
                return true;
            } else {
                for (int i=0; i<2; i++) {
                    d_data[i] = GPUArrayDevice<T>(other.size());
                }
                setHost(other);
            }
            return false;

        }

        size_t size() const { return h_data.size(); }

        T *operator ()(int n) {
            return getDevData(n);
        }
        void dataToDevice() {
            CUCHECK(cudaMemcpy(d_data[activeIdx].data(), h_data.data(), size()*sizeof(T), cudaMemcpyHostToDevice ));

        }
        void dataToHost() {
            dataToHost(activeIdx);
        }      
        void dataToHost(int idx) {
            CUCHECK(cudaMemcpy(h_data.data(), d_data[idx].data(), size()*sizeof(T), cudaMemcpyDeviceToHost));
        }      
        void copyToDeviceArray(void *dest) {
            CUCHECK(cudaMemcpy(dest, d_data[activeIdx].data(), size()*sizeof(T), cudaMemcpyDeviceToDevice));
        }
        bool copyBetweenArrays(int dst, int src) {
            if (dst != src) {
                CUCHECK(cudaMemcpy(d_data[dst].data(), d_data[src].data(), size()*sizeof(T), cudaMemcpyDeviceToDevice));
                return true;
            }
            return false;
        }
        void memsetByVal(T val, int idx) {
            d_data[idx].memsetByVal(val);
        }
        void memsetByVal(T val) {
            memsetByVal(val, activeIdx);
        }


};
#endif
