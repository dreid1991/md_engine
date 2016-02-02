#ifndef GPUARRAYTEXDEVICE_H
#define GPUARRAYTEXDEVICE_H
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
#include "memset_defs.h"
using namespace std;


//for use in runtime loop, not general storage

void MEMSETFUNC(cudaSurfaceObject_t, void *, int, int);

template <class T>
class GPUArrayTexDevice : public GPUArrayTexBase {
    public:
        
        cudaArray *d_data;
        //int capacity;
        int size;
        int capacity;
        int Tsize;
        cudaTextureObject_t tex;
        cudaSurfaceObject_t surf;
        cudaResourceDesc resDesc;
        cudaTextureDesc texDesc;
        bool madeTex;
        void destroyDevice() {
            if (madeTex) {
                //cout << "destroy texture objects for " << this << endl;
                CUCHECK(cudaDestroyTextureObject(tex));
                CUCHECK(cudaDestroySurfaceObject(surf));
            }
            //cout << "d_data is " << d_data << endl;
            if (d_data != (cudaArray *) NULL) {
              //  cout << "and I'm destroying" << endl;
                CUCHECK(cudaFreeArray(d_data));
            }
            madeTex = false;
        }
        void initializeDescriptions() {
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            //.res.array.array is unset.  Set when allocing on device
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.readMode = cudaReadModeElementType;
        }
        GPUArrayTexDevice() : madeTex(false) {
            //cout << "default constructor " << endl;
            //cout << this << endl;
            d_data = (cudaArray *) NULL;
            size = 0;
            capacity = 0;
            Tsize = sizeof(T);
        }
        GPUArrayTexDevice(cudaChannelFormatDesc desc_) : madeTex(false) {
            //cout << "desc constructor " << endl;
            //cout << this << endl;
            d_data = (cudaArray *) NULL;
            channelDesc = desc_;
            initializeDescriptions();
            size = 0;
            capacity = 0;
            Tsize = sizeof(T);
        }
        GPUArrayTexDevice(int size_, cudaChannelFormatDesc desc_) : madeTex(false) {
            //cout << "number, desc constructor" << endl;
            //cout << this << endl;
            size = size_;
            channelDesc = desc_;
            initializeDescriptions();
            allocDevice();
            createTexSurfObjs();
            Tsize = sizeof(T);
        }
        ~GPUArrayTexDevice() {
            //cout << "in destructor!" << endl<<this<<endl;cout.flush();
            
            destroyDevice();
        }

        GPUArrayTexDevice(const GPUArrayTexDevice<T> &other) { //copy constructor
            //cout << this << endl;
            channelDesc = other.channelDesc;
            size = other.size;
            capacity = other.capacity;
            initializeDescriptions();
            allocDevice();
            CUCHECK(cudaMemcpy2DArrayToArray(d_data, 0, 0, other.d_data, 0, 0, NX() * sizeof(T), NY(), cudaMemcpyDeviceToDevice));
            createTexSurfObjs();
            Tsize = sizeof(T);


        }
        GPUArrayTexDevice<T> &operator=(const GPUArrayTexDevice<T> &other) { //copy assignment
            //cout << this << endl;
            channelDesc = other.channelDesc;
            if (other.size) {
                resize(other.size); //creates tex surf objs
            }
            int x = NX();
            int y = NY();
            CUCHECK(cudaMemcpy2DArrayToArray(d_data, 0, 0, other.d_data, 0, 0, x*sizeof(T), y, cudaMemcpyDeviceToDevice));
            return *this;
        }
        void copyFromOther(const GPUArrayTexDevice<T> &other) {
            //I should own no pointers at this point, am just copying other's
            channelDesc = other.channelDesc;
            size = other.size;
            capacity = other.capacity;
            d_data = other.d_data;


        }
        void nullOther(GPUArrayTexDevice<T> &other) {
            other.d_data = (cudaArray *) NULL;
            other.size = 0;
            other.capacity = 0;

        }
        GPUArrayTexDevice(GPUArrayTexDevice<T> &&other) { //move constructor;
            //cout << "move constructor" << endl;
            //cout << this << endl;
            copyFromOther(other);
            d_data = other.d_data;
            initializeDescriptions(); 
            resDesc.res.array.array = d_data;
            if (other.madeTex) {
                createTexSurfObjs();
            }
            nullOther(other);
            Tsize = sizeof(T);
        }
        GPUArrayTexDevice<T> &operator=(GPUArrayTexDevice<T> &&other) { //move assignment
            //cout << "move assignment" << endl;
            //cout << "from " << &other << " to " << this << endl;
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
        void createTexSurfObjs() {

            tex = 0;
            surf = 0;
            cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
            cudaCreateSurfaceObject(&surf, &resDesc);
            madeTex = true;
        }

        int NX() {
            return fmin((int) (PERLINE/sizeof(T)), (int) size);
        }
        int NY() {
            return ceil(size / (float) (PERLINE/sizeof(T)));
        }
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
        void allocDevice() {
            int x = NX();
            int y = NY();
            CUCHECK(cudaMallocArray(&d_data, &channelDesc, x, y) );
            capacity = x*y;
            //assuming address gets set in blocking manner
            resDesc.res.array.array = d_data;
        }
        T *get(T *copyTo) {
            int x = NX();
            int y = NY();

            if (copyTo == (T *) NULL) {
                copyTo = (T *) malloc(x*y*sizeof(T));
            }
            CUCHECK(cudaMemcpy2DFromArray(copyTo, x * sizeof(T), d_data, 0, 0, x * sizeof(T), y, cudaMemcpyDeviceToHost));
            return copyTo;
        }
        void set(T *copyFrom) {
            int x = NX();
            int y = NY();
            cudaMemcpy2DToArray(d_data, 0, 0, copyFrom, x*sizeof(T), x * sizeof(T), y, cudaMemcpyHostToDevice );
        }
        T *getAsync(T *copyTo, cudaStream_t stream) {
            int x = NX();
            int y = NY();

            if (copyTo == (T *) NULL) {
                copyTo = (T *) malloc(x*y*sizeof(T));
            }
            CUCHECK(cudaMemcpy2DFromArrayAsync(copyTo, x * sizeof(T), d_data, 0, 0, x * sizeof(T), y, cudaMemcpyDeviceToHost, stream));
            return copyTo;

        }
        void copyToDeviceArray(void *dest) { //DEST HAD BETTER BE ALLOCATED
            int numBytes = size * sizeof(T);
            copyToDeviceArrayInternal(dest, d_data, numBytes);

        }
        void memsetByVal(T val_) {
            assert(Tsize==4 or Tsize==8 or Tsize==16);
            MEMSETFUNC(surf, &val_, size, Tsize);
        }
};

#endif
