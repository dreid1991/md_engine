#ifndef FIX_PAIR_H
#define FIX_PAIR_H
#include <climits>
#define DEFAULT_FILL INT_MAX
#include "AtomParams.h"
#include <map>
#include "GPUArray.h"
#include "Fix.h"
#include "xml_func.h"
class State;
using namespace std;
template <class T>
__host__ __device__ T squareVectorItem(T *vals, int nCol, int i, int j) {
    return vals[i*nCol + j];
}

template <class T>
__host__ __device__ T &squareVectorRef(T *vals, int nCol, int i, int j) {
    return vals[i*nCol + j];
}
namespace SquareVector {
    template <class T>
        vector<T> create(int size) {
            return vector<T>(size*size, DEFAULT_FILL);
        }
    template <class T>
        void populate(vector<T> *vec, int size, std::function<T (T, T)> fillFunction) {
            for (int i=0; i<size; i++) {
                for (int j=0; j<size; j++) {
                    T val = squareVectorRef<T>(vec->data(), size, i, j);
                    if (i==j) {
                        if (val == DEFAULT_FILL) {
                            cout << "You have not defined interaction parameters for atom type with index " << i << endl;
                            assert(val != DEFAULT_FILL);
                        }
                    } else if (val == DEFAULT_FILL) {
                        squareVectorRef<T>(vec->data(), size, i, j) = fillFunction(squareVectorRef<T>(vec->data(), size, i, i), squareVectorRef<T>(vec->data(), size, j, j));
                    }
                }
            }
        }
    template <class T>
        vector<T> copyToSize(vector<T> &other, int oldSize, int newSize) {
            vector<T> replacement(newSize*newSize, DEFAULT_FILL);
            int copyUpTo = fmin(oldSize, newSize);
            for (int i=0; i<copyUpTo; i++) {
                for (int j=0; j<copyUpTo; j++) {
                    squareVectorRef<T>(replacement.data(), newSize, i, j) = squareVectorItem<T>(other.data(), oldSize, i, j);
                }
            }
            return replacement;

        }
}
/*
template <class T>
class SquareVector{
    public:
        vector<T> vals;
        int size;
        
        SquareVector(int n) : size(n) {
            vals = vector<T>(size*size, DEFAULT_FILL);

        }
        T &operator () (int row, int col) {
            return vals[row*size + col];
        }
        void populate(function<T (T, T)> fillFunction) {
            for (int i=0; i<size; i++) {
                for (int j=0; j<size; j++) {
                    T val = (*this)(i, j);
                    if (i==j) {
                        if (val == DEFAULT_FILL) {
                            cout << "You have not defined interaction parameters for atom type with index " << i << endl;
                            assert(val != DEFAULT_FILL);
                        }
                    } else if (val == DEFAULT_FILL) {
                    (*this)(i, j) = fillFunction((*this)(i, i), (*this)(j, j));
                    }
                }
            }
        }
        void setSize(int n) {
            vector<T> old = vals;
            if (n != size) {
                int sizeOld = size;
                size = n;
                vals = vector<T>(size*size, DEFAULT_FILL);
                int copyUpTo = fmin(size, sizeOld);
                for (int i=0; i<copyUpTo; i++) {
                    for (int j=0; j<copyUpTo; j++) {
                        (*this)(i, j) = squareArrayItem<T>(old.data(), sizeOld, i, j);
                    }
                }
            }
        }
        int totalSize() {
            return size*size;
        }
};
*/
class FixPair : public Fix {
    protected:
        map<string, GPUArray<float> *> paramMap;
        void labelArray(string label, GPUArray<float> &arr) {
            paramMap[label] = &arr;
        }
        void initializeParameters(string paramHandle, GPUArray<float> &);
        void prepareParameters(GPUArray<float> &, std::function<float (float, float)> fillFunction);
        void sendAllToDevice();
        void ensureParamSize(GPUArray<float> &);
        bool readPairParams(pugi::xml_node);
        string restartChunkPairParams(string format);
    public:
        //bool setParameter(string, int, int, double);
        bool setParameter(string, string, string, double);
        FixPair(SHARED(State) state_, string handle_, string groupHandle_, string type_, int applyEvery_) : Fix(state_, handle_, groupHandle_, type_, applyEvery_) {};

    /*
    GPUArrayDevice<float> copySqrToDevice(SquareVector<float> &vec) {
        GPUArrayDevice<float> arr (vec.totalSize());
        arr.set(vec.data());
        return arr;
    }
    */
};
#endif
