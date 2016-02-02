#ifndef GPUARRAYBASE_H
#define GPUARRAYBASE_H
#include <vector>
using namespace std;

class GPUArrayBase {
    public:
        int size;
        virtual void dataToDevice(){};
        virtual void dataToHost(){};
        GPUArrayBase(){};
};
#endif
