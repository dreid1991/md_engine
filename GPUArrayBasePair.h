#pragma once
#ifndef GPUARRAYBASEPAIR_H
#define GPUARRAYBASEPAIR_H
#include "GPUArrayBase.h"
class GPUArrayBasePair : public GPUArrayBase {
    public:
        unsigned int activeIdx;
        unsigned int switchIdx() {
            activeIdx = !activeIdx;
            return activeIdx;
        }
        GPUArrayBasePair() {
            activeIdx = 0;
        }
};
#endif

