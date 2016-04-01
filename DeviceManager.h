#pragma once
#include "Python.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
void export_DeviceManager();

class DeviceManager {
    public:
        int nDevices;
        DeviceManager();
        cudaDeviceProp prop;
        bool setDevice(int, bool output=true);
        int currentDevice;
};
