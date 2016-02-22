#include "DeviceManager.h"
#include "boost_for_export.h"

#include <iostream>
using namespace std;
DeviceManager::DeviceManager() {
    cudaGetDeviceCount(&nDevices);
    setDevice(nDevices-1);
}
bool DeviceManager::setDevice(int i) {
    if (i >= 0 and i < nDevices) {
        //add error handling here
        cout << "Setting device to " << i << endl;
        cudaSetDevice(i);
        cudaGetDeviceProperties(&prop, i);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        currentDevice = i;
        return true;
    }
    return false;
}
void export_DeviceManager() {
    class_<DeviceManager, boost::noncopyable>("DeviceManager", no_init)
        
        .def_readonly("nDevices", &DeviceManager::nDevices)
        .def("setDevice", &DeviceManager::setDevice)
        ;

}


