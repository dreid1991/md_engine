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
        cudaSetDevice(i);
        cudaGetDeviceProperties(&prop, i);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        currentDevice = i;
        cout << "Switching to device " << prop.name << endl;
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


