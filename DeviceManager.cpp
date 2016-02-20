#include "DeviceManager.h"
#include "boost_for_export.h"


DeviceManager::DeviceManager() {
    cudaGetDeviceCount(&nDevices);
    setDevice(0);
}
bool DeviceManager::setDevice(int i) {
    if (i >= 0 and i < nDevices) {
        cudaSetDevice(i);
        cudaGetDeviceProperties(&prop, i);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
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


