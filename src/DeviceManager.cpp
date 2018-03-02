#include "DeviceManager.h"
#include "boost_for_export.h"

#include <iostream>
namespace py = boost::python;
DeviceManager::DeviceManager() {
    cudaGetDeviceCount(&nDevices);
    setDevice(nDevices-1);
}
bool DeviceManager::setDevice(int i, bool output) {
    if (i >= 0 and i < nDevices) {
        //add error handling here
        cudaSetDevice(i);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#ifdef DASH_DOUBLE
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#else
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#endif
        cudaGetDeviceProperties(&prop, i);
        currentDevice = i;
        if (output) {
            std::cout << "Selecting device " << i<<" " << prop.name << std::endl;
        }
        return true;
    }
    return false;
}
void export_DeviceManager() {
    py::class_<DeviceManager, boost::noncopyable>("DeviceManager", py::no_init)
        
        .def_readonly("nDevices", &DeviceManager::nDevices)
        .def_readonly("currentDevice", &DeviceManager::currentDevice)
        .def("setDevice", &DeviceManager::setDevice, (py::arg("i"), py::arg("output")=true ))
        ;

}


