#include "FixPair.h"
#include "GPUArrayGlobal.h"
#include "State.h"

#include <cmath>

void FixPair::prepareParameters(string handle,
                            std::function<float (float, float)> fillFunction, std::function<float (float)> processFunction, bool fillDiag, std::function<float ()> fillDiagFunction) {
    GPUArrayGlobal<float> &array = *paramMap[handle];
    vector<float> *preproc = &paramMapPreproc[handle];
    int desiredSize = state->atomParams.numTypes;
    ensureParamSize(array);
    *preproc = array.h_data;
    if (fillDiag) {
        SquareVector::populateDiagonal<float>(&array.h_data, desiredSize, fillDiagFunction);
    }
    SquareVector::populate<float>(&array.h_data, desiredSize, fillFunction);
    SquareVector::process<float>(&array.h_data, desiredSize, processFunction);
    
    //okay, now ready to go to device!

}

void FixPair::resetToPreproc(string handle) {
    GPUArrayGlobal<float> &array = *paramMap[handle];
    array = paramMapPreproc[handle];
}

void FixPair::ensureParamSize(GPUArrayGlobal<float> &array) {

    int desiredSize = state->atomParams.numTypes;
    if (array.size() != desiredSize*desiredSize) {
        array = SquareVector::copyToSize(array.h_data,
                                         std::sqrt((double) array.size()),
                                         state->atomParams.numTypes);
    }
}

void FixPair::sendAllToDevice() {
    for (auto it=paramMap.begin(); it!=paramMap.end(); it++) {
        GPUArrayGlobal<float> &params = *it->second;
        params.dataToDevice();
    }
}

bool FixPair::setParameter(std::string param,
                           std::string handleA,
                           std::string handleB,
                           double val)
{
    int i = state->atomParams.typeFromHandle(handleA);
    int j = state->atomParams.typeFromHandle(handleB);
    if (paramMap.find(param) != paramMap.end()) {
        int numTypes = state->atomParams.numTypes;
        GPUArrayGlobal<float> &arr = *paramMap[param];
        ensureParamSize(arr);
        if (i>=numTypes or j>=numTypes or i<0 or j<0) {
            std::cout << "Tried to set param " << param
                      << " for invalid atom types " << handleA
                      << " and " << handleB
                      << " while there are " << numTypes
                      << " species." << std::endl;
            return false;
        }
        squareVectorRef<float>(arr.h_data.data(), numTypes, i, j) = val;
        squareVectorRef<float>(arr.h_data.data(), numTypes, j, i) = val;
    } 
    return false;
}

void FixPair::initializeParameters(std::string paramHandle,
                                   GPUArrayGlobal<float> &params) {
    ensureParamSize(params);
    paramMap[paramHandle] = &params;
    paramMapPreproc[paramHandle] = vector<float>();
}


std::string FixPair::restartChunkPairParams(string format) {
    std::stringstream ss;
    //ignoring format for now
    for (auto it=paramMap.begin(); it!=paramMap.end(); it++) {
        ss << "<" << it->first << ">\n";
        for (float x : it->second->h_data) {
            ss << x << "\n";
        }
        ss << "</" << it->first << ">\n";
    }
    return ss.str();
}
void export_FixPair() {
    boost::python::class_<FixPair,
                          boost::python::bases<Fix> > (
        "FixPair", boost::python::no_init  )
    .def("setParameter", &FixPair::setParameter,
            (boost::python::arg("param"),
             boost::python::arg("handleA"),
             boost::python::arg("handleB"),
             boost::python::arg("val"))
        )
    ;
}

