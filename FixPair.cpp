#include "FixPair.h"
#include "State.h"

#include <cmath>
using namespace std;
namespace py = boost::python;
void FixPair::prepareParameters(string handle, std::function<float (float, float)> fillFunction, std::function<float (float)> processFunction, bool fillDiag, std::function<float ()> fillDiagFunction) {
    vector<float> &array = *paramMap[handle];
    vector<float> *preproc = &paramMapPreproc[handle];
    int desiredSize = state->atomParams.numTypes;
    ensureParamSize(array);
    *preproc = array;
    if (fillDiag) {
        SquareVector::populateDiagonal<float>(&array, desiredSize, fillDiagFunction);
    }
    SquareVector::populate<float>(&array, desiredSize, fillFunction);
    SquareVector::process<float>(&array, desiredSize, processFunction);
    
    //okay, now ready to go to device!

}

void FixPair::resetToPreproc(string handle) {
    vector<float> *array = paramMap[handle];
    vector<float> &preproc = paramMapPreproc[handle];
    *array = preproc;
}

void FixPair::ensureParamSize(vector<float> &array) {

    int desiredSize = state->atomParams.numTypes;
    if (array.size() != desiredSize*desiredSize) {
        vector<float> newVals = SquareVector::copyToSize(
                array,
                sqrt((double) array.size()),
                state->atomParams.numTypes
                );
        vector<float> *asPtr = &array;
        *(&array) = newVals;
    }
}

void FixPair::ensureOrderGivenForAllParams() {
    for (auto it=paramMap.begin(); it!=paramMap.end(); it++) {
        string handle = it->first;
        if (find(paramOrder.begin(), paramOrder.end(), handle) == paramOrder.end()) {
            mdError("Order for all parameters not specified");
        }
    }
}
void FixPair::sendAllToDevice() {
    ensureOrderGivenForAllParams();
    int totalSize = 0;
    for (auto it = paramMap.begin(); it!=paramMap.end(); it++) {
        totalSize += it->second->size(); 
    }
    paramsCoalesced = GPUArrayDeviceGlobal<float>(totalSize);
    int runningSize = 0;
    for (string handle : paramOrder) {
        vector<float> &vals = *(paramMap[handle]);
        paramsCoalesced.set(vals.data(), runningSize, vals.size());
        runningSize += vals.size();
    }
}

bool FixPair::setParameter(string param,
                           string handleA,
                           string handleB,
                           double val)
{
    int i = state->atomParams.typeFromHandle(handleA);
    int j = state->atomParams.typeFromHandle(handleB);
    if (paramMap.find(param) != paramMap.end()) {
        int numTypes = state->atomParams.numTypes;
        vector<float> &arr = *(paramMap[param]);
        ensureParamSize(arr);
        if (i>=numTypes or j>=numTypes or i<0 or j<0) {
            cout << "Tried to set param " << param
                      << " for invalid atom types " << handleA
                      << " and " << handleB
                      << " while there are " << numTypes
                      << " species." << endl;
            return false;
        }
        squareVectorRef<float>(arr.data(), numTypes, i, j) = val;
        squareVectorRef<float>(arr.data(), numTypes, j, i) = val;
    } 
    return false;
}

void FixPair::initializeParameters(string paramHandle,
                                   vector<float> &params) {
    ensureParamSize(params);
    paramMap[paramHandle] = &params;
    paramMapPreproc[paramHandle] = vector<float>();
}


string FixPair::restartChunkPairParams(string format) {
    stringstream ss;
    //ignoring format for now
    for (auto it=paramMap.begin(); it!=paramMap.end(); it++) {
        ss << "<" << it->first << ">\n";
        for (float x : *(it->second)) {
            ss << x << "\n";
        }
        ss << "</" << it->first << ">\n";
    }
    return ss.str();
}
void export_FixPair() {
    py::class_<FixPair,
    py::bases<Fix> > (
            "FixPair", py::no_init  )
        .def("setParameter", &FixPair::setParameter,
                (py::arg("param"),
                 py::arg("handleA"),
                 py::arg("handleB"),
                 py::arg("val"))
            )
        ;
}

