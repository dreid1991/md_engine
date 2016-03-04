#include "FixPair.h"
#include "State.h"
void FixPair::prepareParameters(GPUArray<float> &array, std::function<float (float, float)> fillFunction) {
    int desiredSize = state->atomParams.numTypes;
    ensureParamSize(array);
    SquareVector::populate<float>(&array.h_data, desiredSize, fillFunction);
    //okay, now ready to go to device!

}

void FixPair::ensureParamSize(GPUArray<float> &array) {

    int desiredSize = state->atomParams.numTypes;
    if (array.size() != desiredSize*desiredSize) {
        vector<float> newVals = SquareVector::copyToSize(array.h_data, sqrt((double) array.size()), state->atomParams.numTypes);
        array.set(newVals);
    }
}

void FixPair::sendAllToDevice() {
    for (auto it=paramMap.begin(); it!=paramMap.end(); it++) {
        GPUArray<float> &params = *it->second;
        params.dataToDevice();
    }
}

bool FixPair::setParameter(string param, string handleA, string handleB, double val) {
    int i = state->atomParams.typeFromHandle(handleA);
    int j = state->atomParams.typeFromHandle(handleB);
    if (paramMap.find(param) != paramMap.end()) {
        int numTypes = state->atomParams.numTypes;
        GPUArray<float> &arr = *paramMap[param];
        ensureParamSize(arr);
        if (i>=numTypes or j>=numTypes or i<0 or j<0) {
            cout << "Tried to set param " << param << " for invalid atom types " << handleA << " and " << handleB << " while there are " << numTypes << " species." << endl;
            return false;
        }
        squareVectorRef<float>(arr.h_data.data(), numTypes, i, j) = val;
        squareVectorRef<float>(arr.h_data.data(), numTypes, j, i) = val;
    } 
    return false;
}

void FixPair::initializeParameters(string paramHandle, GPUArray<float> &params) {
    ensureParamSize(params);
    labelArray(paramHandle, params);
}


string FixPair::restartChunkPairParams(string format) {
    stringstream ss;
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


