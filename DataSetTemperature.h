#pragma once

#include "DataSet.h"

class DataSetTemperature : public DataSet {
    public:
        DataSetTemperature(uint32_t);
        DataSetTemperature();
        vector<double> vals;

};
