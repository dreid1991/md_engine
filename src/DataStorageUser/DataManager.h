#pragma once
#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include "boost_for_export.h"
#include <vector>
#include <string>
class DataSetTemperature;
class DataSetEnergy;
class DataSetBounds;
//#include "DataSet.h"
class DataSet;
class State;
void export_DataManager();
class DataManager {
    State *state;
	public:
		DataManager(){};
		DataManager(State *); 
        void generateSingleDataSetList();
/*! \brief Record temperature 
 *
 * behavior: if you call record for something that is already being recorded, it will change collect / collect generator of the existing object and continue append to the existing object's data */
        SHARED(DataSetTemperature) recordTemperature(std::string groupHandle, int collectEvery, boost::python::object collectGenerator); 
        void stopRecordTemperature(std::string groupHandle);
        std::vector<SHARED(DataSetTemperature)> dataSetsTemperature;

        SHARED(DataSetEnergy) recordEnergy(std::string groupHandle, int collectEvery, boost::python::object collectGenerator); 
        void stopRecordEnergy(std::string groupHandle);
        std::vector<SHARED(DataSetEnergy)> dataSetsEnergy;

        SHARED(DataSetBounds) recordBounds(int collectEvery, boost::python::object collectGenerator); 
        void stopRecordBounds();
        std::vector<SHARED(DataSetBounds)> dataSetsBounds;//no reason there should ever be more than one of these
        /* 
        void stopRecordTemp(string GroupHandle); // will fail if does not exist

        bool stopRecordEng(string groupHandle);
        */
        std::vector<DataSet *> dataSets; //to be generated each time run is called


        bool computingVirialsInForce; //so this is true if any fix or data set needs virials.  Those are all the things that could possibly need virials, so should never need to specifically ask to compute them
        int64_t turnLastEngs;

        void computeEngs();
};

#endif
