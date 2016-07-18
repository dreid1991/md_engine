#pragma once
#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include "boost_for_export.h"
#include <vector>
#include <string>
class DataSetUser;
class DataComputer;
class State;
void export_DataManager();
class DataManager {
    State *state;
	public:
		DataManager(){};
		DataManager(State *); 
        boost::shared_ptr<DataSetUser> createDataSet(boost::shared_ptr<DataComputer> comp, uint32_t groupTag, int dataMode, int dataType, int interval, boost::python::object collectGenerator);
        boost::shared_ptr<DataSetUser> recordTemperature(std::string groupHandle, int interval, boost::python::object collectGenerator); 
        void stopRecord(boost::shared_ptr<DataSetUser>);
        //std::vector<SHARED(DataSetTemperature)> dataSetsTemperature;

        //SHARED(DataSetEnergy) recordEnergy(std::string groupHandle, int collectEvery, boost::python::object collectGenerator); 
        //void stopRecordEnergy(std::string groupHandle);
        //std::vector<SHARED(DataSetEnergy)> dataSetsEnergy;

        //SHARED(DataSetBounds) recordBounds(int collectEvery, boost::python::object collectGenerator); 
        //void stopRecordBounds();
        //std::vector<SHARED(DataSetBounds)> dataSetsBounds;//no reason there should ever be more than one of these
        /* 
        void stopRecordTemp(string GroupHandle); // will fail if does not exist

        bool stopRecordEng(string groupHandle);
        */
        std::vector<boost::shared_ptr<DataSetUser> > dataSets;  //to be continually maintained


        int64_t turnLastEngs;

        void computeEnergy();
};

#endif
