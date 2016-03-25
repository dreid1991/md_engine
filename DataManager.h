#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include "boost_for_export.h"
#include <vector>
#include <string>
class DataSetTemperature;
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
        /* 
        void stopRecordTemp(string GroupHandle); // will fail if does not exist

        bool stopRecordEng(string groupHandle);
        */
        std::vector<DataSet *> dataSets; //to be generated each time run is called
};

#endif
