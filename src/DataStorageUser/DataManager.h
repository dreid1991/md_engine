#pragma once
#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#include "boost_for_export.h"
#include <vector>
#include <string>
#include <set>
class State;
void export_DataManager();
namespace MD_ENGINE {
class DataSetUser;
class DataComputer;

class DataManager {
    State *state;
	public:
		DataManager(){};
		DataManager(State *); 
        boost::shared_ptr<DataSetUser> createDataSet(boost::shared_ptr<DataComputer> comp, uint32_t groupTag, int dataMode, int dataType, int interval, boost::python::object collectGenerator);

        boost::shared_ptr<MD_ENGINE::DataSetUser> recordTemperature(std::string groupHandle, int interval, boost::python::object collectGenerator); 
        boost::shared_ptr<MD_ENGINE::DataSetUser> recordEnergy(std::string groupHandle, int collectEvery, boost::python::object collectGenerator); 
        boost::shared_ptr<MD_ENGINE::DataSetUser> recordPressure(std::string groupHandle, int collectEvery, boost::python::object collectGenerator); 
        boost::shared_ptr<MD_ENGINE::DataSetUser> recordBounds(int collectEvery, boost::python::object collectGenerator); 

        void stopRecord(boost::shared_ptr<MD_ENGINE::DataSetUser>);


        //SHARED(DataSetBounds) recordBounds(int collectEvery, boost::python::object collectGenerator); 
        //std::vector<SHARED(DataSetBounds)> dataSetsBounds;//no reason there should ever be more than one of these

        std::vector<boost::shared_ptr<DataSetUser> > dataSets;  //to be continually maintained


        int64_t turnLastEngs;

        void computeEnergy();

        void addVirialTurn(int64_t);
        void clearVirialTurn(int64_t);
        std::set<int64_t> virialTurns;//!<Turns on which virial coefs will be calculated
        int64_t turnVirialsComputed;//!<Turn virial coefs last computed
//!flag for if fixes compute virials in the forst kernels or not.  Is true if any data or fixes need them


};

}
#endif
