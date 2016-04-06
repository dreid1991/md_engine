#pragma once
#ifndef READCONFIG_H
#define READCONFIG_H
#include "Python.h"
#include <pugixml.hpp>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include "globalDefs.h"
#include <string>
#include "boost_for_export.h"
void export_ReadConfig();
using namespace std;
class State;
class ReadConfig {
	string fn;
	State *state;
	bool haveReadYet;
	bool read();
	SHARED(pugi::xml_document) doc; //doing pointers b/c copy semantics for these are weird
	SHARED(pugi::xml_node) config;
	public:
        bool fileOpen;
        pugi::xml_node readNode(string nodeTag);
		ReadConfig(State *state_);
        ReadConfig() {
            state = (State *) NULL;
        }
        void loadFile(string); //change to bool or something to give feedback about if it's a file or not
		bool next();
		bool prev();
        bool moveBy(int);
		//bool readConfig(SHARED(State), string, int configIdx=0);
};

#endif
