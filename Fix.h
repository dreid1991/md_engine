#ifndef FIX_H
#define FIX_H

#include "Python.h"
#include "globalDefs.h"
#include "Atom.h"
#include "list_macro.h"
#include <iostream>
#include "GPUArray.h"
#include "GPUArrayTex.h"
#include "State.h"
#include "FixTypes.h"
#include <pugixml.hpp>


#include "boost_for_export.h"
void export_Fix();
//#include "DataManager.h"
using namespace std;
class Fix {
	public:
//        SHARED(State) state_shr;
		State *state;
		string handle;
		string groupHandle;
		int applyEvery;
		unsigned int groupTag;
		string type;
		Fix() {};	
		Fix(SHARED(State) state_, string handle_, string groupHandle_, string type_, int applyEvery_);
		virtual void compute(){};
		bool isEqual(Fix &);
		bool forceSingle;
        int orderPreference;
        void updateGroupTag();
        virtual bool dataToDevice() {return true;};
        virtual bool dataToHost() {return true;};
        //prepareForRun and downloadFromRun ARE called before a run.  data to device and host and NOT. prepare from run should call those if needed
        virtual void addSpecies(string handle) {};
        virtual bool prepareForRun() {return true;};
        virtual bool downloadFromRun() {return true;};
        virtual ~Fix() {};
        virtual bool refreshAtoms(){return true;};
        virtual bool readFromRestart(pugi::xml_node restData){return true;};
        virtual string restartChunk(string format){return "";};
        //virtual vector<pair<int, vector<int> > > neighborlistExclusions();
        const string restartHandle;
        // TODO: think about treatment of different kinds of bonds in fixes
        // right now for ease, each vector of bonds in any given fix that stores
        // bonds has to store them in a vector<BondVariant> variable
        // you can push_back, insert, whatever, other kinds of bonds into this
        // vector
        // you have to get them out using a getBond method, or using the
        // boost::get<BondType>(vec) syntax
        // it's not perfect, but it lets us generically collect vectors without
        // doing any copying
        virtual vector<BondVariant> *getBonds() {
            return nullptr;
        }
        void validAtoms(vector<Atom *> &atoms);
	
};

/*
do it with precompiler instructions, lol!
nah, just do methods of state.  Might have to add other function calls later as fixes become more complicated
*/
#endif
