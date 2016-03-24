#ifndef INTEGRATER_H
#define INTEGRATER_H
#include "Atom.h"
#include <vector>
//#include "Fix.h"
#include <chrono>
#include "GPUArray.h"
#include "GPUArrayTex.h"
#include "boost_for_export.h"
#include <future>
void export_Integrater();
class State;


extern const string IntVerletType;
extern const string IntRelaxType;

class Integrater {

    protected:
        //virtual void preForce(uint);
        void force(bool);
        //virtual void postForce(uint);
        void asyncOperations();
        std::vector<GPUArrayBase *> activeData;
        void basicPreRunChecks();
        void basicPrepare(int);
        void basicFinish();
        void setActiveData();
        void doDataCollection();
        void singlePointEng(); //make a python-wrapped version
        void writeOutput();
        std::future<void> dataGather;
        public:
                string type;
                State *state;
                Integrater() {};
                Integrater(State *state_, string type_);
        
                //double relax(int numTurns, num fTol);
                void forceSingle(bool);
/*	void verletPreForce(vector<Atom *> &atoms, double timestep);
	void verletPostForce(vector<Atom *> &atoms, double timestep);
	void compute(vector<Fix *> &, int);
	void firstTurn(Run &params);
	void run(Run &params, int currentTurn, int numTurns);
	bool rebuildIsDangerous(vector<Atom *> &atoms, double);
	void addKineticEnergy(vector<Atom *> &, Data &);
	void setThermoValues(Run &);*/
};


#endif
