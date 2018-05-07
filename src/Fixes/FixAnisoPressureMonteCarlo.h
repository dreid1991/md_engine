#pragma once
#ifndef FIXANISOPRESSUREMONTECARLO
#define FIXANISOPRESSUREMONTECARLO
#include "Fix.h"
#include "Interpolator.h"
#include "DataComputerPressure.h"
#include "DataComputerEnergy.h"
#include "DataComputerTemperature.h"

void export_FixAnisoPressureMonteCarlo();

namespace MD_ENGINE {
    class FixAnisoPressureMonteCarlo : public Fix {

    private:
        double nacc;
        double natt;
        int    naxis=0;
        bool useX=false;
        bool useY=false;
        bool useZ=false;
        int axisMap [3] = {-1,-1,-1};
        MD_ENGINE::DataComputerEnergy enrgComputer;
    public: 
        FixAnisoPressureMonteCarlo(boost::shared_ptr<State> state_, std::string handle_, 
        double px_=1.0, double sx_ = 0.001, 
        double py_=1.0 ,double sy_ = 0.001, 
        double pz_=1.0 ,double sz_ = 0.001, int applyEvery_ = 100,bool tune_ = false, int tuneFreq_ = 100);

        bool prepareFinal();
        bool stepFinal();
        bool postRun();
        DataComputerPressure pressureComputer;
        void setTempInterpolator();
        bool tune;
        real3 scale;
        real3 vScale;
        real3 targets;
        int tuneFreq;
        Interpolator *tempInterpolator;

    };
};
#endif
