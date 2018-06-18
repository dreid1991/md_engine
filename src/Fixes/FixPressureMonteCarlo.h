#pragma once
#ifndef FIXPRESSUREMONTECARLO
#define FIXPRESSUREMONTECARLO
#include "Fix.h"
#include "Interpolator.h"
#include "DataComputerPressure.h"
#include "DataComputerEnergy.h"
#include "DataComputerTemperature.h"

void export_FixPressureMonteCarlo();

namespace MD_ENGINE {
    class FixPressureMonteCarlo : public Interpolator, public Fix {

    private:
        double nacc;
        double natt;
        int    nfake=0;
        MD_ENGINE::DataComputerEnergy enrgComputer;
    public: 
        FixPressureMonteCarlo(boost::shared_ptr<State> state_, std::string handle_, double pressure_, double scale_ = 0.001, int applyEvery_ = 100, bool tune_= false,int tuneFreq = 1000);

        bool prepareFinal();
        bool stepFinal();
        bool postRun();
        DataComputerPressure pressureComputer;
        void setTempInterpolator();
        real   scale;
        real   vScale;
        bool tune;
        int tuneFreq;
        Interpolator *tempInterpolator;

    };
};
#endif
