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
        MD_ENGINE::DataComputerEnergy enrgComputer;
    public: 
        FixPressureMonteCarlo(boost::shared_ptr<State> state_, std::string handle_, double pressure_, double scale_ = 0.01, int applyEvery_ = 1000, int tuneFreq = 10);

        bool prepareFinal();
        bool stepFinal();
        bool postRun();
        DataComputerPressure pressureComputer;
        void setTempInterpolator();
        float   scale;
        float   vScale;
        int tuneFreq;
        Interpolator *tempInterpolator;

    };
};
#endif
