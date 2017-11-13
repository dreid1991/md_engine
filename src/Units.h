#pragma once
#include <string>
#include "globalDefs.h"
enum UNITS {REAL, LJ};
void export_Units();
class Units {
public:
    real boltz;
    real hbar;
    real mvv_to_eng;
    real qqr_to_eng;
    real nktv_to_press;
    real ftm_to_v;
    real *dt; //points to state's dt
    double toSIDensity;
    int unitType;
    //assumung dialectric constant is 1

    Units(real *dt_) {
        unitType = -1;
        dt = dt_;
        setLJ();
    }

    void setLJ();
    void setReal();
};
