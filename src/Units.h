#pragma once
#include <string>
void export_Units();
class Units {
public:
    float boltz;
    float mvv_to_eng;
    float qqr_to_eng;
    float nktv_to_press;
    float ftm_to_v;
    //assumung dialectric constant is 1

    Units() {
        setReal();
    }

    void setLJ();
    void setReal();
};
