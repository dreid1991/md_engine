#pragma once
#include <string>

class Units {
public:
    float boltz;
    float mvv_to_eng;
    float qqr_to_eng;
    float nktv_to_press;
    //assumung dialectric constant is 1

    Units() {
        setReal();
    }

    void setLJ();
    void setReal();
};
