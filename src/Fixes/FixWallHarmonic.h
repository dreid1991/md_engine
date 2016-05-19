#pragma once
#ifndef FIXWALLHARMONIC_H
#define FIXWALLHARMONIC_H

#include "Fix.h"

void export_FixWallHarmonic();

class FixWallHarmonic : public Fix {

public:
    FixWallHarmonic(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_,
                    Vector origin_, Vector forceDir_, double dist_, double k_);
    Vector origin;
    Vector forceDir;
    double dist;
    double k;
    void compute(bool);
};

#endif
