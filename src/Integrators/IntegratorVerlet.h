#pragma once
#ifndef INTEGRATORVERLET_H
#define INTEGRATORVERLET_H

#include "globalDefs.h"
#include "Integrator.h"

void export_IntegratorVerlet();
class IntegratorVerlet : public Integrator {
    protected:
	void preForce(uint);
	void postForce(uint);
    public:
        IntegratorVerlet(SHARED(State));
        void run(int);
};
#endif
