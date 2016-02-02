#ifndef INTEGRATERLANGEVIN_H
#define INTEGRATERLANGEVIN_H
#include "IntegraterVerlet.h"
// void export_IntegraterVerlet();
class IntegraterLangevin : public IntegraterVerlet {

	void postForce(uint,int);
    public:
        IntegraterLangevin(SHARED(State));
        void run(int);
};
#endif
