#ifndef INTEGRATERVERLET_H
#define INTEGRATERVERLET_H

#include "globalDefs.h"
#include "Integrater.h"

void export_IntegraterVerlet();
class IntegraterVerlet : public Integrater {
    protected:
	void preForce(uint);
	void postForce(uint);
    public:
        IntegraterVerlet(SHARED(State));
        void run(int);
};
#endif
