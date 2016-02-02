#ifndef INTEGRATERRELAX_H
#define INTEGRATERRELAX_H
#include "Integrater.h"
#include "cuda_call.h"
void export_IntegraterRelax();
class IntegraterRelax : public Integrater {
    public:
        double run(int, double);
        IntegraterRelax(SHARED(State));
};

#endif

