#ifndef FIXLJCUT_H
#define FIXLJCUT_H
#include "FixPair.h"
#include "xml_func.h"
void export_FixLJCut(); //make there be a pair base class in boost!
class FixLJCut : public FixPair {
    public:
        //rcut is defined in state, because that has to be used for neighborlist building
        const string epsHandle;
        const string sigHandle;
        GPUArray<float> epsilons;
        GPUArray<float> sigmas;
        FixLJCut(SHARED(State), string handle, string groupHandle);
        void compute();
        bool prepareForRun();
        string restartChunk(string format);
        bool readFromRestart(pugi::xml_node);
        void addSpecies(string handle);
};
#endif
