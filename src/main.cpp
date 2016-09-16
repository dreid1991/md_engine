#include <vector>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>

#include "Python.h"
//#include "cutils_func.h"
//#include "mpi.h"

#include "GPUArrayTex.h"
#include "GPUArrayGlobal.h"
#include "globalDefs.h"

#include "State.h"

#include "FixBondHarmonic.h"
#include "FixPair.h"
#include "includeFixes.h"

#include "InitializeAtoms.h"

#include "IntegratorVerlet.h"
#include "IntegratorRelax.h"

#include "FixChargePairDSF.h"

#include "WriteConfig.h"
#include "ReadConfig.h"

using namespace std;


void testFire() {
    SHARED(State) state = SHARED(State) (new State());
    int baseLen = 50;
    state->shoutEvery = 100;
    double mult = 1.5;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(mult*baseLen, mult*baseLen, 10));
    state->rCut = 3.5;
    state->atomParams.addSpecies("handle", 2);
    state->is2d = true;
    state->periodic[2] = false;
    double eps=0.02;
    std::srand(2);
    for (int i=0; i<baseLen; i++) {
        for (int j=0; j<baseLen; j++) {
            state->addAtom("handle",Vector(i*mult+eps*(double(std::rand())/RAND_MAX), j*mult+eps*(double(std::rand())/RAND_MAX), 0), 0);
        }
    }

    state->periodicInterval = 9;
    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
    state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);
    cout << "last" << endl;
    cout << state->atoms[0].pos[0]<<' '<<state->atoms[0].vel[0]<<' '<<state->atoms[0].force[0]<< endl;

    SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handle", "xml", 10000));
    state->activateWriteConfig(write);

    state->dt=0.003;
    IntegratorRelax integratorR(state);
    integratorR.run(400000,1.0);
    //cout << state->atoms[0].pos[0]<<' '<<state->atoms[0].vel[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[0].forceLast[0]<< endl;

}


void testCharge() {
    SHARED(State) state = SHARED(State) (new State());
    int baseLen = 25;
    int polybeads = 1;
    double b = 0.3;
    state->shoutEvery = 1000;
    double mult = 3.25;
    state->bounds = Bounds(state,
                           Vector(0, 0, 0),
                           Vector(mult*baseLen, mult*baseLen, mult*baseLen));
    state->rCut = 9.5;
    state->atomParams.addSpecies("anion", 2);
    state->atomParams.addSpecies("cation", 3);
    double eps = 2.0;
    std::srand(2);

    for (int i=0; i<baseLen; i++) {
        for (int j=0; j<baseLen; j++) {
            int k=0;
            Vector ppos = Vector(i*mult + eps*(double(std::rand())/RAND_MAX),
                                 j*mult + eps*(double(std::rand())/RAND_MAX),
                                 double(k));
            if ((i*baseLen+j)%2) state->addAtom("anion", ppos, 0);
            else state->addAtom("cation", ppos, 0);
            for (int l=1; l<polybeads; l++) {
                float3 a = make_float3((0.5 - double(std::rand())/RAND_MAX),
                                       (0.5 - double(std::rand())/RAND_MAX),
                                       (double(std::rand())/RAND_MAX));
                a = normalize(a)*b;
                ppos = ppos+Vector(a);
                if ((i*baseLen + j) % 2) state->addAtom("anion", ppos, 0);
                else state->addAtom("cation", ppos, 0);
            }
        }
    }

//     SHARED(FixBondHarmonic) bond (new FixBondHarmonic(state, "bond"));
//     for (int i=0; i<baseLen*baseLen; i++) {
//     for (int l=1; l<polybeads; l++) {
//         bond->createBond(&state->atoms[i*polybeads+l-1], &state->atoms[i*polybeads+l], 10.0, 0.3);
// //         cout<<"added bond "<<state->atoms[i*polybeads+l-1].id<<' '<<state->atoms[i*polybeads+l].id<<'\n';
//     }
//     }
//     state->activateFix(bond);

    state->periodicInterval = 5;
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    nonbond->setParameter("sig", "anion", "anion", 0.1);
    nonbond->setParameter("sig", "anion", "cation", 0.1);
    nonbond->setParameter("sig", "cation", "cation", 0.1);
    nonbond->setParameter("eps", "anion", "anion", 0.2);
    nonbond->setParameter("eps", "anion", "cation", 0.2);
    nonbond->setParameter("eps", "cation", "cation", 0.2);
    state->activateFix(nonbond);

    SHARED(FixChargePairDSF) charge (new FixChargePairDSF(state, "charge","all"));
    charge->setParameters(0.25,9.0);
    for (int i=0; i<baseLen*baseLen; i++) {
        for (int l=0; l<polybeads; l++) {
            cout<<i*polybeads+l<<' '<<state->atoms[i*polybeads+l].id
                <<' '<<state->atoms[i*polybeads+l].type<<' '<<i%2<<'\n';
            state->atoms[i*polybeads+l].q=3.0*(1.0-2.0*(i%2));
            //charge->setCharge(i*polybeads+l,3.0*(1.0-2.0*(i%2)));
        }
    }
    state->activateFix(charge);

    SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handle", "xml", 5000));
    state->activateWriteConfig(write);

    state->dt=0.00031;

//     IntegratorRelax integratorR(state);
//     integratorR.run(500001,1.0);
    IntegratorVerlet integrator(state.get());
    integrator.run(500000);
    /*IntegratorLangevin integrator(state.get());
    integrator.run(100000);*/
    //cout << state->atoms[0].pos[0]<<' '<<state->atoms[0].vel[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[0].forceLast[0]<< endl;

}


void testPair() {
    SHARED(State) state = SHARED(State) (new State());
    state->shoutEvery = 1000;
    int L = 10.0;

    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(L,L,L));
    state->rCut = 5.0;
    state->atomParams.addSpecies("handle", 2);
    state->addAtom("handle",Vector(0, 0, 0), 0);
    state->addAtom("handle",Vector(5, 0, 0), 0);

    //charge
    SHARED(FixChargePairDSF) charge (new FixChargePairDSF(state, "charge","all"));
    charge->setParameters(0.25,4.5);
    state->atoms[0].q=1.0;
    state->atoms[1].q=-1.0;
    state->activateFix(charge);

    state->dt=0.0001;
    IntegratorVerlet integrator(state.get());

    ofstream ofs;
    ofs.open("test_pair.dat",ios::out );
    for (int i=0;i<1000-10;i++){
    state->atoms[0].pos[0]=i*5.0/1000.0;
    state->atoms[0].vel[0]=0.0;
    state->atoms[1].pos[0]=5.0;
    state->atoms[1].vel[0]=0.0;
    integrator.run(1);
//     cout<<state->atoms[0].pos[0]<<' '<<state->atoms[1].pos[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[1].force[0]<<'\n';
    ofs<<state->atoms[0].pos[0]<<' '<<state->atoms[1].pos[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[1].force[0]<<'\n';
    }
    ofs.close();
}


void test_charge_ewald() {
    SHARED(State) state = SHARED(State) (new State());
    state->shoutEvery = 1000;
    int L = 20.0;

    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(L,L,L));
    state->rCut = 5.0;
    state->atomParams.addSpecies("handle", 2);
    state->addAtom("handle",Vector(0.5*L, 0.5*L, 0.5*L), 0);
    state->addAtom("handle",Vector(0.75*L, 0.5*L, 0.5*L), 0);

    //charge
    SHARED(FixChargeEwald) charge (new FixChargeEwald(state, "charge","all"));
    charge->setParameters(64,1.0,3);
    //charge->setParameters(32,3.0);

    state->atoms[0].q=1.0;
    state->atoms[1].q=-1.0;
    state->activateFix(charge);

    state->dt=0.0001;
    IntegratorVerlet integrator(state.get());
    //charge->compute();
    integrator.run(1);

    ofstream ofs;
    ofs.open("test_pair.dat",ios::out );
    for (int i=0;i<1000-10;i++){
        state->atoms[0].pos[0]=0.5*L+i*0.25*L/1000.0;
        state->atoms[0].vel[0]=0.0;
        state->atoms[1].pos[0]=0.75*L;
        state->atoms[1].vel[0]=0.0;
        integrator.run(1);
        //cout<<state->atoms[0].pos[0]<<' '<<state->atoms[1].pos[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[1].force[0]<<'\n';
        ofs<<state->atoms[0].pos[0]<<' '<<state->atoms[1].pos[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[1].force[0]<<'\n';
    }
    ofs.close();
}


void testRead() {
    SHARED(State) state = SHARED(State) (new State());
    state->readConfig->loadFile("test.xml");
    state->readConfig->next();
    for (Atom &a : state->atoms) {
        cout << a.id << endl;
    }
    return;
    int baseLen = 40;
    double mult = 1.5;
    state->periodicInterval = 40;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(mult*baseLen, mult*baseLen, 10));
    state->rCut = 2.9;
    state->atomParams.addSpecies("handle", 2);

    state->addAtom("handle", Vector(1, 1, 1), 0);
    state->addAtom("handle", Vector(3.8, 1, 1), 0);

    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);


    //SHARED(FixBondHarmonic) harmonic = SHARED(FixBondHarmonic) (new FixBondHarmonic(state, "harmonic"));
    //state->activateFix(harmonic);
    //harmonic->createBond(&state->atoms[0], &state->atoms[1], 1, 2);

    SHARED(FixSpringStatic) springStatic = SHARED(FixSpringStatic) (new FixSpringStatic(state, "spring", "all", 1));
    state->activateFix(springStatic);

    SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "base64", 50));
    state->activateWriteConfig(write);
    //state->integrator.run(50);

}

void testWallHarmonic() {
    SHARED(State) state = SHARED(State) (new State());
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(50, 50, 10));
    state->rCut = 3.5;
    state->atomParams.addSpecies("handle", 2);
    state->atomParams.addSpecies("other", 2);
    state->is2d = true;
    state->periodic[2] = false;

    state->addAtom("handle", Vector(8, 1, 0), 0);
    state->atoms[0].vel = Vector(-1, 0, 0);
    state->periodicInterval = 9;
    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
    state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    nonbond->setParameter("sig", "other", "other", 1);
    nonbond->setParameter("eps", "other", "other", 1);
    state->activateFix(nonbond);
    cout << "last" << endl;
    cout << state->atoms[0].id<< endl;
    SHARED(FixWallHarmonic) wall (new FixWallHarmonic(state, "wally", "all", Vector(4, 0, 0), Vector(1, 1, 0), 3, 10));
    state->activateFix(wall);
    state->createGroup("sub");
    //SHARED(FixBondHarmonic) harmonic = SHARED(FixBondHarmonic) (new FixBondHarmonic(state, "harmonic"));
    //state->activateFix(harmonic);
    //harmonic->createBond(&state->atoms[0], &state->atoms[1], 1, 2);

    //SHARED(FixSpringStatic) springStatic = SHARED(FixSpringStatic) (new FixSpringStatic(state, "spring", "all", 1, Py_None));
    //state->activateFix(springStatic);

    SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "base64", 50));
    //state->activateWriteConfig(write);
    //state->integrator.run(200);
    InitializeAtoms::populateRand(state, state->bounds, "handle", 200, 1.1);
    cout << "here" << endl;
    //state->integrator.run(2000);

}


void testBondHarmonic() {
    SHARED(State) state = SHARED(State) (new State());
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(50, 50, 10));
    state->rCut = 3.5;
    state->atomParams.addSpecies("handle", 2);
    state->is2d = true;
    state->periodic[2] = false;

    state->addAtom("handle", Vector(6, 1, 0), 0);
    state->addAtom("handle", Vector(8, 1, 0), 0);
    state->addAtom("handle", Vector(9, 1, 0), 0);

    state->periodicInterval = 9;
    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
    state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);
    cout << state->atoms[0].id<< endl;
    SHARED(FixBondHarmonic) bond (new FixBondHarmonic(state, "bondh"));

    state->activateFix(bond);
    bond->createBond(&state->atoms[0], &state->atoms[1], 1, 2, -1);
    bond->createBond(&state->atoms[1], &state->atoms[2], 1, 2, -1);
    cout << "req" << endl;
    cout << bond->getBond(0).r0 << endl;
    cout << bond->getBond(1).r0 << endl;
    IntegratorRelax integratorR(state);
    integratorR.run(1,1e-8);
    for (int i=0; i<3; i++) {
        cout << state->atoms[i].pos[0] << endl;
    }


}


void testBondHarmonicGrid() {
    SHARED(State) state = SHARED(State) (new State());
    state->is2d = true;
    state->periodic[2] = false;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(50, 50, 10));
    state->rCut = 3.5;
    state->atomParams.addSpecies("handle", 2);

    SHARED(FixBondHarmonic) bond (new FixBondHarmonic(state, "bondh"));

    state->activateFix(bond);
    double spacing = 1.4;
    int n = 50;

    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            state->addAtom("handle", Vector(i*spacing, j*spacing, 0), 0);
        }
    }
  //  state->addAtom("handle", Vector(1, 1, 0), 0);
   // state->addAtom("handle", Vector(3, 1, 0), 0);

    double r0 = 1.0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (i<n-1) {
                bond->createBond(&state->atoms[(i+1)*n+j], &state->atoms[i*n+j], 1, r0, -1);
            }
            if (j<n-1) {
                bond->createBond(&state->atoms[i*n+j+1], &state->atoms[i*n+j], 1, r0, -1);
            }
        }
    }
    state->periodicInterval = 9;
    /*
    State::ExclusionList out = state->generateExclusionList(4);
    for (auto atom : out) {
        cout << "atom id: " << atom.first << endl;
        int depth = 1;
        for (auto excls : atom.second) {
            cout << "  depth " << depth << ": ";
            for (auto e : excls) { std::cout << e << " "; }
            ++depth;
            cout << endl;
        }
    }
    */
    //return;

    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
    state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    //state->activateFix(nonbond);

    IntegratorRelax integratorR(state);
    integratorR.run(60000,1e-8);
    for (Atom &a : state->atoms) {
        cout << a.pos << endl;
    }


}




void testBondHarmonicGridToGPU() {
    SHARED(State) state = SHARED(State) (new State());
    state->is2d = true;
    state->periodic[2] = false;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(50, 50, 10));
    state->rCut = 3.5;
    state->atomParams.addSpecies("handle", 2);

    SHARED(FixBondHarmonic) bond (new FixBondHarmonic(state, "bondh"));

    state->activateFix(bond);
    double spacing = 1.4;
    /*
    int n = 2;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            state->addAtom("handle", Vector(i*spacing, j*spacing, 0), 0);
        }
    }
  //  state->addAtom("handle", Vector(1, 1, 0), 0);
   // state->addAtom("handle", Vector(3, 1, 0), 0);

    double r0 = 1.0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (i<n-1) {
                bond->createBond(&state->atoms[(i+1)*n+j], &state->atoms[i*n+j], 1, r0);
            }
            if (j<n-1) {
                bond->createBond(&state->atoms[i*n+j+1], &state->atoms[i*n+j], 1, r0);
            }
        }
    }
    */
    state->addAtom("handle", Vector(1, 1, 0), 0);
    state->addAtom("handle", Vector(2, 1, 0), 0);
    state->addAtom("handle", Vector(3, 1, 0), 0);
    state->addAtom("handle", Vector(4, 1, 0), 0);
    state->addAtom("handle", Vector(4.1, 1, 0), 0);
    //state->addAtom("handle", Vector(4, 1, 0), 0);
    bond->createBond(&state->atoms[0], &state->atoms[1], 1, 1, -1);
    bond->createBond(&state->atoms[2], &state->atoms[1], 1, 1, -1);
    bond->createBond(&state->atoms[2], &state->atoms[3], 1, 1, -1);
    bond->createBond(&state->atoms[4], &state->atoms[3], 1, 1, -1);
    //bond->createBond(&state->atoms[4], &state->atoms[0], 1, 1);
  //  bond->createBond(&state->atoms[2], &state->atoms[3], 1, 1);
    state->periodicInterval = 9;
    /*
    State::ExclusionList out = state->generateExclusionList(4);
    for (auto atom : out) {
        cout << "atom id: " << atom.first << endl;
        int depth = 1;
        for (auto excls : atom.second) {
            cout << "  depth " << depth << ": ";
            for (auto e : excls) { std::cout << e << " "; }
            ++depth;
            cout << endl;
        }
    }
    */
    //return;
    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
    state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    state->activateFix(nonbond);
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    //state->activateFix(nonbond);

    IntegratorRelax integratorR(state);
    //integratorR.run(60000,1e-8);
    integratorR.run(5000, 1e-3);
    for (BondVariant &bv : bond->bonds) {
        Bond single = get<BondHarmonic>(bv);
        //cout << single.atoms[0]->pos << " " << single.atoms[1]->pos << endl;
    }


}



void hoomdBench() {
    SHARED(State) state = SHARED(State) (new State());
    state->shoutEvery = 10;
    double boxLen = 55.12934875488;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(boxLen, boxLen, boxLen));
    state->rCut = 3.0;
    state->padding = 0.6;
    state->atomParams.addSpecies("handle", 1);
    InitializeAtoms::populateRand(state, state->bounds, "handle", 6000, 0.6);

    cout << "populated" << endl;

    //state->atoms.pos[0] += Vector(0.1, 0, 0);

    state->periodicInterval = 7;
   // SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
  //  state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);
    vector<double> intervals = {0, 1};
    vector<double> temps = {1.2, 1.2};
    //SHARED(FixNVTRescale) thermo = SHARED(FixNVTRescale) (new FixNVTRescale(state, "thermo", "all", intervals, temps, 100));
    //state->activateFix(thermo);
    FILE *input = fopen("/home/daniel/Documents/hoomd_benchmarks/hoomd-benchmarks/lj-liquid/stripped.xml", "r");
    char buf[150];
/*
    for (int i=0; i<64000; i++) {
        fgets(buf, 150, input);
        string s(buf);
        istringstream is(s);
        double pos[3];
        for (int i=0; i<3; i++) {
            is >> pos[i];
        }
        state->addAtom("handle", Vector(pos), 0);
    }
    */
    InitializeAtoms::initTemp(state, "all", 1.2);
    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "xml", 20));
  //  state->activateWriteConfig(write);

    IntegratorVerlet verlet(state.get());
    verlet.run(5000);
    cout.flush();
    //SHARED(FixBondHarmonic) harmonic = SHARED(FixBondHarmonic) (new FixBondHarmonic(state, "harmonic"));
    //state->activateFix(harmonic);
    //harmonic->createBond(&state->atoms[0], &state->atoms[1], 1, 2);

    //SHARED(FixSpringStatic) springStatic = SHARED(FixSpringStatic) (new FixSpringStatic(state, "spring", "all", 1, Py_None));
    //state->activateFix(springStatic);

    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "base64", 50));
    //state->activateWriteConfig(write);
    //state->integrator.run(1000);

}

void testLJ() {
    SHARED(State) state = SHARED(State) (new State());
    state->devManager.setDevice(0);
    int baseLen = 20;
    state->shoutEvery = 1000;
    double mult = 1.5;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(mult*baseLen, mult*baseLen, mult*baseLen));
    state->rCut = 2.5;
    state->padding = 0.5;
    state->atomParams.addSpecies("handle", 2);
    vector<double> intervals = {0, 1};
    vector<double> temps = {1.2, 1.2};

    //SHARED(FixNVTRescale) thermo = SHARED(FixNVTRescale) (new FixNVTRescale(state, "thermo", "all", intervals, temps, 100));
    //state->activateFix(thermo);
    //state->is2d = true;
    //state->periodic[2] = false;
   // for (int i=0; i<32; i++) {
        //state->addAtom("handle", Vector(2*i+1, 1, 0), 0);
     //   state->addAtom("handle", Vector(2*31+1-2*i, 1, 0), 0);
   // }

  //  for (int i=0; i<32; i++) {
   //     state->addAtom("handle", Vector(2*i+1, 5, 0), 0);
 //   }
    //state->addAtom("handle", Vector(1, 1, 0), 0);
    //state->addAtom("handle", Vector(3.0, 1, 0), 0);

   // state->addAtom("handle", Vector(5.0, 1, 0), 0);
   // state->addAtom("handle", Vector(7.0, 1, 0), 0);
    for (int i=0; i<baseLen; i++) {
        for (int j=0; j<baseLen; j++) {
            for (int k=0; k<baseLen; k++) {
            //    state->addAtom("handle", Vector(i*mult + (rand() % 20)/40.0, j*mult + (rand() % 20)/40.0, 0), 0);
                state->addAtom("handle", Vector(i*mult + (rand() % 20)/40.0, j*mult + (rand() % 20)/40.0, k*mult + (rand() % 20)/40.0), 0);
            }
        }
    }

    InitializeAtoms::initTemp(state, "all", 1.2);
    //state->atoms.pos[0] += Vector(0.1, 0, 0);

    state->periodicInterval = 9;
   // SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
  //  state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);

    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "xml", 20));
  //  state->activateWriteConfig(write);

    IntegratorVerlet verlet(state.get());
    cout << state->atoms[0].pos << endl;
    //verlet.run();
    cout << state->atoms[0].pos << endl;
    cout << state->atoms[1].pos << endl;
    cout << state->atoms[0].force << endl;
    cout.flush();
    double sumKe = 0;
    for (Atom &a : state->atoms) {
        sumKe += a.kinetic();
    }
    cout << "temp  is " << (sumKe*(2.0/3.0)/state->atoms.size()) << endl;
    //SHARED(FixBondHarmonic) harmonic = SHARED(FixBondHarmonic) (new FixBondHarmonic(state, "harmonic"));
    //state->activateFix(harmonic);
    //harmonic->createBond(&state->atoms[0], &state->atoms[1], 1, 2);

    //SHARED(FixSpringStatic) springStatic = SHARED(FixSpringStatic) (new FixSpringStatic(state, "spring", "all", 1, Py_None));
    //state->activateFix(springStatic);

    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "base64", 50));
    //state->activateWriteConfig(write);
    //state->integrator.run(1000);

}
void testGPUArrayTex() {

    GPUArrayDeviceTex<int> xs(10, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned));
    vector<int> vals;
    for (int i=0; i<10; i++) {
        vals.push_back(i+1);
    }
    xs.set(vals.data());
    cout << xs.size() << endl;
    cout << xs.capacity() << endl;

}




int main(int argc, char **argv) {

    //MPI_Init(&argc, &argv);

    if (argc > 1) {
        int arg = atoi(argv[1]);
        if (arg==0) {
    //        testDihedral();
            testLJ();
            // testLJ();
            // hoomdBench();
            //testBondHarmonicGridToGPU();
        } else if (arg==1) {
//             testPair();
//             testFire();
        test_charge_ewald();
        } else if (arg==2) {
            testBondHarmonicGrid();
            //sean put your test stuff here
        }
    } else {
        cout << "no argvs specified, doing nothing" << endl;
    }

    //testLJ();
    //testWallHarmonic();
    //testRead();
//      testGPUArrayTex();
//    testCharge();
//      testPair();
    return 0;
    //testSum();
    //return 0;
    SHARED(State) state = SHARED(State) (new State());
    /*
    state->readConfig->loadFile("test.xml");
    state->readConfig->next();
    for (Atom &a : state->atoms) {
        cout << a.pos << endl;
    }
    */
    //ADD A FUNCTION WHICH CHECKS IF IDTOIDX IS CORRECT
    int baseLen = 40;
    double mult = 1.5;
    //state->periodic[2] = false;
    state->periodicInterval = 40;
    //state->is2d = true;
    //breaks at 2670 on work computer
    //SORTING NOT WORKING FOR IS2d = true
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(mult*baseLen, mult*baseLen, 10));
    state->rCut = 2.9;
    state->atomParams.addSpecies("handle", 2);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "nb"));
    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 30));
    state->activateFix(nonbond);
    state->activateFix(f2d);
    //IMPORTANT - NEIGHBORLISTING BREAKS WHEN YOU GO TO 3D
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
  //  state->addAtom("handle", Vector(2, 2, 0));
  //  state->addAtom("handle", Vector(4, 2.7, 0));
  //  state->atoms[0].vel = Vector(2, 0, 0);

    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "xml", 5));
    //state->activateWriteConfig(write);
    int n = 8000;
    //state->integrator.run(n);
    /*
    for (Atom a : state->atoms) {
        cout << a.force << endl;
    }
    */
    //for (int i=0; i<
    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "xml", 1));
    //state->activateWriteConfig(write);
    //SHARED(FixBondHarmonic) harmonic = SHARED(FixBondHarmonic) (new FixBondHarmonic(state, "harmonic", 1));
    //state->activateFix(harmonic);
    //harmonic->createBond(&state->atoms[0], &state->atoms[1], 1, 1);
    //state->integrator.run(1000);
    //state->integrator.test();

    //MPI_Finalize();

}
//for benchmarking sums
    /*
    int n = 100000;
    float *dst;
#define XXX float4
    XXX *xs;
    cudaMalloc(&dst, sizeof(float));
    cudaMalloc(&xs, n*sizeof(XXX));
    std::vector<XXX> src(n);
    for (int i=0; i<n; i++) {
        src[i] = make_float4(i+1, 2*i+1, i+1, 2*i+1);
        //src[i] = i;
    }
    cudaMemcpy(xs, src.data(), n*sizeof(XXX), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
  //  sumSingle<float, float, 1> <<<NBLOCK(n), PERBLOCK, 1*sizeof(float)*PERBLOCK>>>(dst, xs, n, 32);
    int warpsize = state->devManager.prop.warpSize;
    auto t1 = Clock::now();
    for (int j = 0; j<100000; j++) {
        cudaMemset(dst, 0, sizeof(float));
        cudaDeviceSynchronize();
        //printf("NBLOCK IS %d PERBLOCK IS %d\n", NBLOCK(n), PERBLOCK);
        accumulate_gpu<float, float4, KEKE, 4> <<<NBLOCK(n / (double) 4), PERBLOCK, 4*sizeof(float)*PERBLOCK>>>(dst, xs, n, warpsize, KEKE());
        float res;
        cudaMemcpy(&res, dst, sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        float cpures = 0;//make_float4(0, 0, 0, 0);
      //  for (XXX v : src) {
      //      cpures += v.x+v.z + v.y/v.z;
      //  }
        //printf("%f %f\n", res, cpures);
        //if (res != cpures ) {
            //std::cout << "uh oh " << n << " " << (res - cpures) << std::endl;
       //     printf("res is %f\n", res);
       //     printf("cpu is %f\n", cpures);
            //printf("res is %f %f %f %f \n", res.x, res.y, res.z, res.w);
            //printf("cpu is %f %f %f %f \n", cpures.x, cpures.y, cpures.z, cpures.w);
          //  n-=1;
        //}
    }
    auto t2 = Clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
    cudaFree(dst);
    cudaFree(xs);
    */
    /*
    for (int n=9000000; n<10000000; n+=10000) {
        //int n = 100000;
#define LE_SRC float
#define LE_DEST float
        LE_DEST *dst;
        LE_SRC *xs;
        cudaMalloc(&dst, sizeof(LE_DEST));
        cudaMalloc(&xs, n*sizeof(LE_SRC));
        std::vector<LE_SRC> src(n);
        for (int i=0; i<n; i++) {
            src[i] = rand() / (double) RAND_MAX;// make_float4(i, 2*i, i, 2*i);
           // src[i] = i;
        }
        cudaMemcpy(xs, src.data(), n*sizeof(LE_SRC), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
      //  sumSingle<float, float, 1> <<<NBLOCK(n), PERBLOCK, 1*sizeof(float)*PERBLOCK>>>(dst, xs, n, 32);
        int warpsize = state->devManager.prop.warpSize;
        auto t1 = Clock::now();
        for (int j = 0; j<1; j++) {
            cudaMemset(dst, 0, sizeof(LE_DEST));
            cudaDeviceSynchronize();
            //printf("NBLOCK IS %d PERBLOCK IS %d\n", NBLOCK(n), PERBLOCK);
            accumulate_gpu <<<NBLOCK(n), PERBLOCK, sizeof(LE_DEST)*PERBLOCK>>>(dst, xs, n, warpsize, KEKE());
            float res;
            cudaMemcpy(&res, dst, sizeof(LE_DEST), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            double cpures = 0;//make_float4(0, 0, 0, 0);
            float cpuresF = 0;
            for (LE_SRC x : src) {
                cpures += (double) x;
                cpuresF += (float) x;
            }
            //printf("%f %f\n", res, cpures);
            //if (res != cpures ) {
                //std::cout << "uh oh " << n << " " << (res - cpures) << std::endl;
                printf("gpu %f cpu %f, cpuF %f\n", res, cpures, cpuresF);
                //printf("cpu is %f\n", cpures);
                //printf("res is %f %f %f %f \n", res.x, res.y, res.z, res.w);
                //printf("cpu is %f %f %f %f \n", cpures.x, cpures.y, cpures.z, cpures.w);
              //  n-=1;
            //}
        }
        auto t2 = Clock::now();
    //    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
        cudaFree(dst);
        cudaFree(xs);
    }
    exit(0);
    */
