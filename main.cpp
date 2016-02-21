#include "Python.h"
#include "GPUArrayTex.h"
#include "GPUArray.h"
#include "globalDefs.h"
#include <vector>
#include "State.h"
#include "FixBondHarmonic.h"
#include "FixPair.h"
//#include "cutils_func.h"
#include "includeFixes.h"
#include <stdio.h>
#include <cstdlib>
#include "InitializeAtoms.h"
#include  "IntegraterVerlet.h"
#include  "IntegraterRelax.h"
#include  "IntegraterLangevin.h"
#include "FixChargePairDSF.h"
#include "WriteConfig.h"
#include "ReadConfig.h"
#include <stdlib.h>

using namespace std;


void testFire() {
    SHARED(State) state = SHARED(State) (new State());
    int baseLen = 50;
    state->shoutEvery = 100;
    double mult = 1.5;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(mult*baseLen, mult*baseLen, 10));
    state->rCut = 3.5;
    state->grid = AtomGrid(state.get(), 4, 4, 3);
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
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut", "all"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);
    cout << "last" << endl;
    cout << state->atoms[0].pos[0]<<' '<<state->atoms[0].vel[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[0].forceLast[0]<< endl;

    SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handle", "xml", 10000));    
    state->activateWriteConfig(write);

    state->dt=0.003;    
    IntegraterRelax integraterR(state);
    integraterR.run(400000,1.0);  
//     cout << state->atoms[0].pos[0]<<' '<<state->atoms[0].vel[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[0].forceLast[0]<< endl;
    
}



void testCharge() {
    SHARED(State) state = SHARED(State) (new State());
    int baseLen = 25;
    int polybeads=1;
    double b=0.3;
    state->shoutEvery = 1000;
    double mult = 3.25;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(mult*baseLen, mult*baseLen,mult*baseLen));
    state->rCut = 9.5;
    state->grid = AtomGrid(state.get(), 9.5, 9.5, 9.5);
    state->atomParams.addSpecies("anion", 2);
    state->atomParams.addSpecies("cation", 3);
    double eps=2.0;
    std::srand(2);

    for (int i=0; i<baseLen; i++) {
        for (int j=0; j<baseLen; j++) {
	    int k=0;
	    Vector ppos=Vector(i*mult+eps*(double(std::rand())/RAND_MAX), j*mult+eps*(double(std::rand())/RAND_MAX),double(k));
	    if ((i*baseLen+j)%2) state->addAtom("anion",ppos, 0);
	    else state->addAtom("cation",ppos, 0);
	    for (int l=1; l<polybeads; l++) {
		float3 a=make_float3((0.5-double(std::rand())/RAND_MAX),
				 (0.5-double(std::rand())/RAND_MAX),
				 (double(std::rand())/RAND_MAX));
		a=normalize(a)*b;
		ppos=ppos+Vector(a);
		if ((i*baseLen+j)%2) state->addAtom("anion",ppos, 0);
		else state->addAtom("cation",ppos, 0);
        }
        }
    }

//     SHARED(FixBondHarmonic) bond (new FixBondHarmonic(state, "bond"));
//     for (int i=0; i<baseLen*baseLen; i++) {
// 	for (int l=1; l<polybeads; l++) {
// 	    bond->createBond(&state->atoms[i*polybeads+l-1], &state->atoms[i*polybeads+l], 10.0, 0.3);
// // 	    cout<<"added bond "<<state->atoms[i*polybeads+l-1].id<<' '<<state->atoms[i*polybeads+l].id<<'\n';
// 	}
//     }
//     state->activateFix(bond);


    state->periodicInterval = 5;
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut", "all"));
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
// 	    charge->setCharge(i*polybeads+l,3.0*(1.0-2.0*(i%2)));
	}
    }    
    state->activateFix(charge);
    
    SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handle", "xml", 5000));    
    state->activateWriteConfig(write);

    state->dt=0.00031;  

//     IntegraterRelax integraterR(state);
//     integraterR.run(500001,1.0);
    IntegraterVerlet integrater(state);
    integrater.run(500000);   
/*    IntegraterLangevin integrater(state);
    integrater.run(100000);*/     
//     cout << state->atoms[0].pos[0]<<' '<<state->atoms[0].vel[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[0].forceLast[0]<< endl;
    
}


void testPair() {
    SHARED(State) state = SHARED(State) (new State());
    state->shoutEvery = 1000;
    int L = 10.0;

    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(L,L,L));
    state->rCut = 5.0;
    state->grid = AtomGrid(state.get(), L,L,L);
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
    IntegraterVerlet integrater(state);

    ofstream ofs;
    ofs.open("test_pair.dat",ios::out );
    for (int i=0;i<1000-10;i++){
	state->atoms[0].pos[0]=i*5.0/1000.0;
	state->atoms[0].vel[0]=0.0;
	state->atoms[1].pos[0]=5.0;
	state->atoms[1].vel[0]=0.0;
	integrater.run(1);     
// 	cout<<state->atoms[0].pos[0]<<' '<<state->atoms[1].pos[0]<<' '<<state->atoms[0].force[0]<<' '<<state->atoms[1].force[0]<<'\n';
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
    state->grid = AtomGrid(state.get(), 3, 3, 3);
    state->atomParams.addSpecies("handle", 2);

    
    state->addAtom("handle", Vector(1, 1, 1), 0);
    state->addAtom("handle", Vector(3.8, 1, 1), 0);

    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut", "all"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);


    //SHARED(FixBondHarmonic) harmonic = SHARED(FixBondHarmonic) (new FixBondHarmonic(state, "harmonic"));
    //state->activateFix(harmonic);
    //harmonic->createBond(&state->atoms[0], &state->atoms[1], 1, 2);
    
    SHARED(FixSpringStatic) springStatic = SHARED(FixSpringStatic) (new FixSpringStatic(state, "spring", "all", 1, Py_None));
    state->activateFix(springStatic);

    SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "base64", 50));
    state->activateWriteConfig(write);
    //state->integrater.run(50);

}

void testWallHarmonic() {
    SHARED(State) state = SHARED(State) (new State());
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(50, 50, 10));
    state->rCut = 3.5;
    state->grid = AtomGrid(state.get(), 4, 4, 3);
    state->atomParams.addSpecies("handle", 2);
    state->atomParams.addSpecies("other", 2);
    state->is2d = true;
    state->periodic[2] = false;
    

    state->addAtom("handle", Vector(8, 1, 0), 0);
    state->atoms[0].vel = Vector(-1, 0, 0);
    state->periodicInterval = 9;
    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
    state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut", "all"));
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
    //state->integrater.run(200);
    InitializeAtoms::populateRand(state, state->bounds, "handle", 200, 1.1);
    cout << "here" << endl;
    //state->integrater.run(2000);

}


void testBondHarmonic() {
    SHARED(State) state = SHARED(State) (new State());
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(50, 50, 10));
    state->rCut = 3.5;
    state->grid = AtomGrid(state.get(), 4, 4, 3);
    state->atomParams.addSpecies("handle", 2);
    state->is2d = true;
    state->periodic[2] = false;
    

    state->addAtom("handle", Vector(6, 1, 0), 0);
    state->addAtom("handle", Vector(8, 1, 0), 0);
    state->addAtom("handle", Vector(9, 1, 0), 0);

    state->periodicInterval = 9;
    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
    state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut", "all"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);
    cout << state->atoms[0].id<< endl;
    SHARED(FixBondHarmonic) bond (new FixBondHarmonic(state, "bondh"));

    state->activateFix(bond);
    bond->createBond(&state->atoms[0], &state->atoms[1], 1, 2);
    bond->createBond(&state->atoms[1], &state->atoms[2], 1, 2);
    cout << "req" << endl;
    cout << bond->getBond(0).rEq << endl;
    cout << bond->getBond(1).rEq << endl;
    IntegraterRelax integraterR(state);
    integraterR.run(1,1e-8);
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
    state->grid = AtomGrid(state.get(), 4, 4, 3);
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
    
    double rEq = 1.0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (i<n-1) {
                bond->createBond(&state->atoms[(i+1)*n+j], &state->atoms[i*n+j], 1, rEq);
            }
            if (j<n-1) {
                bond->createBond(&state->atoms[i*n+j+1], &state->atoms[i*n+j], 1, rEq);
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
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut", "all"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    //state->activateFix(nonbond);
    
    IntegraterRelax integraterR(state);
    integraterR.run(60000,1e-8);
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
    state->grid = AtomGrid(state.get(), 4, 4, 3);
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
    
    double rEq = 1.0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            if (i<n-1) {
                bond->createBond(&state->atoms[(i+1)*n+j], &state->atoms[i*n+j], 1, rEq);
            }
            if (j<n-1) {
                bond->createBond(&state->atoms[i*n+j+1], &state->atoms[i*n+j], 1, rEq);
            }
            
        }
    }
    */
    state->addAtom("handle", Vector(1, 1, 0), 0);
    state->addAtom("handle", Vector(2, 1, 0), 0);
    state->addAtom("handle", Vector(3, 1, 0), 0);
    state->addAtom("handle", Vector(4, 1, 0), 0);
    state->addAtom("handle", Vector(4.1, 1, 0), 0);
//    state->addAtom("handle", Vector(4, 1, 0), 0);
    bond->createBond(&state->atoms[0], &state->atoms[1], 1, 1);
    bond->createBond(&state->atoms[2], &state->atoms[1], 1, 1);
    bond->createBond(&state->atoms[2], &state->atoms[3], 1, 1);
    bond->createBond(&state->atoms[4], &state->atoms[3], 1, 1);
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
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut", "all"));
    state->activateFix(nonbond);
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    //state->activateFix(nonbond);
    
    IntegraterRelax integraterR(state);
   // integraterR.run(60000,1e-8);
    integraterR.run(5000, 1e-3);
    for (BondVariant &bv : bond->bonds) {
        Bond single = get<BondHarmonic>(bv);
        cout << single.atoms[0]->pos << " " << single.atoms[1]->pos << endl;
    }


}



void testLJ() {
    SHARED(State) state = SHARED(State) (new State());
    int baseLen = 40;
    state->shoutEvery = 100;
    double mult = 1.5;
    state->bounds = Bounds(state, Vector(0, 0, 0), Vector(mult*baseLen, mult*baseLen, mult*baseLen));
    state->rCut = 2.5;
    state->padding = 0.5;
    state->grid = AtomGrid(state.get(), 3.5, 3.5, 3);
    state->atomParams.addSpecies("handle", 2);
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
               // state->addAtom("handle", Vector(i*mult + (rand() % 20)/40.0, j*mult + (rand() % 20)/40.0, 0), 0);
                state->addAtom("handle", Vector(i*mult + (rand() % 20)/40.0, j*mult + (rand() % 20)/40.0, k*mult + (rand() % 20)/40.0), 0);
            }
        }
    }

    
  //  state->atoms.pos[0] += Vector(0.1, 0, 0);

    state->periodicInterval = 9;
   // SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 1));
  //  state->activateFix(f2d);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "ljcut", "all"));
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
    state->activateFix(nonbond);

    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "xml", 20));
  //  state->activateWriteConfig(write);

    cout << "last" << endl;
    IntegraterVerlet verlet = IntegraterVerlet(state);
    cout << state->atoms[80].pos;
    verlet.run(2000);
    cout << state->atoms[0].pos << endl;
    cout << state->atoms[1].pos << endl;
    cout << state->atoms[0].force << endl;
    cout.flush();
    //SHARED(FixBondHarmonic) harmonic = SHARED(FixBondHarmonic) (new FixBondHarmonic(state, "harmonic"));
    //state->activateFix(harmonic);
    //harmonic->createBond(&state->atoms[0], &state->atoms[1], 1, 2);
    
    //SHARED(FixSpringStatic) springStatic = SHARED(FixSpringStatic) (new FixSpringStatic(state, "spring", "all", 1, Py_None));
    //state->activateFix(springStatic);

    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "base64", 50));
    //state->activateWriteConfig(write);
    //state->integrater.run(1000);

}
void testGPUArrayTex() {

    GPUArrayTexDevice<int> xs(10, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned));
    vector<int> vals;
    for (int i=0; i<10; i++) {
        vals.push_back(i+1);
    }
    xs.set(vals.data());
    cout << xs.size << endl;
    cout << xs.capacity<< endl;

}

int main(int argc, char **argv) {
    if (argc > 1) {
        int arg = atoi(argv[1]);
        if (arg==0) {
            testLJ();
            //testBondHarmonicGridToGPU();
        } else if (arg==1) {
//             testPair();
            testFire();
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
    state->grid = AtomGrid(state.get(), 3, 3, 3);
    state->atomParams.addSpecies("handle", 2);
    SHARED(FixLJCut) nonbond = SHARED(FixLJCut) (new FixLJCut(state, "nb", "all"));
    SHARED(Fix2d) f2d = SHARED(Fix2d) (new Fix2d(state, "2d", 30));
    state->activateFix(nonbond);
    state->activateFix(f2d);
    //IMPORTANT - NEIGHBORLISTING BREAKS WHEN YOU GO TO 3D
    nonbond->setParameter("sig", "handle", "handle", 1);
    nonbond->setParameter("eps", "handle", "handle", 1);
  //  state->addAtom("handle", Vector(2, 2, 0));
  //  state->addAtom("handle", Vector(4, 2.7, 0));
  //  state->atoms[0].vel = Vector(2, 0, 0);
    
    for (int i=0; i<baseLen; i++) {
        for (int j=0; j<baseLen; j++) {
            state->atoms.push_back(Atom(Vector(mult*j, mult*i, 0), 0, i*baseLen+j, 2, 0));
            state->atoms.push_back(Atom(Vector(mult*j, mult*i, 3), 0, i*baseLen+j, 2, 0));
            state->atoms.push_back(Atom(Vector(mult*j, mult*i, 6), 0, i*baseLen+j, 2, 0));
            state->atoms.push_back(Atom(Vector(mult*(j+0.5), mult*(i+0.5), 1.5), 0, i*baseLen+j, 2, 0));
            state->atoms.push_back(Atom(Vector(mult*(j+0.5), mult*(i+0.5), 4.5), 0, i*baseLen+j, 2, 0));
            state->atoms.push_back(Atom(Vector(mult*(j+0.5), mult*(i+0.5), 7.5), 0, i*baseLen+j, 2, 0));
            //state->addAtom("handle", Vector(mult*i, mult*j, 0)); //SLOW
        }
    }
    InitializeAtoms::initTemp(state, "all", 0.1);
    //SHARED(WriteConfig) write = SHARED(WriteConfig) (new WriteConfig(state, "test", "handley", "xml", 5));
    //state->activateWriteConfig(write);
    int n = 8000;
    //state->integrater.run(n);
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
    //state->integrater.run(1000);
    //state->integrater.test();

    
}

