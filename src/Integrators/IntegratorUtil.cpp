#include "IntegratorUtil.h"

#include "State.h"
#include "DataSetUser.h"
#include "DataManager.h"
#include "DataComputer.h"
#include "Fix.h"
#include <vector>
#include "Mod.h"
using namespace MD_ENGINE;
IntegratorUtil::IntegratorUtil(State *state_) {
    state = state_;
}

void IntegratorUtil::force(int virialMode) {
    int simTurn = state->turn;
    std::vector<Fix *> &fixes = state->fixes;
    //okay - things with order pref == -1 are pair forces.  They for first.  Afterwards, we compute f dot r if necessary, then do the rest
  //  bool computedFDotR = false;
    for (Fix *f : fixes) {
        if (! (simTurn % f->applyEvery)) {
           // if (virialMode == 1 and f->orderPreference >= 0 and not computedFDotR) {
           //     Mod::FDotR(state);
           //     computedFDotR = true;
           // }
            f->compute(virialMode);
            f->setVirialTurn();
        }
    }
};

void IntegratorUtil::forceInitial(int virialMode) {
    int simTurn = state->turn;
    std::vector<Fix *> &fixes = state->fixes;
    //okay - things with order pref == -1 are pair forces.  They for first.  Afterwards, we compute f dot r if necessary, then do the rest
  //  bool computedFDotR = false;
    for (Fix *f : fixes) {
        if (! (simTurn % f->applyEvery)) {
           // if (virialMode == 1 and f->orderPreference >= 0 and not computedFDotR) {
           //     Mod::FDotR(state);
           //     computedFDotR = true;
           // }
            //std::cout << "Going to compute fix: " << f->handle << " if it is prepared; " << std::endl;
            if (f->prepared) {
            //    std::cout << "Computing fix " << f->handle << std::endl;
                f->compute(virialMode);
                f->setVirialTurn();
            }
        }
    }
    //std::cout << "Looped over all fixes!" << std::endl;
};

void IntegratorUtil::postNVE_V() {
    int simTurn = state->turn;
    std::vector<Fix *> &fixes = state->fixes;
    for (Fix *f : fixes) {
        if (f->willFire(simTurn)) {
            f->postNVE_V();
        }
    }
}

void IntegratorUtil::postNVE_X() {
    int simTurn = state->turn;
    std::vector<Fix *> &fixes = state->fixes;
    for (Fix *f : fixes) {
        if (f->willFire(simTurn)) {
            f->postNVE_X();
        }
    }
}

/*
void IntegratorUtil::singlePointEng() {
    GPUArrayGlobal<real> &perParticleEng = state->gpd.perParticleEng;
    perParticleEng.d_data.memset(0);
    for (Fix *f : state->fixes) {
        f->singlePointEng(perParticleEng.getDevData());
    }

}
*/

void IntegratorUtil::forceSingle(int virialMode) {
    for (Fix *f : state->fixes) {
        if (f->forceSingle and f->willFire(state->turn)) {
            f->compute(virialMode);
            f->setVirialTurn();
        }
    }
}


void IntegratorUtil::doDataComputation() {
    DataManager &dm = state->dataManager;
    int64_t turn = state->turn;
    bool computedAny = false;
    bool requireIds = false;
    // looping over all datasets
    // --- if the data set is computing this turn, and is in 'vector' -mode (per-particle data, typically)
    //     then we need to send 
    for (boost::shared_ptr<DataSetUser> ds : dm.dataSets) {
        if (ds->nextCompute == turn and ds->computer->computeMode == "vector") {
            //if (ds->computer->type != "hamiltonian") {
                // if data computer isn't a DataComputerHamiltonian, and its in vector compute mode, 
                // we'll need the id's. 
                // DataComputerHamiltonian in 'vector' compute mode does not, though.  
                requireIds = true;
            //}
        }
    }
    if (requireIds) {
        state->gpd.ids.dataToHost(); //need ids to map back to original ordering
    }
    for (boost::shared_ptr<DataSetUser> ds : dm.dataSets) {
        if (ds->nextCompute == turn) {
            ds->computeData();
            computedAny = true;
        }
    }
    if (computedAny) {
        cudaDeviceSynchronize();
    }
}

void IntegratorUtil::doDataAppending() {
    DataManager &dm = state->dataManager;
    int64_t turn = state->turn; 
    for (boost::shared_ptr<DataSetUser> ds : dm.dataSets) {
        if (ds->nextCompute == turn) {
            ds->appendData();
            int64_t nextTurn = ds->setNextTurn(turn);
            if (ds->requiresVirials()) {
                dm.addVirialTurn(nextTurn, ds->requiresPerAtomVirials());
            }

        }
    }
}

void IntegratorUtil::handleBoundsChange() {
    for (Fix *f : state->fixes) {
        f->handleBoundsChange();
    }
}

void IntegratorUtil::checkQuit() {
    if (PyErr_CheckSignals() == -1) {
        exit(1);
    }
}

