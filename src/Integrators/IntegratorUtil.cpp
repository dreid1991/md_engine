#include "IntegratorUtil.h"

#include "State.h"
#include "DataSetUser.h"
#include "DataManager.h"
#include "Fix.h"
#include <vector>
using namespace MD_ENGINE;
IntegratorUtil::IntegratorUtil(State *state_) {
    state = state_;
}

void IntegratorUtil::force(bool computeVirials) {
    int simTurn = state->turn;
    std::vector<Fix *> &fixes = state->fixes;
    for (Fix *f : fixes) {
        if (! (simTurn % f->applyEvery)) {
            f->compute(computeVirials);
        }
    }
};


void IntegratorUtil::singlePointEng() {
    GPUArrayGlobal<float> &perParticleEng = state->gpd.perParticleEng;
    perParticleEng.d_data.memset(0);
    for (Fix *f : state->fixes) {
        f->singlePointEng(perParticleEng.getDevData());
    }

}

void IntegratorUtil::forceSingle(bool computeVirials) {
    for (Fix *f : state->fixes) {
        if (f->forceSingle) {
            f->compute(computeVirials);
        }
    }
}


void IntegratorUtil::doDataComputation() {
    DataManager &dm = state->dataManager;
    int64_t turn = state->turn;
    bool computedAny = false;
    for (boost::shared_ptr<DataSetUser> ds : dm.dataSets) {
        if (ds->nextCompute == turn) {
            if (ds->requiresEnergy()) {
                dm.computeEnergy();
            }
            ds->computeData();
            computedAny = true;
        }
    }
    if (computedAny) {
        cudaDeviceSynchronize();
    }
    //append is after post_force, final step to keep to gpu busy while we do slow python list appending
}

void IntegratorUtil::doDataAppending() {
    DataManager &dm = state->dataManager;
    int64_t turn = state->turn; 
    for (boost::shared_ptr<DataSetUser> ds : dm.dataSets) {
        if (ds->nextCompute == turn) {
            ds->appendData();
            ds->setNextTurn(turn);
        }
    }
}



