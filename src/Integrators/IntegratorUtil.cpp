#include "IntegratorUtil.h"

#include "State.h"

IntegtatorUtil::IntegratorUtil(State *state_) {
    state = state_;
}

void Integrator::force(bool computeVirials) {
    int simTurn = state->turn;
    vector<Fix *> &fixes = state->fixes;
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
        if (ds->nextCollectTurn == turn) {
            if (ds->requiresEnergy) {
                dm.computeEnergy();
            }
            ds->computeData();
            ds->setNextTurn(turn);
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
    for (boost::shared_ptr<DataSetUser> ds : dm.dataSets) {
        if (ds->nextCollectTurn == turn) {
            ds->appendData();
        }
    }
}



