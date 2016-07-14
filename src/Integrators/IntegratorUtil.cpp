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


void IntegratorUtil::doDataCollection() {
    DataManager &dm = state->dataManager;
    int64_t turn = state->turn;
    for (DataSet *ds : dm.dataSets) {
        if (ds->nextCollectTurn == turn) {
            ds->collect();
            ds->setNextCollectTurn(turn);
        }
    }
    cudaDeviceSynchronize();
    for (DataSet *ds : dm.dataSets) {
        ds->appendValues();
    }
}
