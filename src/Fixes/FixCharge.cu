#include "FixCharge.h"

#include "State.h"

namespace py = boost::python;

FixCharge::FixCharge(SHARED(State) state_, string handle_, string groupHandle_,
                     string type_, bool forceSingle_)
    : Fix(state_, handle_, groupHandle_, type_, forceSingle_, 1) {
};


bool FixCharge::prepareForRun() {
    //check for electo neutrality
    double sum=0;
    for (int i=0;i<state->atoms.size();i++)
	sum+=state->atoms[i].q;
    
    if (sum!=0.0) cout<<"System is not electroneutral. Total charge is "<<sum<<'\n';
    return true;
}

void export_FixCharge() {
    py::class_<FixCharge, SHARED(FixCharge), py::bases<Fix> > (
        "FixCharge",
        boost::python::no_init
    );
}
