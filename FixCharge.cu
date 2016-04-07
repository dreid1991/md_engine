#include "FixCharge.h"



FixCharge::FixCharge(SHARED(State) state_, string handle_, string groupHandle_, string type_) : Fix(state_, handle_, groupHandle_, type_, 1) {
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
    boost::python::class_<FixCharge,
                          SHARED(FixCharge),
                          boost::python::bases<Fix> > (
        "FixCharge",
        boost::python::init<SHARED(State), string, string,string> (
            boost::python::args("state", "handle", "groupHandle","type"))
    )
    ;
}
