#include "Fix.h"
#include "ReadConfig.h"
Fix::Fix(SHARED(State) state_, string handle_, string groupHandle_, string type_, int applyEvery_) : state(state_.get()), handle(handle_), groupHandle(groupHandle_), applyEvery(applyEvery_), type(type_), forceSingle(false), orderPreference(0), restartHandle(type + "_" + handle) {
    forceSingle = false;
    updateGroupTag();
    if (state->readConfig->fileOpen) {
        auto restData = state->readConfig->readNode(restartHandle);
        if (restData) {
            cout << "Reading restart data for fix " << handle << endl;
            readFromRestart(restData);
        }
    }
}


void Fix::updateGroupTag() {
	map<string, unsigned int> &groupTags = state->groupTags;
	if (groupHandle == "None") {
		groupTag = 0;
	} else {
		assert(groupTags.find(groupHandle) != groupTags.end());
		groupTag = groupTags[groupHandle];
	}
}
bool Fix::isEqual(Fix &f) {
	return f.handle == handle;
}

/*
vector<pair<int, vector<int> > > Fix::neighborlistExclusions() {
    return vector<pair<int, vector<int> > >();
};
*/

void Fix::validAtoms(vector<Atom *> &atoms) {
    for (int i=0; i<atoms.size(); i++) {
        if (!state->validAtom(atoms[i])) {
            cout << "Tried to create for " << handle << " but atom " << i << " was invalid" << endl;
            assert(false);
        }
    }
}

void export_Fix() {
    boost::python::class_<Fix> (
        "Fix"
    )
    .def_readonly("handle", &Fix::handle)
    .def_readwrite("applyEvery", &Fix::applyEvery)
    .def_readwrite("groupTag", &Fix::groupTag)
    ;

}
