#include "Fix.h"
#include "ReadConfig.h"
Fix::Fix(SHARED(State) state_,
         std::string handle_,
         std::string groupHandle_,
         std::string type_,
         int applyEvery_)
    : state(state_.get()),
      handle(handle_),
      groupHandle(groupHandle_),
      type(type_),
      applyEvery(applyEvery_),
      forceSingle(false),
      orderPreference(0),
      restartHandle(type + "_" + handle)
{
    updateGroupTag();
    if (state->readConfig->fileOpen) {
        auto restData = state->readConfig->readNode(restartHandle);
        if (restData) {
            std::cout << "Reading restart data for fix " << handle << std::endl;
            readFromRestart(restData);
        }
    }
}


void Fix::updateGroupTag() {
	std::map<std::string, unsigned int> &groupTags = state->groupTags;
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

void Fix::validAtoms(std::vector<Atom *> &atoms) {
    for (int i=0; i<atoms.size(); i++) {
        if (!state->validAtom(atoms[i])) {
            std::cout << "Tried to create for " << handle
                      << " but atom " << i << " was invalid" << std::endl;
            assert(false);
        }
    }
}

void export_Fix() {
    boost::python::class_<Fix> (
        "Fix"
    )
    .def_readonly("handle", &Fix::handle)
    .def_readonly("type", &Fix::type)
    .def_readwrite("applyEvery", &Fix::applyEvery)
    .def_readwrite("groupTag", &Fix::groupTag)
    ;

}
