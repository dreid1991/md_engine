#include "Fix.h"

#include <iostream>

#include "Atom.h"
#include "boost_for_export.h"
#include "list_macro.h"
#include "ReadConfig.h"
#include "State.h"

Fix::Fix(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
         std::string type_, bool forceSingle_, int applyEvery_,
         int orderPreference_)
    : state(state_.get()), handle(handle_), groupHandle(groupHandle_),
      type(type_), forceSingle(forceSingle_), applyEvery(applyEvery_),
      orderPreference(orderPreference_), restartHandle(type + "_" + handle)
{
    updateGroupTag();

    /*if (state->readConfig->fileOpen) {
      auto restData = state->readConfig->readFix(type, handle);
        if (restData) {
            std::cout << "Reading restart data for fix " << handle << std::endl;
            readFromRestart(restData);
        }
	}*/
}

bool Fix::isEqual(Fix &f) {
    return f.handle == handle;
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
    boost::python::class_<Fix, boost::noncopyable> (
        "Fix",
        boost::python::no_init
    )
    .def_readonly("handle", &Fix::handle)
    .def_readonly("type", &Fix::type)
    .def_readwrite("applyEvery", &Fix::applyEvery)
    .def_readwrite("groupTag", &Fix::groupTag)
    ;

}

