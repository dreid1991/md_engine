#include "Molecule.h"

#include "boost_for_export.h"
namespace py = boost::python;
using namespace std;


Molecule::Molecule(State *state_, vector<int> &ids_) {
    state = state_;
    ids = ids_;
}

void Molecule::translate(Vector &v) {
    //implement please
}
void Molecule::rotate(Vector &around, Vector &axis, double theta) {
    //also this 
}

Vector Molecule::COM() {
    //and this
}
