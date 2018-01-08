#include "Molecule.h"
#include "boost_for_export.h"
namespace py = boost::python;

#include "State.h"
#include "EigenInterface.h"

Molecule::Molecule(State *state_, std::vector<int> &ids_) {
    state = state_;
    ids = ids_;
}

void Molecule::translate(Vector &v) {
    for (int id : ids) {
        Atom &a = state->idToAtom(id);
        a.pos += v;
    }
}
void Molecule::rotate(Vector axis, double theta) {
    std::vector<double> valsAsFlat;
    Vector com = COM();

    for (int id : ids) {
        Atom &a = state->idToAtom(id);
        Vector posRel = a.pos - com;
        valsAsFlat.push_back(posRel[0]);
        valsAsFlat.push_back(posRel[1]);
        valsAsFlat.push_back(posRel[2]);
        
        //Eigen::Vector3d relEig = posEig-comEig;

    }
    double axis_list[3];
    for (int i=0; i<3; i++) {
        axis_list[i] = axis[i];
    }
    
    double com_list[3];
    for (int i=0; i < 3; i++) {
        com_list[i] = com[i];
    };

    // compute the rotations
    std::vector<double> newPositions = eigenInterface_rotate(valsAsFlat, axis_list, com_list, theta);
    
    int counter = 0;
    for (int id : ids) {
        Atom &a = state->idToAtom(id);
        int ix = counter * 3;
        int iy = counter * 3 + 1;
        int iz = counter * 3 + 2;
        double xPos = newPositions[ix];
        double yPos = newPositions[iy];
        double zPos = newPositions[iz];
        a.pos = Vector(xPos,yPos,zPos) + com;

        counter++;
    }


}


void Molecule::rotateRandom() {
    std::uniform_real_distribution<double> dist(0, 1);
    std::mt19937 &generator = state->getRNG();
    double z = dist(generator)*2-1;
    double theta = 2*M_PI*dist(generator);
    rotate(Vector(sqrt(1-z*z)*cos(theta), sqrt(1-z*z)*sin(theta), z), dist(generator)*2*M_PI);

}

std::vector<int> Molecule::getAtoms() {
    std::vector<int> atomIds;
    for (int id: ids) {
        atomIds.push_back(id);
    }
    return atomIds;
}


Vector Molecule::COM() {
    Vector weightedPos(0, 0, 0);
    double sumMass = 0;
    Vector firstPos = state->idToAtom(ids[0]).pos;
    Bounds bounds = state->bounds;
    for (int id : ids) {
        Atom &a = state->idToAtom(id);
        Vector pos = firstPos + bounds.minImage(a.pos - firstPos);
        double mass = a.mass;
        weightedPos += pos * mass;
        sumMass += mass;
    }
    return weightedPos / sumMass;
}

void Molecule::unwrap() {
    Vector weightedPos(0, 0, 0);
    double sumMass = 0;
    Vector firstPos = state->idToAtom(ids[0]).pos;
    Bounds bounds = state->bounds;
    for (int id : ids) {
		int idx = state->idToIdx[id];
		Atom &a = state->atoms[idx];
        a.pos = firstPos + bounds.minImage(a.pos - firstPos);

    }
}

double Molecule::dist(Molecule &other) {
    double minSqr = 1e9; //large value;
    Bounds bounds = state->bounds;
    for (int id : ids) {
        Vector pos = state->idToAtom(id).pos;
        for (int idOther : other.ids) {
            minSqr = fmin(minSqr, bounds.minImage((pos - state->idToAtom(idOther).pos)).lenSqr() );
        }
    }
    return sqrt(minSqr);
}
Vector Molecule::size() {
    Vector firstPos = state->idToAtom(ids[0]).pos;
    Vector lo = firstPos;
    Vector hi = firstPos;
    Bounds bounds = state->bounds;
    for (int id : ids) {
        Atom &a = state->idToAtom(id);
        Vector pos = firstPos + bounds.minImage(a.pos - firstPos);
        for (int i=0; i<3; i++) {
            lo[i] = fmin(lo[i], pos[i]);
            hi[i] = fmax(hi[i], pos[i]);
        }
    }
    return hi - lo;
    

}
void export_Molecule() {
    py::class_<Molecule> ("Molecule", py::no_init)
    .def_readonly("ids", &Molecule::ids)
    .def("translate", &Molecule::translate)
    .def("rotate", &Molecule::rotate, (py::arg("axis"), py::arg("theta")) )
    .def("rotateRandom", &Molecule::rotateRandom)
    .def("COM", &Molecule::COM)
    .def("dist", &Molecule::dist)
    .def("size", &Molecule::size)
    .def("unwrap", &Molecule::unwrap)
    ;
}
