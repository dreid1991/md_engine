#include "Bond.h"



bool Bond::hasAtom(Atom *a) {
	return atoms[0] == a or atoms[1] == a;
}

Atom *Bond::other(const Atom *a) const {
	if (atoms[0] == a) {
		return atoms[1];
	} else if (atoms[1] == a) {
		return atoms[0];
	} 
	return NULL;
}

Atom Bond::getAtom(int i) {
    return *atoms[i];
}

void Bond::swap() {
    Atom *x = atoms[0];
    atoms[0] = atoms[1];
    atoms[1] = x;
}
