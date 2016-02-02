#include "Bond.h"

/*
Vector Bond::vectorFrom(Atom *a) {
	if (a==atoms[0]) {
		return atoms[1]->pos - atoms[0]->pos;
	} else if (a==atoms[1]) {
		return atoms[0]->pos - atoms[1]->pos;
	}
	return Vector();
}
*/
Vector Bond::vectorFrom(Atom *a, Vector &trace) {
    cout << "USING DEPRECATED vectorFrom FUNCTION" << endl;
    return Vector(0, 0, 0);
    /*
    Vector v1, v2;
	if (a==atoms[0]) {
		v1 = atoms[0]->pos;
		v2 = atoms[1]->pos + offset * trace;
        return v2 - v1;
	} else if (a==atoms[1]) {
		v1 = atoms[1]->pos + offset * trace;
		v2 = atoms[0]->pos;
        return v2 - v1;
	}
    return Vector();
    */
}

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
