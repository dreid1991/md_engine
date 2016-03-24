#ifndef ATOM_H
#define ATOM_H
#include "Vector.h"
#include <vector>
#include "OffsetObj.h"

void export_Atom();
void export_Neighbor();
class Atom;
typedef OffsetObj<Atom *> Neighbor;
typedef vector<Atom *> atomlist;
class Atom {
	public:
		Vector pos;
		Vector vel;
		Vector force;
		Vector forceLast;
		//Vector posAtNeighborListing; //can do this elsewhere
		Atom *next;
		double mass;
        double q;
		int type;  //do this here, since otherwise would have to get it from some other array in the heap
		int id;
		uint32_t groupTag;
		vector<Neighbor> neighbors;
		Atom (int type_, int id_) : next((Atom *) NULL), mass(0), q(0), type(type_), id(id_), groupTag(1) {
			mass=-1;
		};
		Atom (Vector pos_, int type_, int id_) : pos(pos_), next((Atom *) NULL), mass(0), q(0), type(type_), id(id_), groupTag(1) {
			mass=-1;
		};
		Atom (Vector pos_, int type_, int id_, double mass_, double q_) : pos(pos_), next((Atom *) NULL), mass(mass_), q(q_), type(type_), id(id_), groupTag(1) {
		};
		Atom () {
			groupTag=1;
			id=0;
			mass=-1;
		};
		double kinetic() {
			return 0.5 * mass * vel.lenSqr();
		}
		bool operator==(const Atom &other) {
			return id==other.id;
		}
		bool operator!=(const Atom &other) {
			return id!=other.id;
		}


};

#endif


