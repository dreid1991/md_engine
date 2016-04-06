#pragma once
#ifndef BOND_H
#define BOND_H

#include <boost/variant.hpp>

#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"


class Bond {
    public:
        Atom *atoms[2];
		bool operator==(const Bond &other ) {
			return (atoms[0] == other.atoms[0] || atoms[0] == other.atoms[1]) && (atoms[1] == other.atoms[0] || atoms[1] == other.atoms[1]);
		}
		bool operator!=(const Bond &other ) {
			return not ((atoms[0] == other.atoms[0] || atoms[0] == other.atoms[1]) && (atoms[1] == other.atoms[0] || atoms[1] == other.atoms[1]));
		}
		int getAtomId(int x) const {
			assert(x==0 or x==1);
			Atom *a = atoms[x];
			return a->id;
		}
		bool hasAtom(Atom *);
		Atom *other(const Atom *) const;
        Atom getAtom(int i);
        void swap();
};

class BondHarmonic : public Bond {
	public:
        //offset is how you have to offset the second atom to be in the same periodic cell as the first
		double k;
		double rEq;
        int type;
		BondHarmonic (){};
		BondHarmonic (double k_, double rEq_, int type_=-1) {
            atoms[0] = (Atom *) NULL;
            atoms[1] = (Atom *) NULL;
			k=k_;
			rEq=rEq_;
            type=type_;
        }
		BondHarmonic (Atom *a, Atom *b, double k_, double rEq_, int type_=-1) {
			atoms[0]=a;
			atoms[1]=b;
			k=k_;
			rEq=rEq_;
            type=type_;

		}
        void takeValues(BondHarmonic &b) {
            k = b.k;
            rEq = b.rEq;
        }

		//Vector vectorFrom(Atom *);


};	

class __align__(16) BondHarmonicGPU {
    public:
        int myId;
        int idOther;
        float k;
        float rEq;
        BondHarmonicGPU(int myId_, int idOther_, float k_, float rEq_) : myId(myId_), idOther(idOther_), k(k_), rEq(rEq_) {};
        void takeValues(BondHarmonic &b) {
            k = b.k;
            rEq = b.rEq;
        }
        BondHarmonicGPU(){};
};

class BondSave {
	public:
		double k;
		double rEq;
		int ids[2];
		BondSave(){};
		BondSave (int *ids_, double k_, double rEq_) {
			ids[0] = ids_[0];
			ids[1] = ids_[1];
			k = k_;
			rEq = rEq_;
		}
		int get (int i) {
			return ids[i];
		}
		void set (int i, double x) {
			ids[i] = x;
		}
		bool operator == (const BondSave &other) {
			return k==other.k and rEq==other.rEq and other.ids[0]==ids[0] and other.ids[1]==ids[1];
		}
		bool operator != (const BondSave &other) {
			return not (k==other.k and rEq==other.rEq and other.ids[0]==ids[0] and other.ids[1]==ids[1]);
		}
};

// lets us store a list of vectors to any kind of bonds we want
typedef boost::variant<
	BondHarmonic, 
	Bond
> BondVariant;

/* ugh
class BondVariantIterator : public std::iterator<std::input_iterator_tag, BondVariant> {
    BondVariant *outer;
    Bond *itBond;
    vector<BondHarmonic> *itBondHarmonic;

    BondVariantIterator &operator++() {
		if(std::vector<Bond> *b = boost::get<std::vector<Bond>>(outer)) {
			if (
			++itBond;
		} else if(std::vector<BondHarmonic> *bh = boost::get<std::vector<BondHarmonic>>(itr)) {
			++itBondHarmonic;
		}
		++itr;
        return *this;
    }
}
*/

#endif
