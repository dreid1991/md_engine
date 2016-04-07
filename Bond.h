#pragma once
#ifndef BOND_H
#define BOND_H

#include <boost/variant.hpp>

#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"

/*! \brief Bond connecting atoms
 *
 * \link Atom Atoms\endlink can be connected by Bonds. Bonds are defined by a
 * potential depending on the separation of the two bonded \link Atom
 * Atoms\endlink. Thus, Bonds do not exist per se in the simulation and Bonds
 * can cross each other if the interaction parameters are not chosen in a way
 * that prohibits bond crossing.
 */
class Bond {
    public:
        Atom *atoms[2]; //!< Pointer to the bonded atoms

        /*! \brief Equality operator
         *
         * \param other Bond to compare this Bond to
         * \return True if both bonds connect the same atoms
         *
         * Two bonds are considered equal if they connect the same atoms. This
         * holds also if they are not of the same type.
         */
		bool operator==(const Bond &other ) {
			return (atoms[0] == other.atoms[0] || atoms[0] == other.atoms[1]) && (atoms[1] == other.atoms[0] || atoms[1] == other.atoms[1]);
		}

        /*! \brief Not-equal operator
         *
         * \param other Bond to compare this Bond to
         * \return True if both bonds connect different atoms
         *
         * See Bond::operator==()
         *
         * \todo This could be simplified by using the equality operator
         */
		bool operator!=(const Bond &other ) {
			return not ((atoms[0] == other.atoms[0] || atoms[0] == other.atoms[1]) && (atoms[1] == other.atoms[0] || atoms[1] == other.atoms[1]));
		}

        /*! Get index of one of the connected atoms
         *
         * \param x Specify which atom of the bond (either 0 or 1)
         * \return Index of the requested atom
         *
         * Return the index of one of the \link Atom Atoms\endlink connected
         * via the bond.
         */
		int getAtomId(int x) const {
			assert(x==0 or x==1);
			Atom *a = atoms[x];
			return a->id;
		}

        /*! \brief Check if a given Atom is part of this bond
         *
         * \param a Pointer to Atom to check
         * \return True if Atom is part of the Bond
         */
		bool hasAtom(Atom *a);

        /*! \brief For one Atom in the bond, get the other Atom
         *
         * \param a Pointer to Atom in the Bond
         * \return Pointer to the other Atom in the Bond. If the given Atom is
         *         not part of the Bond, return NULL
         */
		Atom *other(const Atom *a) const;

        /*! \brief Return a copy of one of the two \link Atom Atoms\endlink
         *
         * \param i Index which of the two Atoms in the Bond to return (0 or 1)
         * \return Copy of the requested Atom
         *
         * \todo This will crash if index i is > 1
         */
        Atom getAtom(int i);

        /*! \brief Swap the position of the two atoms in the internal memory */
        void swap();
};

/*! \brief Bond with a harmonic potential (a spring)
 *
 * Bond with harmonic potential.
 *
 * \todo In LAMMPS k is, in fact, k/2. Specify this explicitely here.
 * \todo I suggest to move this to FixBondHarmonic. Otherwise, this class could
 *       get crowded.
 */
class BondHarmonic : public Bond {
	public:
        //offset is how you have to offset the second atom to be in the same periodic cell as the first
		double k; //!< Spring constant
		double rEq; //!< Equilibrium distance
        int type; //!< Bond type

        /*! \brief Default constructor
         *
         * \todo Set member variables to default values
         */
		BondHarmonic (){};

        /*! \brief Constructor
         *
         * \param k_ Spring constant
         * \param rEq_ Equilibrium distance
         * \param type_ Bond type
         *
         * This constructor sets the connected \link Atom Atoms\endlink to
         * NULL.
         */
		BondHarmonic (double k_, double rEq_, int type_=-1) {
            atoms[0] = (Atom *) NULL;
            atoms[1] = (Atom *) NULL;
			k=k_;
			rEq=rEq_;
            type=type_;
        }

        /*! \brief Constructor
         *
         * \param a Pointer to the first Atom in the Bond
         * \param b Pointer to the second Atom in the Bond
         * \param k_ Spring constant
         * \param rEq_ Equilibrium distance
         * \param type_ Bond type
         */
		BondHarmonic (Atom *a, Atom *b, double k_, double rEq_, int type_=-1) {
			atoms[0]=a;
			atoms[1]=b;
			k=k_;
			rEq=rEq_;
            type=type_;

		}

        /*! \brief Copy parameters from other bond
         *
         * \param b Other bond to copy parameters from
         */
        void takeValues(BondHarmonic &b) {
            k = b.k;
            rEq = b.rEq;
        }

		//Vector vectorFrom(Atom *);


};	

/*! \brief Harmonic Bond on the GPU with aligned memory
 *
 * Aligned memory means faster access. The Bond is stored for each Atom.
 */
class __align__(16) BondHarmonicGPU {
    public:
        int myId; //!< ID of this Atom
        int idOther; //!< ID of the other Atom in the Bond
        float k; //!< Spring constant
        float rEq; //!< Equlibrium distance

        /*! \brief Constructor
         *
         * \param myId_ Index of this Atom
         * \param idOther_ Index of the other Atom
         * \param k_ Spring constant
         * \param rEq_ Equilibrium distance
         */
        BondHarmonicGPU(int myId_, int idOther_, float k_, float rEq_) : myId(myId_), idOther(idOther_), k(k_), rEq(rEq_) {};

        /*! \brief Copy parameters from other bond
         *
         * \param b Other bond to copy parameters from
         */
        void takeValues(BondHarmonic &b) {
            k = b.k;
            rEq = b.rEq;
        }

        /*! \brief Default constructor */
        BondHarmonicGPU(){};
};

/*! \brief Harmonic Bond not derived from universal bond
 *
 * \todo Do we really continue to need this class?
 */
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

/*! \typedef Boost Variant for any bond */
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
