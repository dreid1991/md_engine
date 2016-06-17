#pragma once
#ifndef BOND_H
#define BOND_H

#include <boost/variant.hpp>
#include "xml_func.h"
#include "globalDefs.h"
#include "Atom.h"
#include <array>

/*! \brief Bond connecting atoms
 *
 * \link Atom Atoms\endlink can be connected by Bonds. Bonds are defined by a
 * potential depending on the separation of the two bonded \link Atom
 * Atoms\endlink. 
 */
class Bond {
    public:
        std::array<int, 2> ids;
        int type; //!< Bond type
        /*! \brief Default constructor
         *
         * \todo Set member variables to default values
         */
                Bond (){};        
        /*! \brief Constructor
         *
         * \param a Pointer to the first Atom in the Bond
         * \param b Pointer to the second Atom in the Bond
         * \param k_ Spring constant
         * \param rEq_ Equilibrium distance
         * \param type_ Bond type
         */
        Bond(Atom *a, Atom *b,  int type_=-1) {
            ids[0] = a->id;
            ids[1] = b->id;
            type=type_;
        }
        Bond (int type_) {
            type = type_;
        }

        /*! Get index of one of the connected atoms
         *
         * \param x Specify which atom of the bond (either 0 or 1)
         * \return Index of the requested atom
         *
         * Return the index of one of the \link Atom Atoms\endlink connected
         * via the bond.
         */

        /*! \brief Check if a given Atom is part of this bond
         *
         * \param a Pointer to Atom to check
         * \return True if Atom is part of the Bond
         */
		bool hasAtomId(int id);

        /*! \brief For one Atom in the bond, get the other Atom
         *
         * \return id of other atom in bond or -1 if not in bond
         */
		int otherId(int id) const;

        /*! \brief Swap the position of the two atoms in the internal memory */
        //void swap();

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
        /*! \brief Default constructor
         *
         * \todo Set member variables to default values
         */
		BondHarmonic (){};

		BondHarmonic (Atom *a, Atom *b, double k_, double rEq_, int type_=-1) : Bond(a,b,type_) {
			k=k_;
			rEq=rEq_;
		}
        BondHarmonic (double k_, double rEq_, int type_) : Bond(type) {
			k=k_;
			rEq=rEq_;
        }

        /*! \brief Copy parameters from other bond
         *
         * \param b Other bond to copy parameters from
         */
        void takeParameters(BondHarmonic &b) {
            k = b.k;
            rEq = b.rEq;
        }

		//Vector vectorFrom(Atom *);
	std::string getInfoString();
	bool readFromRestart(pugi::xml_node restData);
};	
void export_BondHarmonic();

/*! \brief Harmonic Bond on the GPU 
 *
 * 
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
        void takeParameters(BondHarmonic &b) {
            k = b.k;
            rEq = b.rEq;
        }

        /*! \brief Default constructor */
        BondHarmonicGPU(){};
};


/*! \typedef Boost Variant for any bond */
typedef boost::variant<
	BondHarmonic, 
	Bond
> BondVariant;


#endif
