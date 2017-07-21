#pragma once
#ifndef FIXRIGID_H
#define FIXRIGID_H

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include "Python.h"
#include "Fix.h"
#include "FixBond.h"
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python/list.hpp>
#include "GPUArrayDeviceGlobal.h"

void export_FixRigid();

// simple class holding static data associated with a given water model
// -- namely, masses & corresponding weights of hydrogens, oxygen;
//    and the geometry of the speciic model
class FixRigidData {
    public:
        // bond lengths - as OH1, OH2, HH, OM
        double4 sideLengths;

        // canonical lengths, with center of mass as origin
        // as (ra, rb, rc, 0.0)
        double4 canonicalTriangle;

        // mass weights - make these double precision!
        // --- note - useful operator definitions were provided in cutils_func and cutils_math..
        //     --- should do this for double2, double3, double4 etc.
        // -- weights.x = massO / (massWater); 
        //    weights.y = massH / massWater;
        //    weights.z = massO;
        //    weights.w = massH;
        double4 weights;
        
        // no point in reducing precision here; invariant parameters of the triangle - 
        // cosines of the apex angles
        double cosA;
        double cosB;
        double cosC;
        
        // so a few things worth calculating beforehand...
        // --- Miyamoto, equations B2, constants in the tau expressions
        double tauAB1; // (m_a / d) * ( 2*(m_a + m_b) - m_a cos^2 C )
        double tauAB2; // (m_a / d) * (m_b cosC cosA - (m_a + m_b) * cosB)
        double tauAB3; // (m_a / d) * ( m_a cosB cosC - 2 * m_b cosA )

        double tauBC1; // ( (m_a + m_b)^2 - (m_b*m_b*cosA*cosA) ) / d 
        double tauBC2; // ( m_a * (m_b * cosA * cosB - (m_a + m_b) * cosC) ) / d
        double tauBC3; // ( m_a * (m_b * cosC * cosA - (M_a + m_b) * cosB) ) / d

        double tauCA1; // ( m_a / d) * ( 2 * (m_a + m_b) - m_a * cosB * cosB)
        double tauCA2; // ( m_a / d) * ( m_a * cosB * cosC - 2 * m_b * cosA )
        double tauCA3; // ( m_a / d) * ( m_b * cosA * cosB - (m_a + m_b) * cosC ) 

        double denominator; // 'd' in expression B2; a constant of the rigid geometry

        FixRigidData() {};
};


class FixRigid : public Fix {
    private:
        // array holding ids of a given water molecule
        GPUArrayDeviceGlobal<int4> waterIdsGPU;

        // array holding positions
        GPUArrayDeviceGlobal<float4> xs_0;

        // array holding velocities before constraints are applied
        GPUArrayDeviceGlobal<float4> vs_0;


        GPUArrayDeviceGlobal<float4> dvs_0;


        GPUArrayDeviceGlobal<float4> fs_0;


        GPUArrayDeviceGlobal<float4> com;

        
        // vector of booleans that will alert us if a molecule has unsatisfied constraints at the end of the turn
        GPUArrayDeviceGlobal<bool> constraints;


        std::vector<int4> waterIds;


        std::vector<BondVariant> bonds;


        std::vector<float4> invMassSums;


        // boolean defaulting to false in the constructor, denoting whether this is TIP4P/2005
        bool TIP4P;

        // boolean defaulting to false in the constructor, denoting whether this is TIP3P
        bool TIP3P;

        // computes the force partition constant for TIP4P for modification of forces on the molecule
        void compute_gamma();

        // the force partition constant to distribute force from M-site to O, H, H atoms
        //  See Feenstra, Hess, and Berendsen, J. Computational Chemistry, Vol. 20, No. 8, 786-798 (1999)
        //  -- specifically, appendix A, expression 6
        float gamma;

        int nMolecules;

        // local constants to be set for assorted supported water models
        // sigma_O is for scaling of bond lengths if LJ units are used
        double sigma_O;
        double r_OH;
        double r_HH;
        double r_OM;

        // a float 4 of the above measures r_OH, r_HH, r_OM;
        double4 fixedSides;

        // data about this water model
        FixRigidData fixRigidData;
        // 
        void set_fixed_sides();

        void setStyleBondLengths();
        // 
    public:


        //! Constructor
        /*!
        * \param state Pointer to the simulation state
        * \param handle "Name" of the Fix
        * \param groupHandle String specifying group of atoms this Fix acts on
        */
        FixRigid(SHARED(State), std::string handle_, std::string groupHandle_);


        //! First half step of the integration
        /*!
         * \return Result of the FixRigid::stepInit() call.
         */
        bool stepInit();

        //! Second half step of the integration
        /*!
         * \return Result of FixRigid::stepFinal() call.
         */
        bool stepFinal();

        // populate the FixRigidData instance
        void populateRigidData();

        //! Prepare FixRigid for simulation run
        bool prepareForRun();
        
        // the style of a given water model to be used.  Defaults to 'DEFAULT',  
        // either TIP3P or TIP4P/2005 depending on whether the model is 3-site or 4-site
        std::string style;

        // permits variants of a given style (e.g., TIP4P, TIP4P/LONG, TIP4P/2005)
        // default styles are TIP3P and TIP4P/2005
        void setStyle(std::string);


        //! Halfstep solution to velocity constraints
        bool postNVE_V();

        //! Reset the position of the M-site after integrating the position of the molecule.
        //  -- Note that we do /not/ solve the rigid body constraints at this point;
        //     However, the massless site does not evolve with the other atoms of the molecule (if TIP4P);
        //     So, manually re-set it here for accuracy.
        //     Also, check if this is rigorously correct or a valid approximation. \TODO/
        void handleBoundsChange();

        //! Create a rigid constraint on a TIP3P water molecule
        /*!
         * \ param id_a The atom id in the simulation state of the Oxygen atom in TIP3P
         * \ param id_b The atom id in the simulation state of a Hydrogen atom in TIP3P
         * \ param id_c The atom id in the simulation state of a Hydrogen atom in TIP3P
         */
        void createRigid(int, int, int);

        //! Create a rigid constraint on a TIP4P/2005 water molecule
        /*!
         * \ param id_a The atom id in the simulation state of the Oxygen atom in TIP4P/2005
         * \ param id_b The atom id in the simulation state of a Hydrogen atom in TIP4P/2005
         * \ param id_c The atom id in the simulation state of a Hydrogen atom in TIP4P/2005
         * \ param id_d The atom id in the simulation state of the M site in TIP4P/2005
         */
        void createRigid(int, int, int, int);


        bool printing;
       

        bool firstPrepare;
        std::vector<BondVariant> *getBonds() {
            return &bonds;
  }
};


#endif /* FIXRIGID_H */
