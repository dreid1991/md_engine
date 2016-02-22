#ifndef BOUNDS_H
#define BOUNDS_H
#include "Python.h"
#include <stdio.h>
#include "Atom.h"
#include "BoundsGeneric.h"
#include <assert.h>
#include "globalDefs.h"
#include "BoundsGPU.h"

#include "boost_for_export.h"
void export_Bounds();

class State;
using namespace std;
class Bounds : public BoundsGeneric {
	public:
		State *state;

		Bounds() : isSet(false) {

		}
		Bounds(SHARED(State) state_, Vector lo_, Vector hi_) : BoundsGeneric(lo_, hi_), state(state_.get()), isSet(true) {
			handle2d();
		}
		Bounds(State *state_, Vector lo_, Vector hi_) : BoundsGeneric(lo_, hi_), state(state_), isSet(true) {
			handle2d();
		}
		Bounds(SHARED(State) state_, Vector lo_, Vector sides_[3]) : BoundsGeneric(lo_, sides_), state(state_.get()), isSet(true) {
			handle2d();
		}
		Bounds(State *state_, Vector lo_, Vector sides_[3]) : BoundsGeneric(lo_, sides_), state(state_), isSet(true) {
			handle2d();
		}
        BoundsGPU makeGPU();
		void handle2d();
	
		string asStr() {
			
			string loStr = "Lower bounds " + lo.asStr();
			string hiStr = "upper bounds " + hi.asStr();
			return loStr + ", " + hiStr ;
		}
		bool atomInBounds(Atom &);
		num volume();
		Bounds copy() {
			return *this;
		}
        bool operator==(const Bounds &other) {
            return lo==other.lo and hi==other.hi and sides[0]==other.sides[0] and sides[1]==other.sides[1] and sides[2]==other.sides[2];
        }
        bool operator!=(const Bounds &other) {
            return not(lo==other.lo and hi==other.hi and sides[0]==other.sides[0] and sides[1]==other.sides[1] and sides[2]==other.sides[2]);
        }
        void set(BoundsGPU &b) {
            lo = Vector(b.lo);
            hi = lo;
            trace = Vector();
            for (int i=0; i<3; i++) {
                sides[i] = Vector(b.sides[i]);
                trace += sides[i];
            }
            hi += trace;
        }
		void set(Bounds &b) {
			lo = b.lo;
			hi = b.hi;
			for (int i=0; i<3; i++) {
				sides[i] = b.sides[i];
			}
			trace = b.trace;
		}
        void setSides() {
            trace = hi-lo;
            for (int i=0; i<3; i++) {
                Vector v = Vector(0, 0, 0);
                v[i] = trace[i];
                sides[i] = v;
            }
        }
        void setPython(Bounds &b) {
            set(b);
        }
        bool skew(Vector );
		num getSkew(int idx);
		num getSkewX();
		num getSkewY();
		Bounds unskewed();
        bool isSkewed();	
        bool isSet;
        Vector minImage(Vector v);
};

//SHARED(Bounds) BoundsCreateSkew(  figure out how to create laters

#endif
