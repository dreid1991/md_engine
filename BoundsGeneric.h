#ifndef BOUNDS_GENERIC_H
#define BOUNDS_GENERIC_H

#include "Vector.h"
#include <string>
#include <iostream>

class BoundsGeneric {
	public:
		Vector lo;
		Vector hi;
		Vector trace;
		Vector sides[3];
		BoundsGeneric() {

		}
		BoundsGeneric(Vector lo_, Vector hi_) : lo(lo_), hi(hi_), trace(hi - lo) {
			for (int i=0; i<3; i++) {
				sides[i] = Vector();
				sides[i][i] = hi[i] - lo[i];
			}
		}
		BoundsGeneric(Vector lo_, Vector sides_[3]) : lo(lo_), hi(lo+sides_[0]+sides_[1]+sides_[2]), trace(hi-lo) {
			for (int i=0; i<3; i++) {
				sides[i] = sides_[i];
			}
		}
		
		string asStr() {
			
			string loStr = "Lower bounds " + lo.asStr();
			string hiStr = "upper bounds " + hi.asStr();
			return loStr + ", " + hiStr ;
		}
		void resize(int dim, double mult, double around) {
			double hiDim = hi[dim];
			double loDim = lo[dim];
			double origin = loDim + around * (hiDim - loDim);
			hi[dim] = mult * (hiDim - origin) + origin;
			lo[dim] = mult * (loDim - origin) + origin;
			trace[dim] *= mult;
		}
		bool operator == (BoundsGeneric &b) {
			for (int i=0; i<3; i++) {
				if (not (fabs(b.lo[i] - lo[i])<EPSILON and fabs(b.hi[i] - hi[i])<EPSILON)) {
					return false;
				}
			}
			return true;
		}
        Vector getSide(int i) {
            if (i < 3 and i >=0) {
                return sides[i];
            } 
            return Vector();
        }
        void setSide(int idx, Vector v) {
            sides[idx] = v;
        }
};
#endif
