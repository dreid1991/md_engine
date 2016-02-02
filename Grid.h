#ifndef GRID_H
#define	GRID_H
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <assert.h>
#include "Bounds.h"
#include <vector>
#include "Vector.h"
#include "GridGPU.h"

using namespace std;
template <class T>
class Grid {
	public:
		vector<T> saved;
		vector<T> raw;
		Vector trace;	

		VectorInt ns;
		Vector ds;
		Vector os;

		Vector dsOrig;
		
		T fillVal;
		bool is2d;
		Grid(){};
			
		Grid(VectorInt ns_, Vector ds_, Vector os_, T fillVal_) : ns(ns_), ds(ds_), os(os_), dsOrig(ds), fillVal(fillVal_) {
			assert(ns[0] > 0 and ns[1] > 0 and ns[2] > 0);
			int n = ns[0]*ns[1]*ns[2];
			raw.reserve(n);
			for (int i=0; i<n; i++) {
				raw.push_back(fillVal);
			}
		};
		void fillVals() {
			assert(ns[0] > 0 and ns[1] > 0 and ns[2] > 0);
			int n = ns[0]*ns[1]*ns[2];
			raw = vector<T>();
			raw.reserve(n);
			for (int i=0; i<n; i++) {
				raw.push_back(fillVal);
			}
		}
		int loopDim(const int x, const int nd, int *loop) { //only works if one of fewer grids offset.  Could do some flooring, but meh.  Not a relevant case.
			if (x < 0) {
				*loop = -1;
				return nd + x;
			} else if (x >= nd) {
				*loop = 1;
				return x - nd;
			} 
			*loop = 0;
			return x;

		};
		T &operator()(const int x, const int y, const int z) {
		//	int idx = x*ny*nz + y*nz + z;
		//	if (idx >= raw.size()) {
		//		cout << "GETTING OUT OF BOUNDS GRID SQUARE" << endl;cout.flush();
		//	}
			return raw[x*ns[1]*ns[2] + y*ns[2] + z];
		};
		int idxFromCoord(VectorInt &coord) {
			return coord[0] * ns[1]*ns[2] + coord[1]*ns[2] + coord[2];
		}
		T &operator()(Vector &v) {
			VectorInt coord = (v - os) / ds;
			return raw[idxFromCoord(coord)];
		};
		T &operator() (int coords[3], int didLoop[3]) {
			int loopRes[3];
			for (int i=0; i<3; i++) {
				loopRes[i] = loopDim(coords[i], ns[i], &(didLoop[i]));
			}
			return (*this)(loopRes[0], loopRes[1], loopRes[2]);
		};

		Vector pos(const int x, const int y, const int z) {
			return os + Vector(ds[0]*x, ds[1]*y, ds[2]*z);
		};
		int idxFromPos(Vector &v) {
			VectorInt coord = (v - os) / ds;
			return idxFromCoord(coord);
		}
		Vector pos(int i) {
			int numInSlice = ns[1] * ns[2];
			int nx = i / numInSlice;
			i -= nx * numInSlice;
			int ny = i / ns[2];
			i -= ny * ns[2];
			int nz = i;
			return pos(nx, ny, nz);
		}
		Vector posElem(T *elem) {
			int i = elem - &raw[0];
			if (i >= 0 and i < raw.size()) {
				return pos(i);
			}
			return Vector(0, 0, 0);
		}
		void setRaw(vector<T> vals) {
			raw = vals;
		};
		void setSaved(vector<T> vals) {
			saved = vals;
		}
		void reset() {
			raw = saved;
		}
		void saveRaw() {
			saved = raw;
		}
		vector<OffsetObj<T*> > getNeighbors(int coords[3], bool loops[3], Vector trace) {
			const int x = coords[0];
			const int y = coords[1];
			const int z = coords[2];
			int zBounds[2];
			if (not is2d) {
				zBounds[0] = z-1;
				zBounds[1] = z+1;
			} else {
				zBounds[0] = z;
				zBounds[1] = z;
			}
			vector<OffsetObj<T*> > neighbors;
			for (int i=x-1; i<=x+1; i++) {
				for (int j=y-1; j<=y+1; j++) {
					for (int k=zBounds[0]; k<=zBounds[1]; k++) {
						if (not (i==x and j==y and k==z)) {
							//Vector v(i, j, k);
							int boxCoords[3];
							boxCoords[0] = i;
							boxCoords[1] = j;
							boxCoords[2] = k;
							int didLoop[3] = {0, 0, 0};
							Vector offset;
							T *neigh = &(*this)(boxCoords, didLoop);
							bool append = true;
							for (int i=0; i<3; i++) {
								append = append and (not didLoop[i] or (didLoop[i] and loops[i]));
							}
							if (append) {
								for (int i=0; i<3; i++) {
									offset[i] = didLoop[i];
								}
								neighbors.push_back(OffsetObj<T*>(neigh, offset));
							}

						}
					}
				}
			}
			return neighbors;

		}
		vector<OffsetObj<T> > getNeighborVals(int coords[3], bool loops[3], Vector trace) {
			const int x = coords[0];
			const int y = coords[1];
			const int z = coords[2];
			vector<OffsetObj<T> > neighbors;
			for (int i=x-1; i<=x+1; i++) {
				for (int j=y-1; j<=y+1; j++) {
					for (int k=z-1; k<=z+1; k++) {
						if (not (i==x and j==y and k==z)) {
							//Vector v(i, j, k);
							int boxCoords[3];
							boxCoords[0] = i;
							boxCoords[1] = j;
							boxCoords[2] = k;
							int didLoop[3] = {0, 0, 0};
							double offsets[3] = {0, 0, 0};
							T neigh = (*this)(boxCoords, didLoop);
							bool append = true;
							for (int i=0; i<3; i++) {
								append = append and (not didLoop[i] or (didLoop[i] and loops[i]));
							}
							if (append) {
								for (int i=0; i<3; i++) {
									offsets[i] = didLoop[i] * trace[i];
								}
								Vector offset(offsets);
								neighbors.push_back(OffsetObj<T>(neigh, offset));
							}

						}
					}
				}
			}
			return neighbors;

		}

	
};
#endif
