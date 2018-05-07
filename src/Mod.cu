#include "Mod.h"
#include "State.h"

#include "Bond.h"
#include "Angle.h"
#include "Vector.h"
#include "helpers.h"
#include "Fix.h"

/* On adding 'real' types...
 You're going to have a bad time if you decide to 
 use namespace std;
*/

	
__global__ void Mod::unskewAtoms(real4 *xs, int nAtoms, real3 xOrig, real3 yOrig, real3 lo) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        real lxo = length(xOrig);
        real lyo = length(yOrig);
        real lxf = xOrig.x;
        real lyf = yOrig.y;
       
        real a = atan2(xOrig.y, xOrig.x);
        real b = atan2(yOrig.x, yOrig.y);

        real invDenom = 1.0 / (lxo*lyo*cos(a)*cos(b) - lxo*lyo*sin(a)*sin(b));

        real c1 = lyo*cos(b) * invDenom;
        real c2 = -lyo*sin(b) * invDenom;
        real c3 = -lxo*sin(a) * invDenom;
        real c4 = lxo*cos(a) * invDenom;
        


        real4 pos = xs[idx];
        real xo = pos.x - lo.x;
        real yo = pos.y - lo.y;
        pos.x = lxf * (xo*c1 + yo*c2) + lo.x;
        pos.y = lyf * (xo*c3 + yo*c4) + lo.y;
        xs[idx] = pos;
    }
}


__global__ void Mod::skewAtomsFromZero(real4 *xs, int nAtoms, real3 xFinal, real3 yFinal, real3 lo) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        const double a = atan2(xFinal.y, xFinal.x);
        const double b = atan2(yFinal.x, yFinal.y);
        const double lxf = length(xFinal);
        const double lyf = length(yFinal);
        
        const double lxo = xFinal.x;
        const double lyo = yFinal.y;

        const double c1 = lxf*cos(a);
        const double c2 = lyf*sin(b);

        const double c3 = lxf*sin(a);
        const double c4 = lyf*cos(b);


        real4 pos = xs[idx];

        real xo = pos.x - lo.x;
        real yo = pos.y - lo.y;

        const double fx = xo / lxo;
        const double fy = yo / lyo;
        pos.x = fx * c1 + fy * c2 + lo.x;
        pos.y = fx * c3 + fy * c4 + lo.y;

        xs[idx] = pos;
    }
}

__global__ void FDotR_cu(int nAtoms, real4 *xs, real4 *fs, Virial *virials) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        real3 x = make_real3(xs[idx]);
        //f only has pair-wise forces right now
        real3 f = make_real3(fs[idx]);
        //virial is zero at this point.  Only time f dot r in valid
        Virial v(0, 0, 0, 0, 0, 0);
        computeVirial(v, f, x);
        virials[idx] = v;
    }
}

template <bool RIGIDBODIES>
__global__ void Mod::scaleCentroids_cu(real4 *xs, int nAtoms,int nPerRingPoly, real3 scaleBy,
                                    int* idToIdxs, BoundsGPU oldBounds,BoundsGPU newBounds) {

    int idx     = GETIDX();
    int rootIdx = threadIdx.x * nPerRingPoly;
    extern __shared__ real3 deltas[];

    int nRingPoly = nAtoms / nPerRingPoly;
    if (idx < nRingPoly) {
        // determine centroid position for this ring polymer
        int baseIdx = idx*nPerRingPoly;
        real3 init     = make_real3(xs[baseIdx]);
        real3 diffSum  = make_real3(0,0,0);
        deltas[rootIdx] = init; 
        for (int i = 1;i<nPerRingPoly;i++) {
            real3 next = make_real3(xs[baseIdx+i]);
            real3 dx = oldBounds.minImage(next-init);
            deltas[rootIdx + i] = next; 
            diffSum += dx;
        }

        diffSum /= nPerRingPoly;
        real3 unwrappedPos = init + diffSum;
        real3 trace = oldBounds.trace();
        real3 diffFromLo = unwrappedPos - oldBounds.lo;
        real3 imgs = floorf(diffFromLo / trace);
        real3 wrappedPos = unwrappedPos - trace * imgs * oldBounds.periodic;

        // compute the differences from the centroid
        for (int i = rootIdx;i<rootIdx + nPerRingPoly;i++) {
            deltas[i] = oldBounds.minImage(deltas[i]-wrappedPos);
        }

        // now find its scaled position
        // need new bounds here!!!!
        real3 center = newBounds.lo + newBounds.rectComponents * 0.5f;
        real3 newRel = (wrappedPos - center)*scaleBy;
        wrappedPos = center+newRel - trace * imgs * newBounds.periodic;

        // reset the relative positions of the ring polymer based on new centroid
        for ( int i = 0; i<nPerRingPoly; i++) {
            real3 newPos = wrappedPos + deltas[rootIdx+i];
            //real3 newPos = center + newRel + deltas[rootIdx+i];
            real3 diffFromLo = newPos - newBounds.lo;
            newPos       -= trace*floorf(diffFromLo / trace)*newBounds.periodic;
            real4 posWhole = xs[baseIdx+i];
            posWhole.x    = newPos.x;
            posWhole.y    = newPos.y;
            posWhole.z    = newPos.z;
            xs[baseIdx+i] = posWhole;
        }
    }
}

template <bool RIGIDBODIES>
__global__ void Mod::scaleCentroidsGroup_cu(real4 *xs, int nAtoms,int nPerRingPoly, real3 scaleBy,uint32_t groupTag,
                                    real4 *fs, int* idToIdxs, BoundsGPU oldBounds,BoundsGPU newBounds) {

    int idx     = GETIDX();
    int rootIdx = threadIdx.x * nPerRingPoly;
    extern __shared__ real3 deltas[];

    int nRingPoly = nAtoms / nPerRingPoly;
    if (idx < nRingPoly) {
        // determine centroid position for this ring polymer
        int baseIdx = idx*nPerRingPoly;
        uint32_t tag = * (uint32_t *) &(fs[baseIdx].w);
        if (tag & groupTag) {
            real3 init     = make_real3(xs[baseIdx]);
            real3 diffSum  = make_real3(0,0,0);
            deltas[rootIdx] = init; 
            for (int i = 1;i<nPerRingPoly;i++) {
                real3 next = make_real3(xs[baseIdx+i]);
                real3 dx = oldBounds.minImage(next-init);
                deltas[rootIdx + i] = next; 
                diffSum += dx;
            }
            diffSum /= nPerRingPoly;
            real3 unwrappedPos = init + diffSum;
            real3 trace = oldBounds.trace();
            real3 diffFromLo = unwrappedPos - oldBounds.lo;
            real3 imgs = floorf(diffFromLo / trace);
            real3 wrappedPos = unwrappedPos - trace * imgs * oldBounds.periodic;

            // compute the differences from the centroid
            for (int i = rootIdx;i<rootIdx + nPerRingPoly;i++) {
                deltas[i] = oldBounds.minImage(deltas[i]-wrappedPos);
            }

            // now find its scaled position
            // need new boiunds here!!!!
            real3 center = newBounds.lo + newBounds.rectComponents * 0.5f;
            real3 newRel = (wrappedPos - center)*scaleBy;

            // reset the relative positions of the ring polymer based on new centroid
            for ( int i = 0; i<nPerRingPoly; i++) {
                real3 newPos = center+ newRel + deltas[rootIdx+i];
                real3 diffFromLo = newPos - newBounds.lo;
                newPos       -= trace*floorf(diffFromLo / trace)*newBounds.periodic;
                real4 posWhole = xs[baseIdx+i];
                posWhole.x    = newPos.x;
                posWhole.y    = newPos.y;
                posWhole.z    = newPos.z;
                xs[baseIdx+i] = posWhole;
            }
        }
    }
}

template <bool RIGIDBODIES>
__global__ void Mod::scaleSystem_cu(real4 *xs, int nAtoms, real3 lo, real3 rectLen, real3 scaleBy,
                                    int* idToIdxs, uint* notRigidBody) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        if (RIGIDBODIES) {
            // assigned a value of either 1 or 0 in state->findRigidBodies, which is called iff. 
            // there is a barostat present in the simulation
            if (notRigidBody[idx]) {
                int thisIdx = idToIdxs[idx];
                real4 posWhole = xs[thisIdx];
                real3 pos = make_real3(posWhole);
                real3 center = lo + rectLen * 0.5;
                real3 newRel = (pos - center) * scaleBy;
                pos = center + newRel;
                posWhole.x = pos.x;
                posWhole.y = pos.y;
                posWhole.z = pos.z;
                xs[thisIdx] = posWhole;
            }

        } else {

            real4 posWhole = xs[idx];
            real3 pos = make_real3(posWhole);
            real3 center = lo + rectLen * 0.5;
            real3 newRel = (pos - center) * scaleBy;
            pos = center + newRel;
            posWhole.x = pos.x;
            posWhole.y = pos.y;
            posWhole.z = pos.z;
            xs[idx] = posWhole;
        }
    }
}

// RIGIDBODIES is constant throughout a given simulation; so, scale system either by idToIdxs, or just by idx,
// whichever is most convenient; since this doesnt change during a given run, it doesnt matter that we 
// have two conventions by which this can proceed.
template <bool RIGIDBODIES>
__global__ void Mod::scaleSystemGroup_cu(real4 *xs, int nAtoms, real3 lo, real3 rectLen, real3 scaleBy, uint32_t groupTag, real4 *fs, int* idToIdxs, uint* notRigidBody) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        if (RIGIDBODIES) {
            // we need to check that it is not a rigid body, and that it is in the group being scaled
            int newIdx = idToIdxs[idx];
            uint32_t tag = * (uint32_t *) &(fs[newIdx].w);
            if (tag & groupTag) {
                // idx --> id; newIdx --> idx
                if (notRigidBody[idx]) {
            
                    real4 posWhole = xs[newIdx];
                    real3 pos = make_real3(posWhole);
                    real3 center = lo + rectLen * 0.5;
                    real3 newRel = (pos - center) * scaleBy;
                    pos = center + newRel;
                    posWhole.x = pos.x;
                    posWhole.y = pos.y;
                    posWhole.z = pos.z;
                    xs[newIdx] = posWhole;
                }
            } 
        } else {
            // keep it aligned by idx, and check the tag
            uint32_t tag = * (uint32_t *) &(fs[idx].w);
            if (tag & groupTag) {
                real4 posWhole = xs[idx];
                real3 pos = make_real3(posWhole);
                real3 center = lo + rectLen * 0.5;
                real3 newRel = (pos - center) * scaleBy;
                pos = center + newRel;
                posWhole.x = pos.x;
                posWhole.y = pos.y;
                posWhole.z = pos.z;
                xs[idx] = posWhole;


            }
        }
    }
}

void Mod::scaleSystemCentroids(State *state, float3 scaleBy, uint32_t groupTag) {
    auto &gpd = state->gpd;
    BoundsGPU oldBounds = state->boundsGPU;
    state->boundsGPU.scale(scaleBy);
    int nRingPoly       = state->atoms.size() / state->nPerRingPoly;
    int partsPerBlock   = 16*1024 / state->nPerRingPoly  / sizeof(float3);
    if (groupTag==1) {
        scaleCentroids_cu<false><<<NBLOCKVAR(nRingPoly,partsPerBlock),partsPerBlock, partsPerBlock*state->nPerRingPoly*sizeof(float3)>>>(gpd.xs.getDevData(), state->atoms.size(),state->nPerRingPoly, scaleBy, gpd.idToIdxs.d_data.data(),oldBounds,state->boundsGPU);
    } else if (groupTag) {
        scaleCentroidsGroup_cu<false><<<NBLOCK(state->atoms.size()), PERBLOCK>>>(gpd.xs.getDevData(), state->atoms.size(),state->nPerRingPoly, scaleBy, groupTag,gpd.fs.getDevData(), gpd.idToIdxs.d_data.data(),oldBounds,state->boundsGPU);
    }
}

void Mod::scaleSystem(State *state, real3 scaleBy, uint32_t groupTag) {
    auto &gpd = state->gpd;
    state->boundsGPU.scale(scaleBy);
    if (groupTag==1) {
        if (state->rigidBodies) {

            scaleSystem_cu<true><<<NBLOCK(state->atoms.size()), PERBLOCK>>>(gpd.xs.getDevData(), state->atoms.size(), state->boundsGPU.lo, state->boundsGPU.rectComponents, scaleBy, gpd.idToIdxs.d_data.data(),state->rigidBodiesMask.d_data.data());
            for (Fix *f: state->fixes)  {
                f->scaleRigidBodies(scaleBy,groupTag); 
            }
        } else {
            scaleSystem_cu<false><<<NBLOCK(state->atoms.size()), PERBLOCK>>>(gpd.xs.getDevData(), state->atoms.size(), state->boundsGPU.lo, state->boundsGPU.rectComponents, scaleBy, gpd.idToIdxs.d_data.data(),state->rigidBodiesMask.d_data.data());
        }
    } else if (groupTag) {
        if (state->rigidBodies) {
            scaleSystemGroup_cu<true><<<NBLOCK(state->atoms.size()), PERBLOCK>>>(gpd.xs.getDevData(), state->atoms.size(), state->boundsGPU.lo, state->boundsGPU.rectComponents, scaleBy, groupTag, gpd.fs.getDevData(),gpd.idToIdxs.d_data.data(), state->rigidBodiesMask.d_data.data());
            for (Fix *f: state->fixes)  {
                f->scaleRigidBodies(scaleBy,groupTag); 
            }

        } else {
            scaleSystemGroup_cu<false><<<NBLOCK(state->atoms.size()), PERBLOCK>>>(gpd.xs.getDevData(), state->atoms.size(), state->boundsGPU.lo, state->boundsGPU.rectComponents, scaleBy, groupTag, gpd.fs.getDevData(),gpd.idToIdxs.d_data.data(), state->rigidBodiesMask.d_data.data());
        }
    }
}


void Mod::FDotR(State *state) {
    //printf("here!\n");
    auto &gpd = state->gpd;
    int nAtoms = state->atoms.size();
    FDotR_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, gpd.xs.getDevData(), gpd.fs.getDevData(), gpd.virials.getDevData());


}
// CPU versions
/*
void Mod::scaleAtomCoords(SHARED(State) state, string groupHandle, Vector around, Vector scaleBy) {
    return scaleAtomCoords(state.get(), groupHandle, around, scaleBy);
}
void Mod::scaleAtomCoords(State *state, string groupHandle, Vector around, Vector scaleBy) {
	//don't need to makeready for this one		
	//this should work for skewed bounds
	int groupTag = state->groupTagFromHandle(groupHandle);
	for (Atom &a : state->atoms) {
		if (a.groupTag & groupTag) {
			Vector diff = a.pos - around;
			diff *= scaleBy;
			a.pos = around + diff;
		}
	}

}

void Mod::unskewAtoms(vector<Atom> &atoms, Vector xOrig, Vector yOrig) {
    const double lxo = xOrig.len();
    const double lyo = yOrig.len();
    const double lxf = xOrig[0];
    const double lyf = yOrig[1];
    
    const double a = atan2(xOrig[1], xOrig[0]);
    const double b = atan2(yOrig[0], yOrig[1]);

    const double denom = lxo*lyo*cos(a)*cos(b) - lxo*lyo*sin(a)*sin(b);

    const double c1 = lyo*cos(b) / denom;
    const double c2 = -lyo*sin(b) / denom;
    const double c3 = -lxo*sin(a) / denom;
    const double c4 = lxo*cos(a) / denom;

	for (Atom &a : atoms) {
        double xo = a.pos[0];
        double yo = a.pos[1];
        a.pos[0] = lxf * (xo*c1 + yo*c2);
        a.pos[1] = lyf * (xo*c3 + yo*c4);
	}
}


void Mod::skewAtomsFromZero(vector<Atom> &atoms, Vector xFinal, Vector yFinal) {
    const double a = atan2(xFinal[1], xFinal[0]);
    const double b = atan2(yFinal[0], yFinal[1]);
    const double lxf = xFinal.len();
    const double lyf = yFinal.len();
    
    const double lxo = xFinal[0];
    const double lyo = yFinal[1];

    const double c1 = lxf*cos(a);
    const double c2 = lyf*sin(b);

    const double c3 = lxf*sin(a);
    const double c4 = lyf*cos(b);

	for (Atom &a : atoms) {
        const double fx = a.pos[0] / lxo;
        const double fy = a.pos[1] / lyo;
        a.pos[0] = fx * c1 + fy * c2;
        a.pos[1] = fx * c3 + fy * c4;
	}

}

void Mod::skewAtoms(vector<Atom> &atoms, Vector xOrig, Vector xFinal, Vector yOrig, Vector yFinal) {
    const double lxo = xOrig.len();
    const double lyo = yOrig.len();
    const double lxf = xFinal.len();
    const double lyf = yFinal.len();

    const double ao = atan2(xOrig[1], xOrig[0]);
    const double bo = atan2(yOrig[0], yOrig[1]);

    const double af = atan2(xFinal[1], xFinal[0]);
    const double bf = atan2(yFinal[0], yFinal[1]);
    //these coefficients are hairy enough that and functions in Vector just wouldn't be portable.  Going to write it here
    const double denom = (lxo* lyo* cos(ao)* cos(bo) - lxo* lyo*sin(ao)* sin(bo));
//four coefficients for x term
    const double c1 = lxf * lyo * cos(af) * cos(bo);
    const double c2 = lxo * lyf *  cos(ao) * sin(bf);
    const double c3 = lxo * lyf * sin(ao) * sin(bf);
    const double c4 = lxf * lyo *  cos(af)* sin(bo);

    const double c5 = lxo * lyf * cos(ao) * cos(bf);
    const double c6 = lxf * lyo * cos(bo) * sin(af);
    const double c7 = lxo * lyf * cos(bf) * sin(ao); 
    const double c8 = lxf * lyo * sin(af) * sin(bo);

    for (Atom &a : atoms) {
        double xo = a.pos[0];
        double yo = a.pos[1];
        a.pos[0] = (c1*xo + c2*yo - c3*xo - c4*yo) / denom;
        a.pos[1] = (c5*yo + c6*xo - c7*xo - c8*yo) / denom;
    }


//p[0] =(lxf * lyo * xo * cos(af) *  cos(bo) + lxo * lyf * yo * cos(ao) * sin(bf) - lxo * lyf * xo* sin(ao) * sin(bf) - lxf * lyo * yo * cos(af)* sin(bo))/(lxo* lyo* cos(ao)* cos(bo) - lxo* lyo*sin(ao)* sin(bo))

//p[1] = (lxo* lyf* yo* cos(ao)* cos(bf) + lxf* lyo* xo* cos(bo)* sin(af) -   lxo* lyf* xo* cos(bf)* sin(ao) -   lxf* lyo* yo* sin(af)* sin(bo))/(lxo* lyo* cos(ao)* cos(bo) - lxo* lyo*sin(ao)* sin(bo))

    

}


void Mod::skew(SHARED(State) state, Vector skewBy) { //x component is how much to shear y principle vector, y is for x vector
	State *raw = state.get();
	Bounds &b = raw->bounds;
    Vector xOrig = b.sides[0];
    Vector yOrig = b.sides[1];
    b.skew(skewBy);
    Vector xFinal = b.sides[0];
    Vector yFinal = b.sides[1];
	Mod::skewAtoms(raw->atoms, xOrig, xFinal, yOrig, yFinal);
}
*/
