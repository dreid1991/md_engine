#include "Mod.h"
#include "State.h"

#include "Bond.h"
#include "Angle.h"
#include "Vector.h"
#include "helpers.h"
#include "Fix.h"
using namespace std;


	
__global__ void Mod::unskewAtoms(float4 *xs, int nAtoms, float3 xOrig, float3 yOrig, float3 lo) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        float lxo = length(xOrig);
        float lyo = length(yOrig);
        float lxf = xOrig.x;
        float lyf = yOrig.y;
       
        float a = atan2(xOrig.y, xOrig.x);
        float b = atan2(yOrig.x, yOrig.y);

        float invDenom = 1.0f / (lxo*lyo*cos(a)*cos(b) - lxo*lyo*sin(a)*sin(b));

        float c1 = lyo*cos(b) * invDenom;
        float c2 = -lyo*sin(b) * invDenom;
        float c3 = -lxo*sin(a) * invDenom;
        float c4 = lxo*cos(a) * invDenom;
        


        float4 pos = xs[idx];
        float xo = pos.x - lo.x;
        float yo = pos.y - lo.y;
        pos.x = lxf * (xo*c1 + yo*c2) + lo.x;
        pos.y = lyf * (xo*c3 + yo*c4) + lo.y;
        xs[idx] = pos;
    }
}


__global__ void Mod::skewAtomsFromZero(float4 *xs, int nAtoms, float3 xFinal, float3 yFinal, float3 lo) {
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


        float4 pos = xs[idx];

        float xo = pos.x - lo.x;
        float yo = pos.y - lo.y;

        const double fx = xo / lxo;
        const double fy = yo / lyo;
        pos.x = fx * c1 + fy * c2 + lo.x;
        pos.y = fx * c3 + fy * c4 + lo.y;

        xs[idx] = pos;
    }
}

__global__ void FDotR_cu(int nAtoms, float4 *xs, float4 *fs, Virial *virials) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float3 x = make_float3(xs[idx]);
        //f only has pair-wise forces right now
        float3 f = make_float3(fs[idx]);
        //virial is zero at this point.  Only time f dot r in valid
        Virial v(0, 0, 0, 0, 0, 0);
        computeVirial(v, f, x);
        virials[idx] = v;

    }
}

template <bool RIGIDBODIES>
__global__ void Mod::scaleCentroids_cu(float4 *xs, int nAtoms,int nPerRingPoly, float3 scaleBy,
                                    int* idToIdxs, BoundsGPU oldBounds,BoundsGPU newBounds) {

    int idx     = GETIDX();
    int rootIdx = threadIdx.x * nPerRingPoly;
    extern __shared__ float3 deltas[];

    int nRingPoly = nAtoms / nPerRingPoly;
    if (idx < nRingPoly) {
        // determine centroid position for this ring polymer
        int baseIdx = idx*nPerRingPoly;
        float3 init     = make_float3(xs[baseIdx]);
        float3 diffSum  = make_float3(0,0,0);
        deltas[rootIdx] = init; 
        for (int i = 1;i<nPerRingPoly;i++) {
            float3 next = make_float3(xs[baseIdx+i]);
            float3 dx = oldBounds.minImage(next-init);
            deltas[rootIdx + i] = next; 
            diffSum += dx;
        }
        diffSum /= nPerRingPoly;
        float3 unwrappedPos = init + diffSum;
        float3 trace = oldBounds.trace();
        float3 diffFromLo = unwrappedPos - oldBounds.lo;
        float3 imgs = floorf(diffFromLo / trace);
        float3 wrappedPos = unwrappedPos - trace * imgs * oldBounds.periodic;

        // compute the differences from the centroid
        for (int i = rootIdx;i<rootIdx + nPerRingPoly;i++) {
            deltas[i] = oldBounds.minImage(deltas[i]-wrappedPos);
        }

        // now find its scaled position
        // need new boiunds here!!!!
        float3 center = newBounds.lo + newBounds.rectComponents * 0.5f;
        float3 newRel = (wrappedPos - center)*scaleBy;

        // reset the relative positions of the ring polymer based on new centroid
        for ( int i = 0; i<nPerRingPoly; i++) {
            float3 newPos = wrappedPos + deltas[rootIdx+i];
            float3 diffFromLo = newPos - newBounds.lo;
            newPos       -= trace*floorf(diffFromLo / trace)*newBounds.periodic;
            float4 posWhole = xs[baseIdx+i];
            posWhole.x    = newPos.x;
            posWhole.y    = newPos.y;
            posWhole.z    = newPos.z;
            xs[baseIdx+i] = posWhole;
        }
    }
}

template <bool RIGIDBODIES>
__global__ void Mod::scaleCentroidsGroup_cu(float4 *xs, int nAtoms,int nPerRingPoly, float3 scaleBy,uint32_t groupTag,
                                    float4 *fs, int* idToIdxs, BoundsGPU oldBounds,BoundsGPU newBounds) {

    int idx     = GETIDX();
    int rootIdx = threadIdx.x * nPerRingPoly;
    extern __shared__ float3 deltas[];

    int nRingPoly = nAtoms / nPerRingPoly;
    if (idx < nRingPoly) {
        // determine centroid position for this ring polymer
        int baseIdx = idx*nPerRingPoly;
        uint32_t tag = * (uint32_t *) &(fs[baseIdx].w);
        if (tag & groupTag) {
            float3 init     = make_float3(xs[baseIdx]);
            float3 diffSum  = make_float3(0,0,0);
            deltas[rootIdx] = init; 
            for (int i = 1;i<nPerRingPoly;i++) {
                float3 next = make_float3(xs[baseIdx+i]);
                float3 dx = oldBounds.minImage(next-init);
                deltas[rootIdx + i] = next; 
                diffSum += dx;
            }
            diffSum /= nPerRingPoly;
            float3 unwrappedPos = init + diffSum;
            float3 trace = oldBounds.trace();
            float3 diffFromLo = unwrappedPos - oldBounds.lo;
            float3 imgs = floorf(diffFromLo / trace);
            float3 wrappedPos = unwrappedPos - trace * imgs * oldBounds.periodic;

            // compute the differences from the centroid
            for (int i = rootIdx;i<rootIdx + nPerRingPoly;i++) {
                deltas[i] = oldBounds.minImage(deltas[i]-wrappedPos);
            }

            // now find its scaled position
            // need new boiunds here!!!!
            float3 center = newBounds.lo + newBounds.rectComponents * 0.5f;
            float3 newRel = (wrappedPos - center)*scaleBy;

            // reset the relative positions of the ring polymer based on new centroid
            for ( int i = 0; i<nPerRingPoly; i++) {
                float3 newPos = wrappedPos + deltas[rootIdx+i];
                float3 diffFromLo = newPos - newBounds.lo;
                newPos       -= trace*floorf(diffFromLo / trace)*newBounds.periodic;
                float4 posWhole = xs[baseIdx+i];
                posWhole.x    = newPos.x;
                posWhole.y    = newPos.y;
                posWhole.z    = newPos.z;
                xs[baseIdx+i] = posWhole;
            }
        }
    }
}

template <bool RIGIDBODIES>
__global__ void Mod::scaleSystem_cu(float4 *xs, int nAtoms, float3 lo, float3 rectLen, float3 scaleBy,
                                    int* idToIdxs, int* notRigidBody) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        if (RIGIDBODIES) {
            if (notRigidBody[idx]) {
                int thisIdx = idToIdxs[idx];
                float4 posWhole = xs[thisIdx];
                float3 pos = make_float3(posWhole);
                float3 center = lo + rectLen * 0.5f;
                float3 newRel = (pos - center) * scaleBy;
                pos = center + newRel;
                posWhole.x = pos.x;
                posWhole.y = pos.y;
                posWhole.z = pos.z;
                xs[thisIdx] = posWhole;
            }

        } else {

            float4 posWhole = xs[idx];
            float3 pos = make_float3(posWhole);
            float3 center = lo + rectLen * 0.5f;
            float3 newRel = (pos - center) * scaleBy;
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
__global__ void Mod::scaleSystemGroup_cu(float4 *xs, int nAtoms, float3 lo, float3 rectLen, float3 scaleBy, uint32_t groupTag, float4 *fs, int* idToIdxs, int* notRigidBody) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        if (RIGIDBODIES) {
            // we need to check that it is not a rigid body, and that it is in the group being scaled
            int newIdx = idToIdxs[idx];
            uint32_t tag = * (uint32_t *) &(fs[newIdx].w);
            if (tag & groupTag) {
                // idx --> id; newIdx --> idx
                if (notRigidBody[idx]) {
            
                    float4 posWhole = xs[newIdx];
                    float3 pos = make_float3(posWhole);
                    float3 center = lo + rectLen * 0.5f;
                    float3 newRel = (pos - center) * scaleBy;
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
                float4 posWhole = xs[idx];
                float3 pos = make_float3(posWhole);
                float3 center = lo + rectLen * 0.5f;
                float3 newRel = (pos - center) * scaleBy;
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

void Mod::scaleSystem(State *state, float3 scaleBy, uint32_t groupTag) {
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
