#include "DataComputerRDF.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
#include "Fix.h"
#include "Group.h"
#include "GPUData.h"
#include "GridGPU.h"

namespace py = boost::python;
using namespace MD_ENGINE;
const std::string computer_type_ = "rdf";


namespace {

template <bool MULTITHREADPERATOM>
__global__ void compute_rdf(int nAtoms,
                            int nPerRingPoly,
                            real * __restrict__ bins,
                            double binWidth,
                            double rCut,
                            int s1_type,
                            int s2_type,
                            real onetwoStr, 
                            real onethreeStr, 
                            real onefourStr, 
                            real4 * __restrict__ xs,
                            const uint16_t *__restrict__ neighborCounts,
                            const uint *__restrict__ neighborlist,      
                            const uint32_t * __restrict__ cumulSumMaxPerBlock,
                            int warpSize,                 
                            BoundsGPU bounds,
                            int nThreadPerBlock,
                            int nThreadPerAtom) 
{
    real multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    int idx = GETIDX();
    if (idx < nAtoms * nThreadPerAtom) {
        int atomIdx;
        if (MULTITHREADPERATOM) {
            atomIdx = idx/nThreadPerAtom;
        } else {
            atomIdx = idx;
        }
        
        int ringPolyIdx = atomIdx / nPerRingPoly;	// which ring polymer
        int beadIdx     = atomIdx % nPerRingPoly;	// which time slice

        int baseIdx;
        if (MULTITHREADPERATOM) {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx, nThreadPerAtom);
        } else {
            baseIdx = baseNeighlistIdxFromRPIndex(cumulSumMaxPerBlock, warpSize, ringPolyIdx);
        }
        
        real4 posWhole = xs[atomIdx];
        int type = __real_as_int(posWhole.w);
        real3 pos = make_real3(posWhole);
 
        if (type == s1_type) {
            int myIdxInTeam;
            if (MULTITHREADPERATOM) {
                myIdxInTeam = threadIdx.x % nThreadPerAtom; // 0..... nThreadPerAtom - 1
            } else {
                myIdxInTeam = 0;
            }
            int numNeigh = neighborCounts[ringPolyIdx];
            for (int nthNeigh=myIdxInTeam; nthNeigh<numNeigh; nthNeigh+=nThreadPerAtom) {
                int nlistIdx;
                if (MULTITHREADPERATOM) {
                    nlistIdx = baseIdx + myIdxInTeam + warpSize * (nthNeigh/nThreadPerAtom);
                } else {
                    nlistIdx = baseIdx + warpSize * nthNeigh;
                }
                
                uint otherIdxRaw = neighborlist[nlistIdx];
                //The leftmost two bits in the neighbor entry say if it is a 1-2, 1-3, or 1-4 neighbor, or none of these
                uint neighDist = otherIdxRaw >> 30;
                real multiplier = multipliers[neighDist];
                
                // Extract corresponding index for pair interaction (at same time slice)
                uint otherRPIdx = otherIdxRaw & EXCL_MASK;
                uint otherIdx   = nPerRingPoly*otherRPIdx + beadIdx;  // atom = P*ring_polymer + k, k = 0,...,P-1
                real4 otherPosWhole = xs[otherIdx];

                //type is stored in w component of position
                int otherType = __real_as_int(otherPosWhole.w);
                real3 otherPos = make_real3(otherPosWhole);

                // if otherType matches and we are not excluding, then do the computation
                if (otherType == s2_type && multiplier) {
                    //based on the two atoms types, which index in each of the square matrices will I need to load from?
                    real3 dr  = bounds.minImage(pos - otherPos);
                    real dist = length(dr);
 
                    // if dist < rCut, we're ok; otherwise, it would map outside of the array.
                    if (dist < rCut) {
                        // translate dist to a bin index; floor because 0-based indexing
                        int binIndex = (int) (floor(dist / binWidth));

                        atomicAdd(&bins[binIndex],1.0);
                    } // dist < rCut
                        //load that pair's parameters into a linear array to be send to the force evaluator
                } // if otherType == s2_type, then do the binning computation.
                  //  note that we must do an atomicAdd.
            } // nlist iteration
        } // if type == s1_type
    } // idx < nAtoms*nThreadPerAtom
}

// very simple..
__global__ void addToCumulativeHistogram(int arraySize,
                                         real * __restrict__ histogram,
                                         real * __restrict__ cumulativeHistogram) {

    int idx = GETIDX();
    if (idx < arraySize) {
        real value = histogram[idx];
        real cumulative = cumulativeHistogram[idx];
        cumulative += value;
        cumulativeHistogram[idx] = cumulative;
    }
}

// normalization!
__global__ void normalizeCumulativeHistogram(int arraySize,
                                             double binWidth,
                                             double numberDensity,
                                             int nSamples, 
                                             int nFrames, 
                                             real * __restrict__ cumulativeHistogram) {

    int idx = GETIDX();

    if (idx < arraySize) {
        double dr0 = ( (double) idx) * binWidth;
        double dr1 = ( (double) idx+1.0) * binWidth;
        double volume = (4.0 / 3.0) * M_PI * ( (dr1*dr1*dr1) - (dr0*dr0*dr0));
        double binValue = (double) cumulativeHistogram[idx];
        double denominator = (double) (numberDensity * volume * nSamples * nFrames);
        double normalizedBinValue = binValue / denominator;
        cumulativeHistogram[idx] = (real) normalizedBinValue;
    }
}

} // namespace {}



DataComputerRDF::DataComputerRDF(State *state_, std::string computeMode_,std::string species1_, std::string species2_, double binWidth_) : DataComputer(state_, computeMode_, false,computer_type_), 
                               binWidth(binWidth_), species1(species1_), species2(species2_) 
{

}


void DataComputerRDF::prepareForRun() {
    DataComputer::prepareForRun();
    // we do need to allocate a few arrays here
    volume = state->bounds.volume();
    nTurns = 0;
    s1_count = 0;
    s2_count = 0;
    s1_type  = state->atomParams.typeFromHandle(species1);
    s2_type  = state->atomParams.typeFromHandle(species2);

    for (auto atom : state->atoms) {
        if (atom.type == s1_type) {
            s1_count++;
        }
    }

    for (auto atom : state->atoms) {
        if (atom.type == s2_type) {
            s2_count++;
        }
    }

    if (s1_count == 0 || s2_count == 0) {
        // error: can't compute RDF of species that isn't present
        mdError("Error in DataComputerRDF: cannot compute RDF of species that is not present.");
    }

    std::cout << "Found " << s1_count << " atoms of s1 and " << s2_count << " atoms of s2 in a volume of " << volume << std::endl;

    numberDensity = (double) s2_count / volume;
    // ok, we have our samples, we have our volume (we do not permit RDF within volume slices - only 
    // total simulation volume)

    nBins = (int) (std::ceil(state->rCut / (double) binWidth));
    
    std::cout << "allocating nBins : " << nBins << std::endl;
    // this is where we actually store the counts at a given turn
    histogram = GPUArrayDeviceGlobal<real>(nBins);
    histogram.memset(0); // memset to initialize to zero..
    
    // this is the final data structure that we put in 'vals' array at postRun.
    cumulativeHistogram = GPUArrayDeviceGlobal<real>(nBins);
    cumulativeHistogram.memset(0);
    cudaDeviceSynchronize();

}


void DataComputerRDF::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    
    int nAtoms = state->atoms.size();
    int NTPB = state->nThreadPerBlock;
    int NTPA = state->nThreadPerAtom;
    // we're using the neighborlist, so use same kernel config as in the potentials / force kernels
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    
    histogram.memset(0); // clear the histogram
    cudaDeviceSynchronize();
    real onetwoStr = 0.0; // don't count immediate near neighbors in the histogram
    real onethreeStr = 0.0; // don't count immediate near neighbors in the histogram
    real onefourStr = 0.0;  // don't count immediate near neighbors in the histogram
    // we pass in s1 type, s2 type, our atoms positions, bounds, histogram
    bool multiThreadPerAtom = NTPA > 1 ? true : false;
    int nPerRingPoly = state->nPerRingPoly;

    if (multiThreadPerAtom) {
        compute_rdf<true><<<NBLOCKTEAM(nAtoms, NTPB, NTPA),NTPB>>>(nAtoms,
                                                             nPerRingPoly,
                                                             histogram.data(),
                                                             binWidth,
                                                             state->rCut,
                                                             s1_type,
                                                             s2_type,
                                                             onetwoStr, 
                                                             onethreeStr, 
                                                             onefourStr, 
                                                             gpd.xs(activeIdx),
                                                             grid.perAtomArray.d_data.data(),
                                                             grid.neighborlist.data(),
                                                             grid.perBlockArray.d_data.data(),
                                                             state->devManager.prop.warpSize,
                                                             state->boundsGPU,
                                                             NTPB,
                                                             NTPA);
    } else {

        compute_rdf<false><<<NBLOCKTEAM(nAtoms, NTPB, NTPA),NTPB>>>(nAtoms,
                                                             nPerRingPoly,
                                                             histogram.data(),
                                                             binWidth,
                                                             state->rCut,
                                                             s1_type,
                                                             s2_type,
                                                             onetwoStr, 
                                                             onethreeStr, 
                                                             onefourStr, 
                                                             gpd.xs(activeIdx),
                                                             grid.perAtomArray.d_data.data(),
                                                             grid.neighborlist.data(),
                                                             grid.perBlockArray.d_data.data(),
                                                             state->devManager.prop.warpSize,
                                                             state->boundsGPU,
                                                             NTPB,
                                                             NTPA);
    }

    // nothing too fancy -- add to cumulative counter
    addToCumulativeHistogram<<<NBLOCK(histogram.size()),PERBLOCK>>>(histogram.size(),
                                                                    histogram.data(),
                                                                    cumulativeHistogram.data());

    // increment our frames counter
    nTurns++;
}



void DataComputerRDF::postRun(boost::python::list &vals) {
    // post-process cumulativeHistogram s.t. the bins are normalized
    normalizeCumulativeHistogram<<<NBLOCK(cumulativeHistogram.size()), PERBLOCK>>> (
                                                cumulativeHistogram.size(),
                                                binWidth,
                                                numberDensity,
                                                s1_count,
                                                nTurns,
                                                cumulativeHistogram.data());

    
    cudaDeviceSynchronize();

    //malloc 
    real *host_cumulativeHistogram = (real *) malloc(cumulativeHistogram.size()*sizeof(real));
    cumulativeHistogram.get(host_cumulativeHistogram);
    
    cudaDeviceSynchronize();

    vals = boost::python::list {};
    
    for (size_t i  = 0; i < cumulativeHistogram.size(); i++ ) {
        double currentVal = (double) host_cumulativeHistogram[i];
        vals.append(currentVal);
    }

    // free
    free(host_cumulativeHistogram);

}

