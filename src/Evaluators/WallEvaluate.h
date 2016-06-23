#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"


template <class EVALUATOR, bool COMPUTE_VIRIALS>
__global__ void compute_wall_iso(int nAtoms,float4 *xs, float4 *fs,float3 origin,
								float3 forceDir,float dist,uint groupTag, EVALUATOR eval) {
// need to normaliez forceDir in constructor TODO //



