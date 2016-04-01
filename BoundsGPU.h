#ifndef BOUNDS_GPU
#define BOUNDS_GPU
#include "cutils_math.h"
#include "Vector.h"
class BoundsGPU {
    public:
        BoundsGPU(float3 lo_, float3 *sides_, float3 periodic_) {
            lo = lo_;
            for (int i=0; i<3; i++) {
                sides[i] = sides_[i];
                periodic = periodic_;

            }
            rectLen = make_float3(sides[0].x, sides[1].y, sides[2].z);
            invRectLen = (float) 1 / rectLen;

        }
        BoundsGPU() {};
        float3 sides[3];
        float3 rectLen;
        float3 invRectLen;
        float3 lo;
        float3 periodic;
        BoundsGPU unskewed() {
            float3 sidesNew[3];
            memset(sidesNew, 0, 3*sizeof(float3));
            sidesNew[0].x = sides[0].x;
            sidesNew[1].y = sides[1].y;
            sidesNew[2].z = sides[2].z;
            return BoundsGPU(lo, sidesNew, periodic);
        }
        GPUMEMBER float3 trace() {
            return make_float3(sides[0].x, sides[1].y, sides[2].z);

        }
        /*
        __device__ float3 minImageUnskewed(float3 v) {
            float3 imgs = floorf(float3((v - make_float3(lo)) * make_float3(invRectLen));
            v -= periodic * imgs;
            return v;
        }
        */
        GPUMEMBER float3 minImage(float3 v) {
        //float4 imgs = floorf((pos - bounds.lo) / trace); //are unskewed at this point
            
            int img = rintf(v.x * invRectLen.x);
            v -= sides[0] * img * periodic.x;

            img = rintf(v.y * invRectLen.y);
            v -= sides[1] * img * periodic.y;

            img = rintf(v.z * invRectLen.z);
            v -= sides[2] * img * periodic.z;
            return v;
        }
        GPUMEMBER bool inBounds(float3 v) {
            float3 diff = v - lo;
            return diff.x < sides[0].x and diff.y < sides[1].y and diff.z < sides[2].z;
        }
        
};
#endif
