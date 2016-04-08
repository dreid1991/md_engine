#pragma once
#ifndef BOUNDS_GPU
#define BOUNDS_GPU

#include "cutils_math.h"
#include "globalDefs.h"
#include "Vector.h"

/*! \brief store Boundaries on the GPU
 *
 * This class stores boundaries of the simulation box on the GPU. The class is
 * typically constructed by calling Bounds::makeGPU()
 *
 * Bounds on the GPU are defined by a point of origin and three vectors
 * defining the x-, y-, and z- directions of the box. Furthermore, the box can
 * be periodic or fixed in each direction.
 */
class BoundsGPU {
public:
    /*! \brief Constructor
     *
     * \param lo_ Lower values of the boundaries (Point of origin)
     * \param sides_ 3-dimensional array storing the x-, y-, and z-vectors
     * \param periodic_ Stores whether the box is periodic in x-, y-, and
     *                  z-direction
     */
    BoundsGPU(float3 lo_, float3 *sides_, float3 periodic_) {
        lo = lo_;
        for (int i=0; i<3; i++) {
            sides[i] = sides_[i];
            periodic = periodic_;
        }
        rectLen = make_float3(sides[0].x, sides[1].y, sides[2].z);
        invRectLen = (float) 1 / rectLen;
    }

    /*! \brief Default constructor */
    BoundsGPU() {};

    float3 sides[3]; //!< 3 vectors defining the x-, y-, and z- direction
    float3 rectLen; //!< Length of box in standard coordinates
    float3 invRectLen; //!< Inverse of the box expansion in standard
                       //!< coordinates
    float3 lo; //!< Point of origin
    float3 periodic; //!< Stores whether box is periodic in x-, y-, and
                     //!< z-direction

    /*! \brief Return an unskewed copy of this box
     *
     * \return Unskewed copy of this box.
     */
    BoundsGPU unskewed() {
        float3 sidesNew[3];
        memset(sidesNew, 0, 3*sizeof(float3));
        sidesNew[0].x = sides[0].x;
        sidesNew[1].y = sides[1].y;
        sidesNew[2].z = sides[2].z;
        return BoundsGPU(lo, sidesNew, periodic);
    }

    /*! \brief Return trace of this box
     *
     * \return Trace for the box
     *
     * \todo Trace is identical to rectLen, isn't it?
     */
    GPUMEMBER float3 trace() {
        return make_float3(sides[0].x, sides[1].y, sides[2].z);
    }

    /*! \brief Return vector wrapped into the main simulation box
     *
     * \param v %Vector to be wrapped
     * \return Copy of the vector, wrapped into main simulation box
     */
    GPUMEMBER float3 minImage(float3 v) {
        int img = rintf(v.x * invRectLen.x);
        v -= sides[0] * img * periodic.x;
        img = rintf(v.y * invRectLen.y);
        v -= sides[1] * img * periodic.y;
        img = rintf(v.z * invRectLen.z);
        v -= sides[2] * img * periodic.z;
        return v;
    }

    /*! \brief Test if point is within simulation box
     *
     * \param v Point to test
     * \return True if inside simulation box
     */
    GPUMEMBER bool inBounds(float3 v) {
        float3 diff = v - lo;
        return diff.x < sides[0].x and diff.y < sides[1].y and diff.z < sides[2].z and diff.x >= 0 and diff.y >= 0 and diff.z >= 0;
    }
};

#endif
