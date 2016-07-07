#pragma once
#ifndef BOUNDS_GPU
#define BOUNDS_GPU

#include "cutils_math.h"
#include "globalDefs.h"

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
    BoundsGPU(float3 lo_, float3 rectComponents_, float3 periodic_) {
        lo = lo_;
        rectComponents = rectComponents_;
        invRectLen = 1.0f / rectComponents;
        periodic = periodic_;
    }

    /*! \brief Default constructor */
    BoundsGPU() {};

    float3 rectComponents; //!< 3 sides - xx, yy, zz
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
        /*
        float3 sidesNew[3];
        memset(sidesNew, 0, 3*sizeof(float3));
        sidesNew[0].x = sides[0].x;
        sidesNew[1].y = sides[1].y;
        sidesNew[2].z = sides[2].z;
        return BoundsGPU(lo, sidesNew, periodic);
        */
        return *this;
    }

    /*! \brief Return trace of this box
     *
     * \return Trace for the box
     *
     * Will be updated to handle box shearing
     */
    GPUMEMBER float3 trace() {
        return rectComponents;
        //return make_float3(sides[0].x, sides[1].y, sides[2].z);
    }

    /*! \brief Return vector wrapped into the main simulation box
     *
     * \param v %Vector to be wrapped
     * \return Copy of the vector, wrapped into main simulation box
     */
    GPUMEMBER float3 minImage(float3 v) {
        float3 img = make_float3(rintf(v.x * invRectLen.x), rintf(v.y * invRectLen.y), rintf(v.z * invRectLen.z));
        v -= rectComponents * img * periodic;
        return v;
    }
    GPUMEMBER float volume() {
        return rectComponents.x * rectComponents.y * rectComponents.z;
    }

    /*! \brief Test if point is within simulation box
     *
     * \param v Point to test
     * \return True if inside simulation box
     */
    GPUMEMBER bool inBounds(float3 v) {
        float3 diff = v - lo;
        return diff.x < rectComponents.x and diff.y < rectComponents.y and diff.z < rectComponents.z and diff.x >= 0 and diff.y >= 0 and diff.z >= 0;
    }
    bool isSkewed() {
        //dummy function until skewing added
        return false;
    }
};

#endif
