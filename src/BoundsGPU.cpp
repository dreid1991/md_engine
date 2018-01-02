#include "boost_for_export.h"
#include "BoundsGPU.h"

// convert to values used internally
BoundsGPU::BoundsGPU(Vector lo_, Vector rectComponents_, Vector periodic_) {

        //consider calcing invrectcomponents using doubles
        lo = lo_.asreal3();
        rectComponents = rectComponents_.asreal3();
        invRectComponents = 1.0 / rectComponents;
        periodic = periodic_.asreal3();
        periodicD = make_double3(periodic.x, periodic.y, periodic.z);
        rectComponentsD = make_double3(rectComponents.x, rectComponents.y,rectComponents.z);
        invRectComponentsD = make_double3(invRectComponents.x, invRectComponents.y, invRectComponents.z);

}

__host__ Vector BoundsGPU::minImagePy(Vector v) {

    real3 dr = v.asreal3();
    real3 result = minImage(dr);
    Vector resultToReturn = Vector(result.x, result.y, result.z);
    return resultToReturn;

}


// exposing methods to python
void export_BoundsGPU() {
    boost::python::class_<BoundsGPU, SHARED(BoundsGPU), boost::noncopyable> (
        "BoundsGPU",
		boost::python::init<Vector, Vector, Vector> (
            boost::python::args("lo","rectComponents","periodic")
		)
    )
    .def("minImage", &BoundsGPU::minImagePy,
          (boost::python::arg("dr")
          )
        )
    ;

}



