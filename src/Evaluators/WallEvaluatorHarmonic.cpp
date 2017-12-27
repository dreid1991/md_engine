#include "boost_for_export.h"
#include "WallEvaluatorHarmonic.h"

// expose to python interface; this is /only/ to be used by pytest, or for static evaluation of forces
void export_EvaluatorWallHarmonic() {
    boost::python::class_<EvaluatorWallHarmonic, SHARED(EvaluatorWallHarmonic), boost::noncopyable> (
        "EvaluatorWallHarmonic",
		boost::python::init<real, real> (
			boost::python::args("k", "r0")
		)
    )
    .def("force", &EvaluatorWallHarmonic::force,
          (boost::python::arg("magProj"),
           boost::python::arg("forceDir")
          )
        )
    .def_readwrite("k", &EvaluatorWallHarmonic::k)
    .def_readwrite("r0",&EvaluatorWallHarmonic::r0)
    ;

}
