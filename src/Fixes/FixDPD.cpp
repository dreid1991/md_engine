#include "FixDPD.h"
#include "State.h"

namespace py = boost::python;


// export FixDPD()
void export_FixDPD() {
	py::class_<FixDPD, SHARED(FixDPD), py::bases<Fix> > (
		"FixDPD",
		py::no_init
	);

}

