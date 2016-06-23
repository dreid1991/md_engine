#include "FixWall.h"
#include "State.h"

namespace py = boost::python;


// have this here for completeness, and if it should be used in the future;
// what would it be used for in this class though?
bool FixWall::prepareForRun() {

	return true;

};


// export FixWall()
void export_FixWall() {
	py::class_<FixWall, SHARED(FixWall), py::bases<Fix> > (
		"FixWall",
		boost::python::no_init
	);

}

