#include "Python.h"
#include <functional>
#include <map>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/args.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/operators.hpp>

using namespace boost::python;
#include <tuple>
using namespace std;
#include "State.h"
#include "Atom.h"
#include "AtomGrid.h"
#include "Vector.h" 
#include "InitializeAtoms.h"
#include "Bounds.h"
#include "includeFixes.h"
#include "IntegraterVerlet.h"
#include "IntegraterRelax.h"
#include "boost_stls.h"
#include "PythonOperation.h"
//#include "DataManager.h"
//#include "DataSet.h"
#include "ReadConfig.h"
//#include "DataTools.h"
BOOST_PYTHON_MODULE(Sim) {
    export_stls();	

    export_Vector();
    export_VectorInt();	
    export_Atom();
    export_Neighbor();
    export_Bounds();
    export_Integrater();
    export_IntegraterVerlet();
    export_IntegraterRelax();
    export_Fix();
    export_FixBondHarmonic();
    export_FixWallHarmonic();
    export_FixSpringStatic();
    export_FixLJCut(); //make there be a pair base class in boost!
    export_Fix2d();
    
    export_FixCharge();
    export_FixChargePairDSF();
    
    export_FixNVTRescale();

    export_AtomGrid();
    export_AtomParams();
    export_DataManager();
    export_ReadConfig();
    export_PythonOperation();

    export_WriteConfig();
    export_InitializeAtoms();
        

    export_State(); 	
    export_DeviceManager();

    /*
	class_<ModPythonWrap> ("Mod")
		.def("deleteBonds", &Mod::deleteBonds, (python::arg("groupHandle")))
		.staticmethod("deleteBonds")
		.def("bondWithCutoff", &Mod::bondWithCutoff, (python::arg("groupHandle"), python::arg("sigMultCutoff"), python::arg("k")) )
		.staticmethod("bondWithCutoff")
		.def("scaleAtomCoords", &Mod::scaleAtomCoords, (python::arg("groupHandle"), python::arg("around"), python::arg("scaleBy")) )
		.staticmethod("scaleAtomCoords")
		.def("computeNumBonds", &Mod::computeNumBonds, (python::arg("groupHandle")) )
		.staticmethod("computeNumBonds")
		.def("computeBondStresses", &Mod::computeBondStresses)
		.staticmethod("computeBondStresses")
		.def("deleteAtomsWithSingleSideBonds", &Mod::deleteAtomsWithSingleSideBonds, (python::arg("groupHandle")) )
		.staticmethod("deleteAtomsWithSingleSideBonds")
		.def("deleteAtomsWithBondThreshold", &Mod::deleteAtomsWithBondThreshold, (python::arg("groupHandle"), python::arg("thresh"), python::arg("polarity")) )
		.staticmethod("deleteAtomsWithBondThreshold")
		.def("computeZ", &Mod::computeZ, (python::arg("groupHandle")) )
		.staticmethod("computeZ")
		.def("setZValue", &Mod::setZValue, (python::arg("neighThresh"), python::arg("target"), python::arg("tolerance")=.05, python::arg("kBond")=1, python::arg("display")=false))
		.staticmethod("setZValue")
        .def("skew", &Mod::skew, (python::arg("skewBy")))
        .staticmethod("skew")
		;
        */
/*
	class_<DataToolsPythonWrap> ("DataTools")
		.def("logHistogram", &DataTools::logHistogram, (python::arg("xs"), python::arg("binXs")))
		.staticmethod("logHistogram")
		;
        */
}
