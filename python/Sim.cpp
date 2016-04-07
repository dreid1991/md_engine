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
#include "IntegraterLangevin.h"
#include "boost_stls.h"
#include "PythonOperation.h"
//#include "DataManager.h"
#include "DataSet.h"
#include "DataSetTemperature.h"
#include "DataSetEnergy.h"
#include "DataSetBounds.h"
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
    export_IntegraterLangevin();
    export_Fix();
    export_FixBondHarmonic();
    export_FixAngleHarmonic();
    export_FixImproperHarmonic();
    export_FixDihedralOPLS();
    export_FixWallHarmonic();
    export_FixSpringStatic();
    export_Fix2d();

    export_FixPair();
    export_FixLJCut(); //make there be a pair base class in boost!
    
    export_FixCharge();
    export_FixChargePairDSF();
    export_FixChargeEwald();
    
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
    export_DataSet();
    export_DataSetTemperature();
    export_DataSetEnergy();
    export_DataSetBounds();

}
