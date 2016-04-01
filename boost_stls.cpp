#include "boost_stls.h"
#include "Bounds.h"
void export_stls() {
    boost::python::class_<std::map<std::string, int> >("stringInt")
        .def(boost::python::map_indexing_suite<std::map<std::string, int> >())
        ;
    boost::python::class_<std::vector<double> >("vecdouble")
        .def(boost::python::vector_indexing_suite<std::vector<double> >() )
        ;
    boost::python::class_<std::vector<vector<double> > >("vecdouble")
        .def(boost::python::vector_indexing_suite<std::vector< vector<double> > >() )
        ;
    boost::python::class_<std::vector<int> >("vecInt")
        .def(boost::python::vector_indexing_suite<std::vector<int> >() )
        ;

    boost::python::class_<std::vector<int64_t> >("vecLong")
        .def(boost::python::vector_indexing_suite<std::vector<int64_t> >() )
        ;
    boost::python::class_<std::vector<Atom> >("vecAtom")
        .def(boost::python::vector_indexing_suite<std::vector<Atom> >() )
        ;

    boost::python::class_<std::vector<Atom *> >("vecAtomPtr")
        .def(boost::python::vector_indexing_suite<std::vector<Atom *> >() )
        ;
    boost::python::class_<std::vector<Bond> >("vecBond")
        .def(boost::python::vector_indexing_suite<std::vector<Bond> >() )
        ;

    boost::python::class_<std::vector<Neighbor> >("vecNeighbor")
        .def(boost::python::vector_indexing_suite<std::vector<Neighbor> >() )
        ;
    boost::python::class_<std::vector<SHARED(WriteConfig) > >("vecWriteConfig")
        .def(boost::python::vector_indexing_suite<std::vector<SHARED(WriteConfig) > >() )
        ;
    boost::python::class_<std::vector<BondSave > >("vecBondSave")
        .def(boost::python::vector_indexing_suite<std::vector<BondSave > >() )
        ;
    boost::python::class_<std::vector<SHARED(Fix) > >("vecFix")
        .def(boost::python::vector_indexing_suite<std::vector<SHARED(Fix) > >() )
        //	.def("remove", &vectorRemove<Fix>)
        //	.staticmethod("remove")
        ;
    boost::python::class_<std::vector<SHARED(Bounds)> >("vecBounds")
        .def(boost::python::vector_indexing_suite<std::vector<SHARED(Bounds)> >() )
        ;
    boost::python::class_<std::vector<Bounds> >("vecBoundsRaw")
        .def(boost::python::vector_indexing_suite<std::vector<Bounds> >() )
        ;
}
