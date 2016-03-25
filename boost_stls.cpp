#include "boost_stls.h"

void export_stls() {
    class_<std::map<std::string, int> >("stringInt")
        .def(map_indexing_suite<std::map<std::string, int> >())
        ;
    class_<std::vector<double> >("vecdouble")
        .def(vector_indexing_suite<std::vector<double> >() )
        ;

    class_<std::vector<vector<double> > >("vecdouble")
        .def(vector_indexing_suite<std::vector< vector<double> > >() )
        ;
    class_<std::vector<int> >("vecInt")
        .def(vector_indexing_suite<std::vector<int> >() )
        ;

    class_<std::vector<int64_t> >("vecLong")
        .def(vector_indexing_suite<std::vector<int64_t> >() )
        ;
    class_<std::vector<Atom> >("vecAtom")
        .def(vector_indexing_suite<std::vector<Atom> >() )
        ;

    class_<std::vector<Atom *> >("vecAtomPtr")
        .def(vector_indexing_suite<std::vector<Atom *> >() )
        ;
    class_<std::vector<Bond> >("vecBond")
        .def(vector_indexing_suite<std::vector<Bond> >() )
        ;

    class_<std::vector<Neighbor> >("vecNeighbor")
        .def(vector_indexing_suite<std::vector<Neighbor> >() )
        ;
    class_<std::vector<SHARED(WriteConfig) > >("vecWriteConfig")
        .def(vector_indexing_suite<std::vector<SHARED(WriteConfig) > >() )
        ;
    class_<std::vector<BondSave > >("vecBondSave")
        .def(vector_indexing_suite<std::vector<BondSave > >() )
        ;
    class_<std::vector<SHARED(Fix) > >("vecFix")
        .def(vector_indexing_suite<std::vector<SHARED(Fix) > >() )
        //	.def("remove", &vectorRemove<Fix>)
        //	.staticmethod("remove")
        ;
}
