#include <iostream>
#include "Python.h"
#include "boost_for_export.h"
namespace py = boost::python;
#include "Vector.h"

std::ostream &operator<<(std::ostream &os, const Vector &v) {
    os << v.asStr();
    return os;
}


std::ostream &operator<<(std::ostream &os, const float4 &v) {
    os << "x: " << v.x << " y: " << v.y<< " z: " << v.z<< " w: " << v.w;
    return os;
}

void export_Vector() {
    py::class_<Vector>("Vector", py::init<double, double, double>())
        .def("__getitem__", &Vector::get)
        .def("__setitem__", &Vector::set)
        .def("__str__", &Vector::asStr)
        .def("dist", &Vector::dist<double>)
        .def("len", &Vector::len)
        .def("lenSqr", &Vector::lenSqr)
        .def("dot", &Vector::dot<double>)
        .def("normalized", &Vector::normalized)
        .def("copy", &Vector::copy)

        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self * double())
        .def(py::self / double())
        ;

}
void export_VectorInt() {
    py::class_<VectorInt>("VectorInt", py::init<int, int, int>())
        .def("__getitem__", &VectorInt::get)
        .def("__setitem__", &VectorInt::set)
        .def("__str__", &VectorInt::asStr)
        .def("copy", &VectorInt::copy)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        ;

}
