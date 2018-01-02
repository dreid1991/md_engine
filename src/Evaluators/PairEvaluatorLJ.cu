#include "boost_for_export.h"
#include "PairEvaluatorLJ.h"

EvaluatorLJ::EvaluatorLJ() {};

// so, the force calculation assumes theres a vector 'dr', some 'multiplier', along with params..
//inline __device__ real3 force(real3 dr, real params[3], real lenSqr, real multiplier) {

// likewise, energy assumes lenSqr is available, as is some multiplier and params[3]..
//inline __device__ real energy(real params[3], real lenSqr, real multiplier) {


// python interface that calls force function
__host__ Vector EvaluatorLJ::forcePy(double sigma_, double epsilon_, double rcut_, Vector dr_) {
// we're calling the following function
//inline __device__ real3 force(real3 dr, real params[3], real lenSqr, real multiplier) {
    real sigma = sigma_;
    real sigmaSqr = sigma * sigma;
    real sig6 = sigmaSqr * sigmaSqr * sigmaSqr; // ok, have sigma ** 6.0
    // get epsilon * 24.0
    real epsTimes24 = epsilon_ * 24.0;

    // get rcutSqr
    real rcutSqr = rcut_ * rcut_;
    real params [] = {rcutSqr, epsTimes24, sig6};
    real3 dr = dr_.asreal3();
    real multiplier = 1.0;
    real lenSqr = (real) dr_.lenSqr();
      
    real3 forceCalculated = force(dr, params, lenSqr, multiplier);
    Vector result = Vector(forceCalculated.x, forceCalculated.y, forceCalculated.z);
    return result;
}

// python interface that calls the energy function
__host__ double EvaluatorLJ::energyPy(double sigma_, double epsilon_, double rcut_, double distance) {

    //inline __host__ __device__ real energy(real params[3], real lenSqr, real multiplier) {
    // 24 * epsilon = params[1];
    // rcutSqr = params[0];
    // sigma^6 = params[2];
    real sigma = sigma_;
    real sigmaSqr = sigma * sigma;
    real sig6 = sigmaSqr * sigmaSqr * sigmaSqr; // ok, have sigma ** 6.0
    // get epsilon * 24.0
    real epsTimes24 = epsilon_ * 24.0;

    // get rcutSqr
    real rcutSqr = rcut_ * rcut_;
    // same params set as force 
    real params [] = {rcutSqr, epsTimes24, sig6};
    real multiplier = 1.0;
    real lenSqr = distance * distance;
      
    real energyCalculated = energy(params, lenSqr, multiplier);
    double result = (double) energyCalculated;
    return result;

}

// python interface that calls the force function, and calculates on the GPU
__host__ Vector EvaluatorLJ::forcePy_device(double sigma, double epsilon, double rcut, Vector dr) {

    // nominal return value before we put the code in 
    return Vector(0.0, 0.0, 0.0);
}

// python interface that calls the energy function, and calculates on the GPU
__host__ double EvaluatorLJ::energyPy_device(double sigma, double epsilon, double rcut, double distance) {

    // nominal return value before we put the code in 
    return 0.0;

}




// exposing methods to python
void export_EvaluatorLJ() {
    boost::python::class_<EvaluatorLJ, SHARED(EvaluatorLJ), boost::noncopyable> (
        "EvaluatorLJ",
		boost::python::init<> (
		)
    )
//forcePy(double sigma_, double epsilon_, double rcut_, Vector dr_) {
    .def("force", &EvaluatorLJ::forcePy,
          (boost::python::arg("sigma"),
           boost::python::arg("epsilon"),
           boost::python::arg("rcut"),
           boost::python::arg("dr")
          )
        )
    .def("force_device", &EvaluatorLJ::forcePy_device,
          (boost::python::arg("sigma"),
           boost::python::arg("epsilon"),
           boost::python::arg("rcut"),
           boost::python::arg("dr")
          )
        )
    .def("energy", &EvaluatorLJ::energyPy,
          (boost::python::arg("sigma"),
           boost::python::arg("epsilon"),
           boost::python::arg("rcut"),
           boost::python::arg("distance")
          )
        )
    .def("energy_device", &EvaluatorLJ::energyPy_device,
          (boost::python::arg("sigma"),
           boost::python::arg("epsilon"),
           boost::python::arg("rcut"),
           boost::python::arg("distance")
         )
        )
    ;

}



