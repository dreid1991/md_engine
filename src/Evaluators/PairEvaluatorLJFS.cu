#include "boost_for_export.h"
#include "PairEvaluatorLJFS.h"

EvaluatorLJFS::EvaluatorLJFS() {};

// so, the force calculation assumes theres a vector 'dr', some 'multiplier', along with params..
//inline __device__ real3 force(real3 dr, real params[3], real lenSqr, real multiplier) {

// likewise, energy assumes lenSqr is available, as is some multiplier and params[3]..
//inline __device__ real energy(real params[3], real lenSqr, real multiplier) {


// python interface that calls force function
__host__ Vector EvaluatorLJFS::forcePy(double sigma_, double epsilon_, double rcut_, Vector dr_) {
// we're calling the following function
// inline __device__ real3 force(real3 dr, real params[4], real lenSqr, real multiplier) {

    // dr is r_ij as usual..
    // params holds the following: params[0] = rcutSqr
    //                             params[1] = epsTimes24
    //                             params[2] = sig6
    //                             params[3] = unshifted LJ force at rc
    real sigma = sigma_;
    real sigmaSqr = sigma * sigma;
    real sig6 = sigmaSqr * sigmaSqr * sigmaSqr; // ok, have sigma ** 6.0
    // get epsilon * 24.0
    real epsTimes24 = epsilon_ * 24.0;

    // get rcutSqr
    real rcutSqr = rcut_ * rcut_;
    real3 dr = dr_.asreal3();
    real multiplier = 1.0;
    real lenSqr = (real) dr_.lenSqr();

    // pre-compute fcut
    real rcut_inv_sqr = 1.0 / rcutSqr;
    real rcut_inv     = 1.0 / rcut_;
    real rcut_inv_6   = rcut_inv_sqr * rcut_inv_sqr * rcut_inv_sqr;


    real fcut = epsTimes24 * rcut_inv * rcut_inv_6 * ((2.0 * sig6 * sig6 * rcut_inv_6) - sig6);

    real params [] = {rcutSqr, epsTimes24, sig6,fcut};
    real3 forceCalculated = force(dr, params, lenSqr, multiplier);
    Vector result = Vector(forceCalculated.x, forceCalculated.y, forceCalculated.z);
    return result;
}

// python interface that calls the energy function
__host__ double EvaluatorLJFS::energyPy(double sigma_, double epsilon_, double rcut_, double distance) {

    //inline __device__ real energy(real params[4], real lenSqr, real multiplier) {

    // dr is r_ij as usual..
    // params holds the following: params[0] = rcutSqr
    //                             params[1] = epsTimes24
    //                             params[2] = sig6
    //                             params[3] = unshifted LJ force at rc
    real sigma = sigma_;
    real sigmaSqr = sigma * sigma;
    real sig6 = sigmaSqr * sigmaSqr * sigmaSqr; // ok, have sigma ** 6.0
    // get epsilon * 24.0
    real epsTimes24 = epsilon_ * 24.0;

    // get rcutSqr
    real rcutSqr = rcut_ * rcut_;
    real multiplier = 1.0;
    real lenSqr = (real) ( distance * distance );

    // pre-compute fcut
    real rcut_inv_sqr = 1.0 / rcutSqr;
    real rcut_inv     = 1.0 / rcut_;
    real rcut_inv_6   = rcut_inv_sqr * rcut_inv_sqr * rcut_inv_sqr;

    real fcut = epsTimes24 * rcut_inv * rcut_inv_6 * ((2.0 * sig6 * sig6 * rcut_inv_6) - sig6);

    real params [] = {rcutSqr, epsTimes24, sig6,fcut};
    
    real energyCalculated = energy(params, lenSqr, multiplier);
    double resultToReturn = (double) energyCalculated;
    return resultToReturn;

}

// python interface that calls the force function, and calculates on the GPU
__host__ Vector EvaluatorLJFS::forcePy_device(double sigma, double epsilon, double rcut, Vector dr) {

    // nominal return value before we put the code in 
    return Vector(0.0, 0.0, 0.0);
}

// python interface that calls the energy function, and calculates on the GPU
__host__ double EvaluatorLJFS::energyPy_device(double sigma, double epsilon, double rcut, double distance) {

    // nominal return value before we put the code in 
    return 0.0;

}




// exposing methods to python
void export_EvaluatorLJFS() {
    boost::python::class_<EvaluatorLJFS, SHARED(EvaluatorLJFS), boost::noncopyable> (
        "EvaluatorLJFS",
		boost::python::init<> (
		)
    )
//forcePy(double sigma_, double epsilon_, double rcut_, Vector dr_) {
    .def("force", &EvaluatorLJFS::forcePy,
          (boost::python::arg("sigma"),
           boost::python::arg("epsilon"),
           boost::python::arg("rcut"),
           boost::python::arg("dr")
          )
        )
    .def("force_device", &EvaluatorLJFS::forcePy_device,
          (boost::python::arg("sigma"),
           boost::python::arg("epsilon"),
           boost::python::arg("rcut"),
           boost::python::arg("dr")
          )
        )
    .def("energy", &EvaluatorLJFS::energyPy,
          (boost::python::arg("sigma"),
           boost::python::arg("epsilon"),
           boost::python::arg("rcut"),
           boost::python::arg("distance")
          )
        )
    .def("energy_device", &EvaluatorLJFS::energyPy_device,
          (boost::python::arg("sigma"),
           boost::python::arg("epsilon"),
           boost::python::arg("rcut"),
           boost::python::arg("distance")
         )
        )
    ;

}



