#include "boost_for_export.h"
#include "WallEvaluatorHarmonic.h"
// our non-default ctor
EvaluatorWallHarmonic::EvaluatorWallHarmonic (real k_, real r0_) {
            k = k_;
            r0= r0_;
            mdAssert(r0 > 0.0, "Cutoff in EvaluatorWallHarmonic must be greater than 0!");
            mdAssert(k > 0.0, "Spring constant k in EvaluatorWallHarmonic must be greater than 0!");
};

// for testing device code
template <class EVALUATOR>
__global__ void call_force_EvaluatorWallHarmonic(real *  __restrict__  distFromWall,  // input
                                                  real3 * __restrict__ forceDirection, // input
                                                  real3 * forceToReturn,               // output
                                                  int nthreads,                       // nthreads   
                                                  EVALUATOR eval) {                  

    int idx = GETIDX();
    if (idx < nthreads) {
        real distance = distFromWall[idx];
        real3 direction = forceDirection[idx];
        real3 calculatedForce = eval.force(distance,direction);
        forceToReturn[idx] = calculatedForce;
    }
}


// for testing device code
template <class EVALUATOR>
__global__ void call_energy_EvaluatorWallHarmonic(real *  __restrict__  distFromWall,
                                                  real * energyToReturn,
                                                  int nthreads,
                                                  EVALUATOR eval) {

    int idx = GETIDX();
    if (idx < nthreads) {
        real distance = distFromWall[idx];
        real calculatedEnergy = eval.energy(distance);
        energyToReturn[idx] = calculatedEnergy;
    }
}



// call 'force' via python interface - for use with test suite,
__host__ Vector EvaluatorWallHarmonic::forcePy (double distanceFromWall, Vector forceDirection) {
    // cast as real, real3 so that we can directly call force() function
    real distFromWall = (real) distanceFromWall;
    real3 forceDir = forceDirection.asreal3();
    real3 result = force(distFromWall, forceDir);
    // return to Vector type, since this is being called from pytest
    Vector resultToReturn = Vector(result.x, result.y, result.z);
    return resultToReturn;
};

// call 'force' via python interface - for use with test suite,
__host__ Vector EvaluatorWallHarmonic::forcePy_device (double distanceFromWall, Vector forceDirection) {
    // cast as real, real3 so that we can directly call force() function
    real distFromWall = (real) distanceFromWall;
    real3 forceDir = forceDirection.asreal3();

    /* allocate memory on host and device */
    real3 *h_forceResultCPU;
    real3 *d_forceResultGPU;
    real3 *d_forceDir;
    real  *d_distFromWall;
    real  *h_distFromWall = &distFromWall;
    real3 *h_forceDir     = &forceDir;
    h_forceResultCPU = (real3 *)malloc(sizeof(real3));
    // use checkCudaErrors; we don't care about performance - this function is used in the test suite only
    CUCHECK(cudaMalloc((void **) &d_forceResultGPU, sizeof(real3)));
    CUCHECK(cudaMalloc((void **) &d_forceDir,       sizeof(real3)));
    CUCHECK(cudaMalloc((void **) &d_distFromWall,   sizeof(real)));

    // copy the input data; *dest, *src, size, kind
    CUCHECK(cudaMemcpy(d_forceDir,     h_forceDir,     sizeof(real3), cudaMemcpyHostToDevice));
    CUCHECK(cudaMemcpy(d_distFromWall, h_distFromWall, sizeof(real),  cudaMemcpyHostToDevice));

    // do the force calculation on device, and write to d_forceResultGPU
    int nthreads = 1; // we only use one thread
    SAFECALL((call_force_EvaluatorWallHarmonic<EvaluatorWallHarmonic><<<1,1,0>>>(d_distFromWall, d_forceDir, d_forceResultGPU, nthreads, *this)));

    /* check if call was successful */
    CUT_CHECK_ERROR("call_force() execution failed\n");
    
    /* synchronize devices */
    CUCHECK(cudaDeviceSynchronize());
    
    /* Retrieve results from GPU */
    CUCHECK(cudaMemcpy(h_forceResultCPU, d_forceResultGPU, sizeof(real3), cudaMemcpyDeviceToHost));
    
    /* Free the memory on device */
    CUCHECK(cudaFree(d_forceResultGPU));
    CUCHECK(cudaFree(d_forceDir));
    CUCHECK(cudaFree(d_distFromWall));

    /* This will cause catastrophic failure if called outside of the test suite - as it should :) */
    CUCHECK(cudaDeviceReset());

    //real3 result = force(distFromWall, forceDir);
    // return to Vector type, since this is being called from pytest
    real3  resultFromPointer = *h_forceResultCPU;
    Vector resultToReturn = Vector(resultFromPointer.x, resultFromPointer.y, resultFromPointer.z);
    return resultToReturn;
};

__host__ double EvaluatorWallHarmonic::energyPy (double distance_) {
    // cast as real ( could be no change; or, real could be float )
    real distance = (real) distance_;
    real result = energy(distance);
    double resultToReturn = (double) result;
    return resultToReturn;
};

__host__ double EvaluatorWallHarmonic::energyPy_device (double distance_) {
    // cast as real ( could be no change; or, real could be float )
    real distFromWall = (real) distance_;

    /* allocate memory on host and device */
    real  *d_energyCalculated;
    real  *d_distFromWall;

    real  *h_energyCalculated;
    real  *h_distFromWall = &distFromWall;

    h_energyCalculated = (real *)malloc(sizeof(real));
    
    // use CUCHECK; we don't care about performance - this function is used in the test suite only
    CUCHECK(cudaMalloc((void **) &d_distFromWall,   sizeof(real)));
    CUCHECK(cudaMalloc((void **) &d_energyCalculated, sizeof(real)));

    // copy the input data; *dest, *src, size, kind; no input data for d_energyCalculated
    CUCHECK(cudaMemcpy(d_distFromWall, h_distFromWall, sizeof(real),  cudaMemcpyHostToDevice));

    // do the force calculation on device, and write to d_forceResultGPU
    int nthreads = 1; // we only use one thread
    SAFECALL((call_energy_EvaluatorWallHarmonic<EvaluatorWallHarmonic><<<1,1,0>>>(d_distFromWall, d_energyCalculated, nthreads, *this)));

    /* check if call was successful */
    CUT_CHECK_ERROR("call_energy() execution failed\n");
    
    /* synchronize devices */
    CUCHECK(cudaDeviceSynchronize());
    
    /* Retrieve results from GPU */
    CUCHECK(cudaMemcpy(h_energyCalculated, d_energyCalculated, sizeof(real), cudaMemcpyDeviceToHost));
    
    /* Free the memory on device */
    CUCHECK(cudaFree(d_distFromWall));
    CUCHECK(cudaFree(d_energyCalculated));

    /* This will cause catastrophic failure if called outside of the test suite - as it should :) */
    CUCHECK(cudaDeviceReset());

    real  resultToReturn = *h_energyCalculated;
    return resultToReturn;
};

// exposing methods to python
void export_EvaluatorWallHarmonic() {
    boost::python::class_<EvaluatorWallHarmonic, SHARED(EvaluatorWallHarmonic), boost::noncopyable> (
        "EvaluatorWallHarmonic",
		boost::python::init<real, real> (
			boost::python::args("k", "r0")
		)
    )
    .def("force", &EvaluatorWallHarmonic::forcePy,
          (boost::python::arg("distanceFromWall"),
           boost::python::arg("forceDirection")
          )
        )
    .def("energy", &EvaluatorWallHarmonic::energyPy,
         (boost::python::arg("distanceFromWall")
         )
        )
    .def("force_device", &EvaluatorWallHarmonic::forcePy_device,
         (boost::python::arg("distanceFromWall"),
          boost::python::arg("forceDirection")
         )
        )
    .def("energy_device", &EvaluatorWallHarmonic::energyPy_device,
         (boost::python::arg("distanceFromWall")
         )
        )
    .def_readwrite("k", &EvaluatorWallHarmonic::k)
    .def_readwrite("r0",&EvaluatorWallHarmonic::r0)
    ;

}
