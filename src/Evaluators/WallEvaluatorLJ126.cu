#include "boost_for_export.h"
#include "WallEvaluatorLJ126.h"



EvaluatorWallLJ126::EvaluatorWallLJ126(real sigma_, real epsilon_, real r0_) {
        sigma = sigma_;
        epsilon = epsilon_;
        epsilonTimes24 = 24.0 * epsilon_;
        epsilonTimes4  = 4.0 * epsilon_;
        r0 = r0_;
        sig2 = sigma_ * sigma_;
        sig6 = sig2 * sig2 * sig2;
        sig12 = sig6 * sig6;

        real cutSqr = r0 * r0;
        real cut6   = cutSqr * cutSqr * cutSqr;
        real cut12  = cut6 * cut6;

        // shift /up/ so that energy at the cutoff is 0.0; note that we add, so put the negative here
        engShift = -4.0 * epsilon * ( (sig12 / cut12) - (sig6 / cut6));

        mdAssert(r0 > 0.0, "Cutoff in EvaluatorWallLJ126 must be greater than 0.0!");
        mdAssert(sigma > 0.0, "Sigma in EvaluatorWallLJ126 must be greater than 0.0!");

};

// for testing device code
template <class EVALUATOR>
__global__ void call_force_EvaluatorWallLJ126(real *  __restrict__  distFromWall,  // input
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
__global__ void call_energy_EvaluatorWallLJ126(real *  __restrict__  distFromWall,
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

__host__ Vector EvaluatorWallLJ126::forcePy(double distanceFromWall_, Vector forceDir_) {

    real distanceFromWall = (real) distanceFromWall_; 
    real3 forceDir = forceDir_.asreal3();
    real3 result = force(distanceFromWall,forceDir);
    Vector resultToReturn = Vector(result.x, result.y, result.z);
    return resultToReturn;
}

__host__ Vector EvaluatorWallLJ126::forcePy_device(double distFromWall_, Vector forceDirection) {

    // cast as real, real3 so that we can directly call force() function
    real distFromWall = (real) distFromWall_;
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
    SAFECALL((call_force_EvaluatorWallLJ126<EvaluatorWallLJ126><<<1,1,0>>>(d_distFromWall, d_forceDir, d_forceResultGPU, nthreads, *this)));

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

}

__host__ double EvaluatorWallLJ126::energyPy(double distance_) {
    
    real distance = (real) distance_;
    real result = energy(distance);
    double resultToReturn = (double) result;
    return resultToReturn;
}

__host__ double EvaluatorWallLJ126::energyPy_device(double distance_) {
    
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

    // copy the input data; *dest, *src, size, kind
    CUCHECK(cudaMemcpy(d_distFromWall, h_distFromWall, sizeof(real),  cudaMemcpyHostToDevice));

    // do the force calculation on device, and write to d_forceResultGPU
    int nthreads = 1; // we only use one thread
    SAFECALL((call_energy_EvaluatorWallLJ126<EvaluatorWallLJ126><<<1,1,0>>>(d_distFromWall, d_energyCalculated, nthreads, *this)));

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
}

// exposing methods to python
void export_EvaluatorWallLJ126() {
    boost::python::class_<EvaluatorWallLJ126, SHARED(EvaluatorWallLJ126), boost::noncopyable> (
        "EvaluatorWallLJ126",
		boost::python::init<real, real,real> (
			boost::python::args("sigma","epsilon","r0")
		)
    )
    .def("force", &EvaluatorWallLJ126::forcePy,
          (boost::python::arg("distanceFromWall"),
           boost::python::arg("forceDirection")
          )
        )
    .def("force_device", &EvaluatorWallLJ126::forcePy_device,
          (boost::python::arg("distanceFromWall"),
           boost::python::arg("forceDirection")
          )
        )
    .def("energy", &EvaluatorWallLJ126::energyPy,
         (boost::python::arg("distanceFromWall")
         )
        )
    .def("energy_device", &EvaluatorWallLJ126::energyPy_device,
         (boost::python::arg("distanceFromWall")
         )
        )
    .def_readonly("sigma", &EvaluatorWallLJ126::sigma)
    .def_readonly("epsilon", &EvaluatorWallLJ126::epsilon)
    .def_readonly("r0",&EvaluatorWallLJ126::r0)
    ;

}


