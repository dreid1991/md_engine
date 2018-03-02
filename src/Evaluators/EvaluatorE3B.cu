#include "boost_for_export.h"
#include "EvaluatorE3B.h"
#include <vector>
#include <stdio.h>
// ctors are in header file


// for testing device code
/*
template <class EVALUATOR>
__global__ void call_force_EvaluatorE3B(real *  __restrict__  distFromWall,  // input
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
*/

// for testing device code
/*
template <class EVALUATOR>
__global__ void call_energy_EvaluatorE3B(real *  __restrict__  distFromWall,
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
*/


// call 'force' via python interface - for use with test suite,
// ---- these positions /must/ be minimum image positions with respect to each other.
//      
__host__ std::vector<double> EvaluatorE3B::forcePy (boost::python::list positions_) {
    // cast as real, real3 so that we can directly call force() function
    // --- so, it should just be a list of doubles.  No 'Vector' type or anything
    std::vector<double> positions;
    int len = boost::python::len(positions_);
    for (int i=0; i<len; i++) {
        boost::python::extract<double> valPy(positions_[i]);
        if (!valPy.check()) {
            assert(valPy.check());
        }
        double val = valPy;
        positions.push_back(val);
    }

    for (size_t i = 0; i < positions.size(); i++) {
        std::cout << "Found position " << positions[i] << " at index " << i << std::endl;
    }
    // TODO implement this
    return positions;
};

// call 'force' via python interface - for use with test suite,
__host__ std::vector<double> EvaluatorE3B::forcePy_device (boost::python::list positions_) {
    /*
    // cast as real, real3 so that we can directly call force() function
    real distFromWall = (real) distanceFromWall;
    real3 forceDir = forceDirection.asreal3();

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
    SAFECALL((call_force_EvaluatorE3B<EvaluatorE3B><<<1,1,0>>>(d_distFromWall, d_forceDir, d_forceResultGPU, nthreads, *this)));

    CUT_CHECK_ERROR("call_force() execution failed\n");
    
    CUCHECK(cudaDeviceSynchronize());
    
    CUCHECK(cudaMemcpy(h_forceResultCPU, d_forceResultGPU, sizeof(real3), cudaMemcpyDeviceToHost));
    
    CUCHECK(cudaFree(d_forceResultGPU));
    CUCHECK(cudaFree(d_forceDir));
    CUCHECK(cudaFree(d_distFromWall));

    CUCHECK(cudaDeviceReset());

    //real3 result = force(distFromWall, forceDir);
    // return to Vector type, since this is being called from pytest
    real3  resultFromPointer = *h_forceResultCPU;
    Vector resultToReturn = Vector(resultFromPointer.x, resultFromPointer.y, resultFromPointer.z);
    return resultToReturn;
    */
    return std::vector<double>();
};

__host__ std::vector<double> EvaluatorE3B::energyPy (boost::python::list positions_) {
    // cast as real ( could be no change; or, real could be float )
    // -- ok, extracting the value of the positions
    std::vector<double> positions;
    int len = boost::python::len(positions_);
    for (int i=0; i<len; i++) {
        boost::python::extract<double> valPy(positions_[i]);
        if (!valPy.check()) {
            assert(valPy.check());
        }
        double val = valPy;
        positions.push_back(val);
    }

    for (size_t i = 0; i < positions.size(); i++) {
        std::cout << "Found position " << positions[i] << " at index " << i << std::endl;
    }
    // let's assume that we pass in a vector containing the positions of 3 molecules..
    std::vector<real3> positionsAsReal3;
    for (size_t i = 0; i < positions.size(); i += 3) {
        positionsAsReal3.push_back(make_real3(positions[i],positions[i+1],positions[i+2]));
    };
 
    auto shuffle_vectors = [this] (std::vector<real3> atoms) {

        // ok, we have our molecules as i,j,k;
        // get them as j,k,i
        // then k,i,j
        // --- i.e., put the first three entries at the end
        std::vector<real3> val;
        // skip the first three, put the rest in val
        for (size_t i = 3; i < atoms.size(); i++) {
            val.push_back(atoms[i]);
        }
        // put the first three at the end
        val.push_back(atoms[0]);
        val.push_back(atoms[1]);
        val.push_back(atoms[2]);
        // return val
        return val;
    };

    real eng_sum_a, eng_sum_b, eng_sum_c;
    real tmp_a, tmp_b, tmp_c;
    eng_sum_a = 0.0;
    eng_sum_b = 0.0;
    eng_sum_c = 0.0;
    real3 r_a1b2,r_a1c2,r_b1a2,r_c1a2;
    real3 r_a1b3,r_a1c3,r_b1a3,r_c1a3;
    real3 r_a2b3,r_a2c3,r_b2a3,r_c2a3;
    real  r_a1b2_scalar,r_a1c2_scalar,r_b1a2_scalar,r_c1a2_scalar;
    real  r_a1b3_scalar,r_a1c3_scalar,r_b1a3_scalar,r_c1a3_scalar;
    real  r_a2b3_scalar,r_a2c3_scalar,r_b2a3_scalar,r_c2a3_scalar;
    for (int i = 0; i < 3; i++) {
        tmp_a = tmp_b = tmp_c = 0.0;

        // at first, this is i, then j, then k
        real3 pos_a1 = positionsAsReal3[0];
        real3 pos_b1 = positionsAsReal3[1];
        real3 pos_c1 = positionsAsReal3[2];

        // j, then k, then i
        real3 pos_a2 = positionsAsReal3[3];
        real3 pos_b2 = positionsAsReal3[4];
        real3 pos_c2 = positionsAsReal3[5];

        // k, then i, then j
        real3 pos_a3 = positionsAsReal3[6];
        real3 pos_b3 = positionsAsReal3[7];
        real3 pos_c3 = positionsAsReal3[8];

        // get the vectors between the atoms..
        r_a1b2 = pos_a1 - pos_b2;
        r_a1b2_scalar = length(r_a1b2);
        // rij: i = a1, j = c2
        r_a1c2 = (pos_a1 - pos_c2);
        r_a1c2_scalar = length(r_a1c2);
        // rij: i = b1, j = a2
        r_b1a2 = (pos_b1 - pos_a2);
        r_b1a2_scalar = length(r_b1a2);
        // rij: i = c1, j = a2
        r_c1a2 = (pos_c1 - pos_a2);
        r_c1a2_scalar = length(r_c1a2); 
        // i = 1, j = 3
        //
        // rij: i = a1, j = b3
        r_a1b3 = (pos_a1 - pos_b3);
        r_a1b3_scalar = length(r_a1b3);
        // rij: i = a1, j = c3
        r_a1c3 = (pos_a1 - pos_c3);
        r_a1c3_scalar = length(r_a1c3);
        // rij: i = b1, j = a3
        r_b1a3 = (pos_b1 - pos_a3);
        r_b1a3_scalar = length(r_b1a3);
        // rij: i = c1, j = a3
        r_c1a3 = (pos_c1 - pos_a3);
        r_c1a3_scalar = length(r_c1a3);

        // i = 2, j = 3
        //
        // rij: i = a2, j = b3
        r_a2b3 = (pos_a2 - pos_b3);
        r_a2b3_scalar = length(r_a2b3);
        // rij: i = a2, j = c3
        r_a2c3 = (pos_a2 - pos_c3);
        r_a2c3_scalar = length(r_a2c3);
        // rij: i = b2, j = a3
        r_b2a3 = (pos_b2 - pos_a3);
        r_b2a3_scalar = length(r_b2a3);
        // rij: i = c2, j = a3
        r_c2a3 = (pos_c2 - pos_a3);
        r_c2a3_scalar = length(r_c2a3);

        // we now have th positions of our three molecules and their constituent atoms
        this->threeBodyEnergy(tmp_a, tmp_b, tmp_c,
                              r_a1b2_scalar, r_a1c2_scalar, r_b1a2_scalar, r_c1a2_scalar,
                              r_a1b3_scalar, r_a1c3_scalar, r_b1a3_scalar, r_c1a3_scalar,
                              r_a2b3_scalar, r_a2c3_scalar, r_b2a3_scalar, r_c2a3_scalar);

        // we are deliberately summing over this triplet three times,
        // as to simulate the approach that is natural to the GPU, and the approach 
        // that we take in practice.
        eng_sum_a += (tmp_a);
        eng_sum_b += (tmp_b);
        eng_sum_c += (tmp_c);

        // shuffle the vectors
        positionsAsReal3 = shuffle_vectors(positionsAsReal3);
    }
    std::vector<double> engSumsAsVector;
    engSumsAsVector.push_back(eng_sum_a);
    engSumsAsVector.push_back(eng_sum_b);
    engSumsAsVector.push_back(eng_sum_c);

    return engSumsAsVector;
};

__host__ std::vector<double> EvaluatorE3B::energyPy_device (boost::python::list positions_){
    // cast as real ( could be no change; or, real could be float )
    /*
    real distFromWall = (real) distance_;

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
    SAFECALL((call_energy_EvaluatorE3B<EvaluatorE3B><<<1,1,0>>>(d_distFromWall, d_energyCalculated, nthreads, *this)));

    CUT_CHECK_ERROR("call_energy() execution failed\n");
    
    CUCHECK(cudaDeviceSynchronize());
    
    CUCHECK(cudaMemcpy(h_energyCalculated, d_energyCalculated, sizeof(real), cudaMemcpyDeviceToHost));
    
    CUCHECK(cudaFree(d_distFromWall));
    CUCHECK(cudaFree(d_energyCalculated));

    CUCHECK(cudaDeviceReset());

    real  resultToReturn = *h_energyCalculated;
    return resultToReturn;
    */
    return std::vector<double>();
};

// exposing methods to python
void export_EvaluatorE3B() {
    boost::python::class_<EvaluatorE3B, SHARED(EvaluatorE3B), boost::noncopyable> (
        "EvaluatorE3B",
		boost::python::init<double,double,double,double,double,double,double,double> (
			boost::python::args("rs","rf","E2",
                                "Ea", "Eb", "Ec",
                                "k2", "k3")
		)
    )
    // needs the initial position of the atoms... input these as a list of coordinates
    .def("force", &EvaluatorE3B::forcePy,
          (boost::python::arg("positions")
          )
        )
    // -- just needs the initial positions of the atoms
    .def("energy", &EvaluatorE3B::energyPy,
         (boost::python::arg("positions")
         )
        )
    .def("force_device", &EvaluatorE3B::forcePy_device,
         (boost::python::arg("positions")
         )
        )
    .def("energy_device", &EvaluatorE3B::energyPy_device,
         (boost::python::arg("positions")
         )
        )
    .def("switching", &EvaluatorE3B::switching,
         (boost::python::arg("r")
         )
        )
    .def("dswitchdr", &EvaluatorE3B::dswitchdr,
         (boost::python::arg("r")
         )
        )
    .def("threeBodyForceScalar", &EvaluatorE3B::threeBodyForceScalar,
         (boost::python::arg("r1"),
          boost::python::arg("r2"),
          boost::python::arg("prefactor")
         )
        )

    ;

}
