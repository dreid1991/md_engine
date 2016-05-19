#ifndef INTEGRATORLANGEVIN_H
#define INTEGRATORLANGEVIN_H

#include "IntegratorVerlet.h"
#include "GPUArrayGlobal.h"
#include "Bounds.h"

void export_IntegratorLangevin();

/*! \class IntegratorLangevin
 * \brief Langevin dynamics integrator
 *
 * Two step Langevin dynamics 
 */
class IntegratorLangevin : public IntegratorVerlet {

private:
    int seed;
    float T;
    float gamma;
    
    // thermostat parameters
    std::vector<double> intervals;
    std::vector<double> temps;
    int curInterval;
    double curTemperature();
    bool finished;

    // bounds parameters
    boost::shared_ptr<Bounds> thermoBounds;   
    bool usingBounds;
    BoundsGPU boundsGPU;        
    
    //unsigned int groupTag;
    void preForce(uint);        
    void postForce(uint,int);/*! \todo handle 2D case  */

    // debug
    GPUArrayGlobal<float> VDotV;
    
public:
    /*! \brief Constant temperature Constructor */
    IntegratorLangevin(State *state_,/*string groupHandle_,*/float T_);

    /*! \brief Thermostat Constructor python list */
    IntegratorLangevin(State *state_, /*string groupHandle_,*/
                       boost::python::list intervals, boost::python::list temps,
                       boost::shared_ptr<Bounds> thermoBounds_ = boost::shared_ptr<Bounds>(NULL));
    /*! \brief Thermostat Constructor */
    IntegratorLangevin(State *state_, /*string groupHandle_,*/
                       std::vector<double> intervals, std::vector<double> temps,
                       boost::shared_ptr<Bounds> thermoBounds_ = boost::shared_ptr<Bounds>(NULL));

    void run(int);

    /*! \brief set the parameters
     *
     * \param seed for BD force
     * \param T temperature
     * \param gamma drag coeffictient
     */
    void set_params(int seed_ /*default 0*/, float gamma_ /*default 1.0*/) {
        seed = seed_;
        gamma = gamma_;
    }

};

#endif
