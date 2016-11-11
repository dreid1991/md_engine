#pragma once
#ifndef FIX_PAIR_H
#define FIX_PAIR_H

#define DEFAULT_FILL -1000

#include <climits>
#include <map>
#include <string>
#include <vector>
#include <iostream>//makes it compile on my machine  (error: cout is not a member of std)

#include "AtomParams.h"
#include "GPUArrayGlobal.h"
#include "Fix.h"
#include "xml_func.h"
#include "SquareVector.h"
#include "BoundsGPU.h"
class EvaluatorWrapper;
void export_FixPair();

class State;

class FixPair : public Fix {
public:
    //! Constructor
    /*!
     * \param state_ Shared pointer to the simulation state
     * \param handle_ String specifying the Fix name
     * \param groupHandle_ String specifying the group of Atoms to act on
     * \param type_ String specifying the type of the Fix
     * \param applyEvery_ Apply this Fix every this many timesteps
     */
    FixPair(SHARED(State) state_, std::string handle_, std::string groupHandle_,
            std::string type_, bool forceSingle_, bool requiresCharges_, int applyEvery_)
        : Fix(state_, handle_, groupHandle_, type_, forceSingle_, false, requiresCharges_, applyEvery_), chargeCalcFix(nullptr)
        {
            // Empty constructor
        };

protected:
    //! Initialize the parameters
    /*!
     * \param paramHandle String containing the parameter handle
     * \param params Reference to GPUArrayGlobal which will store the
     *               parameters
     *
     * This function can be used to set the parameters of the pair
     * interaction fix.
     */
    void initializeParameters(std::string paramHandle,
                              std::vector<float> &params);

    //! Fill the vector containing the pair interaction parameters
    /*!
     * \param handle String specifying the interaction parameter to set
     * \param fillFunction Function to calculate the interaction parameters
     *                     from the diagonal elements
     * \param processFunction Function to be called on all interaction pairs
     * \param fillDiag If True, the diagonal elements are set first
     * \param fillDiagFunction Function to set the diagonal elements
     *
     * This function prepares the parameter array used in the GPU
     * calculations. If fillDiag is True, first the diagonal elements are set
     * using the fillDiagFunction. Then, the off-diagonal elements are
     * calculated from the diagonal elements using the fillFunction. Finally,
     * all elements are modified using the processFunction.
     */
    void prepareParameters(std::string handle,
                           std::function<float (float, float)> fillFunction,
                           std::function<float (float)> processFunction,
                           bool fillDiag,
                           std::function<float ()> fillDiagFunction= std::function<float ()> ());
    void prepareParameters_from_other(std::string handle,
                           std::function<float (int, int)> fillFunction,
                           std::function<float (float)> processFunction,
                           bool fillDiag,
                           std::function<int ()> fillDiagFunction= std::function<int  ()> ());    
    void prepareParameters(std::string handle,
                           std::function<float (int, int)> fillFunction);
    void prepareParameters(std::string handle,
                           std::function<float (float)> processFunction);    
    //! Send parameters to all GPU devices
    void sendAllToDevice();

    //! Ensure GPUArrayGlobal storing the parameters has right size
    /*!
     * \param array Reference to GPUArrayGlobal storing the interaction
     *              parameters
     *
     * This function checks whether the GPUArrayGlobal storing the
     * interaction parameters has the right size. If not, the array is
     * automatically resized.
     */
    void ensureParamSize(std::vector<float> &array);

    //! Read pair parameters from XML node (Not yet implemented)
    /*!
     * \param xmlNode Node to read the parameters from
     *
     * \returns True if parameters have been read successfully and False
     *          otherwise.
     *
     * \todo Implement function reading pair parameters from XML node.
     *
     * This function reads the pair parameters from a given xmlNode.
     */
    bool readFromRestart();

    //! Create restart chunk for pair parameters
    /*!
     * \param format Unused parameter
     *
     * \returns String containing the pair parameters for the different
     *          particle types.
     *
     * This function creates a restart chunk from the internally stored
     * pair parameter map. The chunk is the used for outputting the current
     * configuration.
     */
    std::string restartChunkPairParams(std::string format);

    //! Map mapping string labels onto the vectors containing the
    //! pair potential parameters
    std::map<std::string, std::vector<float> *> paramMap;

    //! Parameter map after preparing the parameters
    std::map<std::string, std::vector<float> > paramMapProcessed;

    //! Parameters to be sent to the GPU
    GPUArrayDeviceGlobal<float> paramsCoalesced;

    //! Order in which the parameters are processed
    std::vector<std::string> paramOrder;

    //! Make sure that all parameters are in paramOrder
    /*!
     * This function throws an error if parameters are missing in paramOrder
     */
    void ensureOrderGivenForAllParams();
    Fix *chargeCalcFix;
    BoundsGPU boundsLast;
    boost::shared_ptr<EvaluatorWrapper> evalWrap;
    void acceptChargePairCalc(Fix *chargeFix); 
    virtual void setEvalWrapper() = 0;
public:
    //! Set a specific parameter for specific particle types
    /*!
     * \param param String specifying the parameter to set
     * \param handleA String specifying first atom type
     * \param handleB String specigying second atom type
     * \param val Value of the parameter
     *
     * \return False always
     *
     * This function sets a specific parameter for the pair potential
     * between two atom types.
     *
     * \todo Shouldn't this function return True is parameters are set
     *       successfully?
     */
    bool setParameter(std::string param,
                      std::string handleA,
                      std::string handleB,
                      double val);

    //! Reste parameters to before processing
    /*!
     * \param handle String specifying the parameter
     */
    void handleBoundsChange();
};
#endif
