#pragma once
#ifndef FIX_PAIR_H
#define FIX_PAIR_H

#define DEFAULT_FILL INT_MAX

#include <climits>
#include <map>
#include <string>
#include <vector>
#include <iostream>//makes it compile on my machine  (error: cout is not a member of std)

#include "AtomParams.h"
#include "GPUArrayGlobal.h"
#include "Fix.h"
#include "xml_func.h"

void export_FixPair();

class State;

//! Global function returning a single SquareVector item
/*!
 * \param vals Pointer to SquareVector array
 * \param nCol Number of columns
 * \param i Row
 * \param j Column
 *
 * \returns Element (i,j) from vals
 *
 * This function returns a single element from a given SquareVector array.
 */
template <class T>
__host__ __device__ T squareVectorItem(T *vals, int nCol, int i, int j) {
    return vals[i*nCol + j];
}
inline __device__ int squareVectorIndex(int nCol, int i, int j) {
    return i*nCol + j;
}

//! Global function returning a reference to a single Square Vector item
/*!
 * \param vals Pointer to SquareVector array
 * \param nCol Number of columns
 * \param i Row
 * \param j Column
 *
 * \returns Reference to element (i,j) from vals
 *
 * This function returns a reference to specific entry of a given SquareVector
 * array.
 */
template <class T>
__host__ __device__ T &squareVectorRef(T *vals, int nCol, int i, int j) {
    return vals[i*nCol + j];
}

//! Two dimensional array
/*!
 * This namespace contains functions to create std::vector elements which can
 * be treated as two-dimensional arrays.
 */
namespace SquareVector {

    //! Create SquareVector
    /*!
     * \tparam T Type of data stored in the SquareVector
     * \param size Number of rows and columns
     *
     * \returns New SquareVector
     *
     * This function creates a new SquareVector
     */
    template <class T>
    std::vector<T> create(int size) {
        return std::vector<T>(size*size, DEFAULT_FILL);
    }

    //! Set the diagonal elements of the SquareVector
    /*!
     * \tparam T Type of data stored in the SquareVector
     * \param vec Pointer to the vector to be modified
     * \param size Number of rows of the square vector
     * \param fillFunction Function taking no argument, returning the values
     *                     for the diagonal elements
     *
     * Set the diagonal elements to a value determined by the function passed.
     */
    template <class T>
    void populateDiagonal(std::vector<T> *vec, int size,
            std::function<T ()> fillFunction) {
        for (int i=0; i<size; i++) {
            T val = squareVectorRef<T>(vec->data(), size, i, i);
            if (val == DEFAULT_FILL) {
                squareVectorRef<T>(vec->data(), size, i, i) = fillFunction();
            }
        }
    }

    //! Fill SquareVector with values
    /*!
     * \tparam T Type of data stored in the SquareVector
     * \param vec Pointer to SquareVector
     * \param size Number of Rows/Columns of the SquareVector
     * \param fillFunction Funtion pointer to the function used to determine
     *                     the elements in the SquareVector
     *
     * This function can be used to set the off-diagonal elements of the
     * SquareVector. The off-diagonal elements are calculated base on the
     * diagonal elements which are passed to the fillFunction. This function
     * only sets values that have not been set before.
     *
     * \todo I think this function should overwrite values previously set
     *       instead of silently doing nothing.
     */
    template <class T>
    void populate(std::vector<T> *vec, int size, std::function<T (T, T)> fillFunction) {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                T val = squareVectorRef<T>(vec->data(), size, i, j);
                if (i==j) {
                    if (val == DEFAULT_FILL) {
                        std::cout << "You have not defined interaction parameters "
                            "for atom type with index " << i << std::endl;
                        assert(val != DEFAULT_FILL);
                    }
                } else if (val == DEFAULT_FILL) {
                    squareVectorRef<T>(vec->data(), size, i, j) =
                        fillFunction(squareVectorRef<T>(vec->data(), size, i, i),
                                     squareVectorRef<T>(vec->data(), size, j, j));
                }
            }
        }
    }
    //for Fcut LJFS 
    template <class T>
    void populate(std::vector<T> *vec, int size, std::function<T (int, int)> fillFunction) {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                squareVectorRef<T>(vec->data(), size, i, j) = fillFunction(i,j);
            }
        }
    }        

    
    //in case you want it flag any unfilled parameters
    
    template <class T>
    void check_populate(std::vector<T> *vec, int size) {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                T val = squareVectorRef<T>(vec->data(), size, i, j);
                if (val == DEFAULT_FILL) {
                    std::cout << "You have not defined interaction parameters "
                        "for atom types with indices " << i <<" "<< j << std::endl;
                    assert(val != DEFAULT_FILL);
                }
            }
        }
    }    
    //! Call function on each element of the SquareVector
    /*!
     * \tparam Type of data stored in the SquareVector
     * \param vec Pointer to the vector to be modified
     * \param size Number of rows in the SquareVector
     * \param processFunction Function to be called on each element
     *
     * Call a function on each element in the SquareVector, taking the current
     * value as the argument and replacing it with the return value.
     */
    template <class T>
    void process(std::vector<T> *vec, int size, std::function<T (T)> processFunction) {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                squareVectorRef<T>(vec->data(), size, i, j) =
                        processFunction(squareVectorRef<T>(vec->data(), size, i, j));
            }
        }
    }

    //! Copy SquareVector onto another SquareVector with a different size
    /*!
     * \tparam T Type of data stored in the SquareVector
     * \param other Reference to old SquareVector
     * \param oldSize Number of Rows/Columns of old SquareVector
     * \param newSize Number of Rows/Columns of new SquareVector
     *
     * \returns New SquareVector
     *
     * This Function copies a SquareVector and gives it a new size.
     */
    template <class T>
    std::vector<T> copyToSize(std::vector<T> &other, int oldSize, int newSize) {
        std::vector<T> replacement(newSize*newSize, DEFAULT_FILL);
        int copyUpTo = std::fmin(oldSize, newSize);
        for (int i=0; i<copyUpTo; i++) {
            for (int j=0; j<copyUpTo; j++) {
                squareVectorRef<T>(replacement.data(), newSize, i, j) =
                            squareVectorItem<T>(other.data(), oldSize, i, j);
            }
        }
        return replacement;
    }
} // namespace SquareVector

//! Fix for pair interactions
/*!
 * This fix is the parent class for all types of pair interaction fixes.
 */
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
        : Fix(state_, handle_, groupHandle_, type_, forceSingle_, false, requiresCharges_, applyEvery_)
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
    bool readPairParams(pugi::xml_node xmlNode);

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
};
#endif
