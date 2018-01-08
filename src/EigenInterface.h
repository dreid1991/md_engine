#include <vector>

/* Interface to Eigen library
 *
 * Compiler complains if any file in which 'real' (DASH type) is seen also sees Eigen, 
 * because Eigen has a bunch of 'using std::real' statements
 *
 */


std::vector<double> eigenInterface_rotate(std::vector<double> xs, double *axis, double *com, double theta);

