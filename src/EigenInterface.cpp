#include "EigenInterface.h"
#include "../Eigen/Dense"


std::vector<double> eigenInterface_rotate(std::vector<double> xs, double *axis, double *com,  double theta) {
    // xs will be a multiple of 3 

    Eigen::Vector3d axisEig = {axis[0], axis[1], axis[2]};
    Eigen::AngleAxisd ax(theta, axisEig);
    Eigen::Matrix3d rot;
    rot = ax;

    Eigen::Vector3d comEig = {com[0], com[1], com[2]};

    std::vector<double> valsToReturn;
    size_t numAtoms = xs.size() / 3;
    for (size_t i = 0; i < numAtoms; i++) {
        int ix = i * 3;     // index of x coord
        int iy = i * 3 + 1; // index of y coord
        int iz = i * 3 + 2; // index of z coord

        double xpos = xs[ix];
        double ypos = xs[iy];
        double zpos = xs[iz];

        Eigen::Vector3d posEig = {xpos, ypos, zpos};
        Eigen::Vector3d relEig = posEig - comEig;

        relEig = rot * relEig;
        valsToReturn.push_back(relEig[0]);
        valsToReturn.push_back(relEig[1]);
        valsToReturn.push_back(relEig[2]);
    }
}
