#pragma once
#ifndef FIXDPD_H
#define FIXDPD_H

#include "Fix.h"
//! Make FixDPD available to the pair base class in boost
void export_FixDPD();
//! Base class for dissipative particle dynamics fixes
/*!
 *
 *
 */

class FixDPD : public Fix {
    public:

        // some constructor here
        FixDPD () {};

}

#endif




