#pragma once
#ifndef FIX_TYPES
#define FIX_TYPES
#include <string>
#include <vector>
using namespace std;
extern const string _2dType;
extern const string bondHarmType;
extern const string angleHarmType;
extern const string dihedralOPLSType;
extern const string constPressureType;
extern const string springStaticType;
extern const string dampType;
extern const string expCutType;
extern const string LJCutType;
extern const string LJRadType;
extern const string softSphereType;
extern const string NVTType;
extern const string NVTRescaleType;
extern const string scaleBoxType;
extern const string skewBoxType;
extern const string wallHarmonicType;
extern const string chargePairDSF;
extern const string improperHarmonicType;
extern const vector<string> FORCERTYPES;
#endif
