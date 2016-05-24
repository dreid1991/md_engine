#pragma once
#ifndef VIRIAL_H
#define VIRIAL_H

class Virial {
    public:
        float3 diagTerms;
        float3 crossTerms;
        Virial() : diagTerms(make_float3(0, 0, 0)), crossTerms(make_float3(0, 0, 0)) {};
};

#endif
