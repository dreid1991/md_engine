#pragma once

class Tunable {
private:
    int nThreadPerBlock_;
public:
    void nThreadPerBlock(int set) {
        nThreadPerBlock_ = set;
    }
    int nThreadPerBlock() {
        return nThreadPerBlock_;
    }
    Tunable () {
        nThreadPerBlock_ = 256;
    }
};
