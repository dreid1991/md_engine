#include "GPUArrayDevice.h"
void MEMSETFUNC(void *ptr, void *val_, int n, int Tsize) {
    if (Tsize == 4) {
        int val = * (int *) val_;
        memsetByValList_32<<<NBLOCK(n), PERBLOCK>>>((int *) ptr, val, n);
    } else if (Tsize == 8) {
        int2 val = * (int2 *) val_;
        memsetByValList_64<<<NBLOCK(n), PERBLOCK>>>((int2 *) ptr, val, n);
    } else if (Tsize == 12) {
        int3 val = * (int3 *) val_;
        memsetByValList_96<<<NBLOCK(n), PERBLOCK>>>((int3 *) ptr, val, n);
    } else if (Tsize == 16) {
        int4 val = * (int4 *) val_;
        memsetByValList_128<<<NBLOCK(n), PERBLOCK>>>((int4 *) ptr, val, n);
    } else {
        assert(false);
    }
}
