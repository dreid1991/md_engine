#include "GPUArrayDevice.h"
void MEMSETFUNC(void *ptr, void *val, size_t n, size_t Tsize) {
    if (Tsize == 4) {
        memsetByValList_32<<<NBLOCK(n), PERBLOCK>>>((int *) ptr,
                                                    *(int *)val, n);
    } else if (Tsize == 8) {
        memsetByValList_64<<<NBLOCK(n), PERBLOCK>>>((int2 *) ptr,
                                                    *(int2 *)val, n);
    } else if (Tsize == 12) {
        memsetByValList_96<<<NBLOCK(n), PERBLOCK>>>((int3 *) ptr,
                                                    *(int3 *)val, n);
    } else if (Tsize == 16) {
        memsetByValList_128<<<NBLOCK(n), PERBLOCK>>>((int4 *) ptr,
                                                     *(int4 *)val, n);
    } else {
        assert(false);
    }
}
