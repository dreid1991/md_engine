#include "helpers.h"
void cumulativeSum(int *data, int n) {
    int currentVal= 0;
    for (int i=0; i<n-1; i++) { 
        int numInCell = data[i];
        data[i] = currentVal;
        currentVal += numInCell;
    }
    data[n-1] = currentVal; //okay, so now nth place has grid's starting Idx, n+1th place has ending
}


