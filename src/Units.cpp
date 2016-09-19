#include "Units.h"

void Units::setLJ() {
    boltz = 1;
    mvv_to_eng = 1;
    qqr_to_eng = 1;
    nktv_to_press = 1;
}

void Units::setReal() {
    //kcal, ang, femptoseconds
    boltz = 0.0019872067;
    mvv_to_eng = 48.88821291 * 48.88821291;
    nktv_to_press = 68568.415;
    qqr_to_eng = 332.06371;
}
