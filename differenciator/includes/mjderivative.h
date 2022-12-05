// https://github.com/MahanFathi/iLQG-MuJoCo/blob/master/inc/mjderivative.h
#pragma once

#include "mujoco/mujoco.h"

typedef mjtNum (*stepCostFn_t)(const mjData*);

void cpMjData(const mjModel* m, mjData* d_dest, const mjData* d_src);

void calcMJDerivatives(mjModel* m, mjData* dmain, mjtNum* deriv, stepCostFn_t stepCostFn);
