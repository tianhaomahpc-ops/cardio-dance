#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void initConsts(double* CONSTANTS, double* RATES, double* STATES);
void computeRates(double VOI, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC);
void computeVariables(double VOI, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC);

#ifdef __cplusplus
}
#endif
