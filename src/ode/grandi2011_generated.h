#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void grandi2011_initConsts(double* CONSTANTS, double* RATES, double* STATES);
void grandi2011_computeRates(double VOI,
                             double* CONSTANTS,
                             double* RATES,
                             double* STATES,
                             double* ALGEBRAIC);
void grandi2011_computeVariables(double VOI,
                                 double* CONSTANTS,
                                 double* RATES,
                                 double* STATES,
                                 double* ALGEBRAIC);

#ifdef __cplusplus
}
#endif
