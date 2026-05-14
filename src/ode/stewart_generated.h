#pragma once

// Stewart-Aslanidi-Noble-Noble-Boyett-Zhang 2009 Purkinje cell model.
//
// Auto-generated C source from CellML via models.cellml.org/exposure/
// 38cf8387b0707f0ef6947f009710aeb5/stewart_aslanidi_noble_noble_boyett_zhang_2009.cellml
// (codegen view: @@cellml_codegen/C). Function names prefixed with
// "stewart_" to avoid clashing with tt06_generated's identical signatures.
//
// Sizes (must match the .c file):
//   STATES_SIZE    = 20
//   RATES_SIZE     = 20
//   CONSTANTS_SIZE = 52
//   ALGEBRAIC_SIZE = 76
//
// Sign convention (TT06 / CellML standard):
//   RATES[0] = -(sum of ionic currents); positive ALGEBRAIC currents are
//   outward / repolarizing.

#ifdef __cplusplus
extern "C" {
#endif

void stewart_initConsts(double* CONSTANTS, double* RATES, double* STATES);
void stewart_computeRates(double VOI, double* CONSTANTS, double* RATES,
                          double* STATES, double* ALGEBRAIC);
void stewart_computeVariables(double VOI, double* CONSTANTS, double* RATES,
                              double* STATES, double* ALGEBRAIC);

#ifdef __cplusplus
}
#endif
