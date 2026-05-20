#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "mfem.hpp"

namespace mono {

// Pseudo-ECG (extracellular potential) probe.
//
// Computes the standard far-field approximation
//
//   phi(x_p, t) = (sigma_i / (4*pi*sigma_b)) * integral_Omega
//                  ( grad V_m(r,t) . (x_p - r) / |x_p - r|^3 ) dr
//
// using element-wise gradients of V_m on the heart mesh and a piecewise
// constant per-element volume. The probe location x_p sits in an
// (assumed isotropic) extracellular bath, so sigma_b is a single scalar.
// We default to sigma_i = 1 mS/mm and sigma_b = 1 mS/mm so the output is
// in arbitrary units; absolute calibration would require knowing the
// torso conductivity properly.
//
// Each probe writes a single CSV column. The class supports multiple
// probes (one CSV file with `t, phi_lead0, phi_lead1, ...`).
//
// Sign convention: the standard formula above gives an extracellular
// potential consistent with depolarizing wavefronts producing an
// upward deflection when approaching the probe. Sources:
// Plonsey & Barr 1987; Gima & Rudy 2002 cable-pseudo-ECG; standard
// Niederer benchmark post-processing.
class PseudoEcg {
 public:
  struct Probe {
    std::string name;     // CSV column header
    double x, y, z;       // Position in mesh coordinates (mm)
  };

  PseudoEcg(MPI_Comm comm,
            const mfem::ParFiniteElementSpace& pfes,
            const std::vector<Probe>& probes,
            double sigma_i_mS_per_mm,
            double sigma_b_mS_per_mm,
            const std::string& csv_path);
  ~PseudoEcg();

  // Append a row to the CSV: t_ms, phi(probe_0), phi(probe_1), ...
  // vm_gf must live on the same FE space passed at construction.
  void Sample(double t_ms, const mfem::ParGridFunction& vm_gf);

 private:
  // Cache per-element data so Sample() doesn't re-traverse the mesh
  // structure each time.
  struct ElemCache {
    double cx, cy, cz;   // centroid
    double volume;       // mm^3
  };

  void BuildElementCache(const mfem::ParFiniteElementSpace& pfes);

  MPI_Comm comm_;
  int rank_;
  std::vector<Probe> probes_;
  double sigma_i_;
  double sigma_b_;
  std::vector<ElemCache> elem_cache_;
  const mfem::ParFiniteElementSpace* pfes_;

  // Output CSV (rank 0 only).
  std::ofstream csv_;
};

}  // namespace mono
