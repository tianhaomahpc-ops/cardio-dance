#pragma once

#include <string>
#include <vector>

namespace mono {

struct StimulusRegion {
  enum class Type { Box, Ball };
  Type type = Type::Box;

  // Box: (x1,y1,z1) and (x2,y2,z2) are opposite corners.
  // Ball: (x1,y1,z1) is center, x2 is radius; y2/z2 are unused.
  double x1 = 0.0;
  double y1 = 0.0;
  double z1 = 0.0;
  double x2 = 0.0;
  double y2 = 0.0;
  double z2 = 0.0;
};

// Runtime options for monodomain + TT06 simulations.
// Units are explicit in field names and comments to avoid mixed-unit bugs.
struct SimulationConfig {
  std::string mesh_path;

  std::string fiber_f_path;
  std::string fiber_s_path;
  std::string fiber_n_path;
  bool use_fiber_gf = false;

  bool enable_wholebody = false;
  bool use_conforming_wholebody = false;
  std::string wholebody_mesh_path;
  std::string torso_mesh_path;

  double cm_uF_per_mm2 = 0.01;
  double chi_per_mm = 140.0;

  double sigma_f_mS_per_mm = 0.1334;
  double sigma_s_mS_per_mm = 0.0176;
  double sigma_n_mS_per_mm = 0.0176;

  // Conductivity tensors used by extracellular recovery:
  // sigma_i and sigma_e are parameterized in (f,s,n) fiber coordinates.
  double sigma_i_f_mS_per_mm = 0.174;
  double sigma_i_s_mS_per_mm = 0.019;
  double sigma_i_n_mS_per_mm = 0.019;
  double sigma_e_f_mS_per_mm = 0.625;
  double sigma_e_s_mS_per_mm = 0.236;
  double sigma_e_n_mS_per_mm = 0.236;

  // Torso conductivity (isotropic scalar).
  double sigma_torso_mS_per_mm = 0.2;
  // Interface mapping and penalty-Dirichlet controls.
  double interface_map_max_dist_mm = 1.0;
  double torso_dirichlet_penalty = 1e6;
  std::vector<int> heart_volume_attrs;
  std::vector<int> torso_volume_attrs;
  std::vector<int> heart_interface_bdr_attrs;
  std::vector<int> torso_interface_bdr_attrs;

  double dt_pde_ms = 0.02;
  double dt_ode_ms = 0.01;
  double t_end_ms = 5.0;

  double stim_start_ms = 0.0;
  double stim_end_ms = 2.0;
  double stim_amp = -15.0;  // TT06 sign convention: negative pulse is depolarizing
  // Preferred stimulus definition: Cartesian box [xmin,xmax] x [ymin,ymax] x [zmin,zmax] in mm.
  // If any max <= min, code falls back to stim_fraction mode.
  double stim_xmin_mm = 0.0;
  double stim_xmax_mm = -1.0;
  double stim_ymin_mm = 0.0;
  double stim_ymax_mm = -1.0;
  double stim_zmin_mm = 0.0;
  double stim_zmax_mm = -1.0;
  double stim_fraction = 0.05;  // fallback mode: stimulate x in [xmin, xmin + fraction*(xmax-xmin)]
  // Preferred modern stimulus definition: multiple geometric regions.
  std::vector<StimulusRegion> stim_regions;

  bool use_petsc = false;
  bool use_hypre_boomeramg = false;
  bool wholebody_solve_every_step = false;
  int ksp_max_it = 500;
  double ksp_rtol = 1e-8;
  bool petsc_use_geometric_asm = true;
  int petsc_asm_nx = 1;
  int petsc_asm_ny = 1;
  int petsc_asm_nz = 1;

  int output_stride = 20;
  int checkpoint_stride = 100;
  std::string output_dir = "output";
  std::string checkpoint_dir = "checkpoint";
};

// Parse key=value config file with strict key validation.
SimulationConfig LoadConfigFile(const std::string& path);
// Apply command-line overrides for quick experiments.
void OverrideFromArgs(int argc, char* argv[], SimulationConfig& cfg);
// Render selected fields for startup logging.
std::string ToString(const SimulationConfig& cfg);

}  // namespace mono
