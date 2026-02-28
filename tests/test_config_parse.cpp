#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "config/SimulationConfig.hpp"

int main() {
  // Smoke test for key=value parser and basic type conversions.
  const std::string path = "test_config_parse.options";
  {
    std::ofstream out(path);
    out << "mesh_path=benchmarks/niederer/niederer_benchmark.mesh\n";
    out << "enable_wholebody=1\n";
    out << "torso_mesh_path=benchmarks/torso_box/torso_box.mesh\n";
    out << "dt_pde_ms=0.02\n";
    out << "dt_ode_ms=0.01\n";
    out << "t_end_ms=1.0\n";
    out << "sigma_i_f_mS_per_mm=0.174\n";
    out << "sigma_i_s_mS_per_mm=0.019\n";
    out << "sigma_i_n_mS_per_mm=0.019\n";
    out << "sigma_e_f_mS_per_mm=0.625\n";
    out << "sigma_e_s_mS_per_mm=0.236\n";
    out << "sigma_e_n_mS_per_mm=0.236\n";
    out << "sigma_torso_mS_per_mm=0.2\n";
    out << "interface_map_max_dist_mm=1.5\n";
    out << "torso_dirichlet_penalty=2e6\n";
    out << "heart_interface_bdr_attrs=1,2\n";
    out << "stim_region=ball,0.0,0.0,0.0,1.5\n";
    out << "stim_region=box,0.0,0.0,0.0,1.0,2.0,3.0\n";
    out << "use_hypre_boomeramg=1\n";
    out << "petsc_use_geometric_asm=1\n";
    out << "petsc_asm_nx=2\n";
    out << "petsc_asm_ny=3\n";
    out << "petsc_asm_nz=4\n";
    out << "enable_electromech=1\n";
    out << "mech_order=2\n";
    out << "mech_young_modulus=12.5\n";
    out << "mech_poisson_ratio=0.29\n";
    out << "mech_traction_x=0.8\n";
    out << "mech_ca_half_mM=0.0003\n";
    out << "mech_ca_hill=3.0\n";
    out << "mech_output_subdir=heart/electromech\n";
    out << "use_petsc=0\n";
  }

  try {
    auto cfg = mono::LoadConfigFile(path);
    if (cfg.mesh_path.empty()) return 1;
    if (cfg.dt_pde_ms != 0.02) return 2;
    if (cfg.dt_ode_ms != 0.01) return 3;
    if (cfg.use_petsc) return 4;
    if (cfg.stim_regions.size() != 2) return 5;
    if (cfg.stim_regions[0].type != mono::StimulusRegion::Type::Ball) return 6;
    if (std::abs(cfg.stim_regions[0].x2 - 1.5) > 1e-12) return 7;
    if (cfg.stim_regions[1].type != mono::StimulusRegion::Type::Box) return 8;
    if (std::abs(cfg.stim_regions[1].z2 - 3.0) > 1e-12) return 9;
    if (!cfg.enable_wholebody) return 11;
    if (cfg.torso_mesh_path.empty()) return 12;
    if (cfg.heart_interface_bdr_attrs.size() != 2) return 13;
    if (std::abs(cfg.torso_dirichlet_penalty - 2e6) > 1e-12) return 14;
    if (!cfg.use_hypre_boomeramg) return 15;
    if (!cfg.petsc_use_geometric_asm) return 16;
    if (cfg.petsc_asm_nx != 2 || cfg.petsc_asm_ny != 3 || cfg.petsc_asm_nz != 4) return 17;
    if (!cfg.enable_electromech) return 18;
    if (cfg.mech_order != 2) return 19;
    if (std::abs(cfg.mech_young_modulus - 12.5) > 1e-12) return 20;
    if (std::abs(cfg.mech_poisson_ratio - 0.29) > 1e-12) return 21;
    if (std::abs(cfg.mech_traction_x - 0.8) > 1e-12) return 22;
    if (std::abs(cfg.mech_ca_half_mM - 3e-4) > 1e-12) return 23;
    if (std::abs(cfg.mech_ca_hill - 3.0) > 1e-12) return 24;
    if (cfg.mech_output_subdir != "heart/electromech") return 25;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 10;
  }

  // mesh_path is mandatory.
  const std::string bad_path = "test_config_parse_missing_mesh.options";
  {
    std::ofstream out(bad_path);
    out << "dt_pde_ms=0.02\n";
    out << "dt_ode_ms=0.01\n";
  }
  try {
    (void)mono::LoadConfigFile(bad_path);
    return 11;
  } catch (const std::exception&) {
  }

  // wholebody mode requires torso_mesh_path.
  const std::string bad_wholebody_path = "test_config_parse_bad_wholebody.options";
  {
    std::ofstream out(bad_wholebody_path);
    out << "mesh_path=benchmarks/niederer/niederer_benchmark.mesh\n";
    out << "enable_wholebody=1\n";
  }
  try {
    (void)mono::LoadConfigFile(bad_wholebody_path);
    return 12;
  } catch (const std::exception&) {
  }

  // conforming wholebody mode can omit mesh_path but requires wholebody mesh + attrs.
  const std::string conforming_path = "test_config_parse_conforming.options";
  {
    std::ofstream out(conforming_path);
    out << "enable_wholebody=1\n";
    out << "use_conforming_wholebody=1\n";
    out << "wholebody_mesh_path=benchmarks/wholebody/wholebody.mesh\n";
    out << "heart_volume_attrs=1\n";
    out << "torso_volume_attrs=2\n";
  }
  try {
    auto cfg = mono::LoadConfigFile(conforming_path);
    if (!cfg.use_conforming_wholebody) return 13;
    if (cfg.wholebody_mesh_path.empty()) return 14;
    if (cfg.heart_volume_attrs.size() != 1 || cfg.torso_volume_attrs.size() != 1) return 15;
  } catch (const std::exception&) {
    return 16;
  }

  // use_conforming_wholebody requires enable_wholebody.
  const std::string bad_conforming_flag = "test_config_parse_bad_conforming_flag.options";
  {
    std::ofstream out(bad_conforming_flag);
    out << "use_conforming_wholebody=1\n";
    out << "wholebody_mesh_path=benchmarks/wholebody/wholebody.mesh\n";
    out << "heart_volume_attrs=1\n";
    out << "torso_volume_attrs=2\n";
  }
  try {
    (void)mono::LoadConfigFile(bad_conforming_flag);
    return 17;
  } catch (const std::exception&) {
  }

  // regional heart model config parse.
  const std::string regional_path = "test_config_parse_regional.options";
  {
    std::ofstream out(regional_path);
    out << "enable_wholebody=1\n";
    out << "use_conforming_wholebody=1\n";
    out << "wholebody_mesh_path=benchmarks/wholebody/wholebody.mesh\n";
    out << "heart_volume_attrs=11,12,13\n";
    out << "torso_volume_attrs=2\n";
    out << "enable_regional_heart_models=1\n";
    out << "atria_volume_attrs=11\n";
    out << "ventricles_volume_attrs=12\n";
    out << "fibrosis_volume_attrs=13\n";
    out << "fibrosis_sigma_scale=0.1\n";
  }
  try {
    auto cfg = mono::LoadConfigFile(regional_path);
    if (!cfg.enable_regional_heart_models) return 18;
    if (cfg.atria_volume_attrs.size() != 1 || cfg.ventricles_volume_attrs.size() != 1 ||
        cfg.fibrosis_volume_attrs.size() != 1) {
      return 19;
    }
    if (std::abs(cfg.fibrosis_sigma_scale - 0.1) > 1e-12) return 20;
  } catch (const std::exception&) {
    return 21;
  }

  const std::string bad_regional_overlap = "test_config_parse_bad_regional_overlap.options";
  {
    std::ofstream out(bad_regional_overlap);
    out << "mesh_path=benchmarks/niederer/niederer_benchmark.mesh\n";
    out << "enable_regional_heart_models=1\n";
    out << "atria_volume_attrs=11\n";
    out << "ventricles_volume_attrs=11\n";
    out << "fibrosis_volume_attrs=13\n";
  }
  try {
    (void)mono::LoadConfigFile(bad_regional_overlap);
    return 22;
  } catch (const std::exception&) {
  }

  return 0;
}
