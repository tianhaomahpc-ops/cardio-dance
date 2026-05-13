// Verify EM-related config keys parse and validate.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "config/SimulationConfig.hpp"

int main() {
  using namespace mono;

  const std::string path = "em_config_parse_tmp.options";
  {
    std::ofstream f(path);
    f << "mesh_path=dummy.mesh\n"
      << "use_fiber_gf=1\n"
      << "fiber_f_path=f.gf\nfiber_s_path=s.gf\nfiber_n_path=n.gf\n"
      << "dt_pde_ms=0.01\ndt_ode_ms=0.005\nt_end_ms=1.0\n"
      << "mechanics_enable=1\n"
      << "mech_substep=100\n"
      << "mech_ho_a=0.06\nmech_ho_b=8.0\n"
      << "mech_ho_af=18.0\nmech_ho_bf=16.0\n"
      << "mech_ho_as=2.5\nmech_ho_bs=11.0\n"
      << "mech_ho_afs=0.2\nmech_ho_bfs=11.0\n"
      << "mech_ho_kappa=1000.0\n"
      << "mech_bdr_base_attr=1\nmech_bdr_endo_attr=2\nmech_bdr_epi_attr=3\n"
      << "mech_endo_pressure_pa=0.0\n"
      << "land_Tref_kPa=120.0\n"
      << "land_Ca50_uM=0.805\n"
      << "mech_snes_max_it=20\n"
      << "mech_ksp_rtol=1e-7\n";
  }
  try {
    SimulationConfig cfg = LoadConfigFile(path);
    if (!cfg.mechanics_enable) { std::cerr << "expect mechanics_enable=1\n"; return 1; }
    if (cfg.mech_substep != 100) return 2;
    if (std::abs(cfg.mech_ho_kappa - 1000.0) > 1e-9) return 3;
    if (std::abs(cfg.land_Tref_kPa - 120.0) > 1e-9) return 4;
    if (cfg.mech_snes_max_it != 20) return 5;
    std::cout << "EM config parse ok\n";
    std::remove(path.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "exception: " << e.what() << "\n";
    std::remove(path.c_str());
    return 10;
  }
}
