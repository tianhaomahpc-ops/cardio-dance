#include "config/SimulationConfig.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mono {
namespace {

// Trim leading/trailing ASCII whitespace.
std::string Trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

// Parse loose bool syntax used in options files.
bool ParseBool(const std::string& v) {
  const std::string t = Trim(v);
  if (t == "1" || t == "true" || t == "TRUE" || t == "on") {
    return true;
  }
  if (t == "0" || t == "false" || t == "FALSE" || t == "off") {
    return false;
  }
  throw std::runtime_error("Invalid bool value: " + v);
}

std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, ',')) {
    out.push_back(Trim(token));
  }
  return out;
}

std::vector<int> ParseIntList(const std::string& v) {
  const auto tokens = SplitCsv(v);
  std::vector<int> values;
  values.reserve(tokens.size());
  for (const auto& t : tokens) {
    if (t.empty()) {
      continue;
    }
    values.push_back(std::stoi(t));
  }
  return values;
}

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

StimulusRegion ParseStimRegion(const std::string& v, int line_no) {
  const auto tokens = SplitCsv(v);
  if (tokens.empty()) {
    throw std::runtime_error("Empty stim_region at line " + std::to_string(line_no));
  }

  const std::string kind = ToLower(tokens[0]);
  StimulusRegion reg;
  if (kind == "ball") {
    if (tokens.size() != 5) {
      throw std::runtime_error(
          "stim_region ball expects 5 fields: ball,cx,cy,cz,r (line " +
          std::to_string(line_no) + ")");
    }
    reg.type = StimulusRegion::Type::Ball;
    reg.x1 = std::stod(tokens[1]);
    reg.y1 = std::stod(tokens[2]);
    reg.z1 = std::stod(tokens[3]);
    reg.x2 = std::stod(tokens[4]);
    if (reg.x2 <= 0.0) {
      throw std::runtime_error("stim_region ball radius must be > 0 at line " +
                               std::to_string(line_no));
    }
    return reg;
  }

  if (kind == "box") {
    if (tokens.size() != 7) {
      throw std::runtime_error(
          "stim_region box expects 7 fields: box,x1,y1,z1,x2,y2,z2 (line " +
          std::to_string(line_no) + ")");
    }
    reg.type = StimulusRegion::Type::Box;
    reg.x1 = std::stod(tokens[1]);
    reg.y1 = std::stod(tokens[2]);
    reg.z1 = std::stod(tokens[3]);
    reg.x2 = std::stod(tokens[4]);
    reg.y2 = std::stod(tokens[5]);
    reg.z2 = std::stod(tokens[6]);
    return reg;
  }

  throw std::runtime_error("Unknown stim_region type '" + tokens[0] + "' at line " +
                           std::to_string(line_no));
}

}  // namespace

SimulationConfig LoadConfigFile(const std::string& path) {
  SimulationConfig cfg;
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open config file: " + path);
  }

  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const auto hash_pos = line.find('#');
    if (hash_pos != std::string::npos) {
      line = line.substr(0, hash_pos);
    }
    line = Trim(line);
    if (line.empty()) {
      continue;
    }

    const auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      throw std::runtime_error("Config parse error at line " + std::to_string(line_no));
    }

    const std::string key = Trim(line.substr(0, eq_pos));
    const std::string val = Trim(line.substr(eq_pos + 1));

    // Keep mapping explicit so unknown keys fail fast.
    if (key == "mesh_path") cfg.mesh_path = val;
    else if (key == "fiber_f_path") cfg.fiber_f_path = val;
    else if (key == "fiber_s_path") cfg.fiber_s_path = val;
    else if (key == "fiber_n_path") cfg.fiber_n_path = val;
    else if (key == "use_fiber_gf") cfg.use_fiber_gf = ParseBool(val);
    else if (key == "enable_wholebody") cfg.enable_wholebody = ParseBool(val);
    else if (key == "use_conforming_wholebody") cfg.use_conforming_wholebody = ParseBool(val);
    else if (key == "wholebody_mesh_path") cfg.wholebody_mesh_path = val;
    else if (key == "torso_mesh_path") cfg.torso_mesh_path = val;
    else if (key == "cm_uF_per_mm2") cfg.cm_uF_per_mm2 = std::stod(val);
    else if (key == "chi_per_mm") cfg.chi_per_mm = std::stod(val);
    else if (key == "sigma_f_mS_per_mm") cfg.sigma_f_mS_per_mm = std::stod(val);
    else if (key == "sigma_s_mS_per_mm") cfg.sigma_s_mS_per_mm = std::stod(val);
    else if (key == "sigma_n_mS_per_mm") cfg.sigma_n_mS_per_mm = std::stod(val);
    else if (key == "sigma_i_f_mS_per_mm") cfg.sigma_i_f_mS_per_mm = std::stod(val);
    else if (key == "sigma_i_s_mS_per_mm") cfg.sigma_i_s_mS_per_mm = std::stod(val);
    else if (key == "sigma_i_n_mS_per_mm") cfg.sigma_i_n_mS_per_mm = std::stod(val);
    else if (key == "sigma_e_f_mS_per_mm") cfg.sigma_e_f_mS_per_mm = std::stod(val);
    else if (key == "sigma_e_s_mS_per_mm") cfg.sigma_e_s_mS_per_mm = std::stod(val);
    else if (key == "sigma_e_n_mS_per_mm") cfg.sigma_e_n_mS_per_mm = std::stod(val);
    else if (key == "sigma_torso_mS_per_mm") cfg.sigma_torso_mS_per_mm = std::stod(val);
    else if (key == "interface_map_max_dist_mm") cfg.interface_map_max_dist_mm = std::stod(val);
    else if (key == "torso_dirichlet_penalty") cfg.torso_dirichlet_penalty = std::stod(val);
    else if (key == "heart_volume_attrs") cfg.heart_volume_attrs = ParseIntList(val);
    else if (key == "torso_volume_attrs") cfg.torso_volume_attrs = ParseIntList(val);
    else if (key == "heart_interface_bdr_attrs") cfg.heart_interface_bdr_attrs = ParseIntList(val);
    else if (key == "torso_interface_bdr_attrs") cfg.torso_interface_bdr_attrs = ParseIntList(val);
    else if (key == "dt_pde_ms") cfg.dt_pde_ms = std::stod(val);
    else if (key == "dt_ode_ms") cfg.dt_ode_ms = std::stod(val);
    else if (key == "t_end_ms") cfg.t_end_ms = std::stod(val);
    else if (key == "stim_start_ms") cfg.stim_start_ms = std::stod(val);
    else if (key == "stim_end_ms") cfg.stim_end_ms = std::stod(val);
    else if (key == "stim_amp") cfg.stim_amp = std::stod(val);
    else if (key == "stim_xmin_mm") cfg.stim_xmin_mm = std::stod(val);
    else if (key == "stim_xmax_mm") cfg.stim_xmax_mm = std::stod(val);
    else if (key == "stim_ymin_mm") cfg.stim_ymin_mm = std::stod(val);
    else if (key == "stim_ymax_mm") cfg.stim_ymax_mm = std::stod(val);
    else if (key == "stim_zmin_mm") cfg.stim_zmin_mm = std::stod(val);
    else if (key == "stim_zmax_mm") cfg.stim_zmax_mm = std::stod(val);
    else if (key == "stim_fraction") cfg.stim_fraction = std::stod(val);
    else if (key == "stim_region") cfg.stim_regions.push_back(ParseStimRegion(val, line_no));
    else if (key == "use_petsc") cfg.use_petsc = ParseBool(val);
    else if (key == "use_hypre_boomeramg") cfg.use_hypre_boomeramg = ParseBool(val);
    else if (key == "wholebody_solve_every_step") cfg.wholebody_solve_every_step = ParseBool(val);
    else if (key == "ksp_max_it") cfg.ksp_max_it = std::stoi(val);
    else if (key == "ksp_rtol") cfg.ksp_rtol = std::stod(val);
    else if (key == "petsc_use_geometric_asm") cfg.petsc_use_geometric_asm = ParseBool(val);
    else if (key == "petsc_asm_nx") cfg.petsc_asm_nx = std::stoi(val);
    else if (key == "petsc_asm_ny") cfg.petsc_asm_ny = std::stoi(val);
    else if (key == "petsc_asm_nz") cfg.petsc_asm_nz = std::stoi(val);
    else if (key == "output_stride") cfg.output_stride = std::stoi(val);
    else if (key == "checkpoint_stride") cfg.checkpoint_stride = std::stoi(val);
    else if (key == "output_dir") cfg.output_dir = val;
    else if (key == "checkpoint_dir") cfg.checkpoint_dir = val;
    // Purkinje + PVJ
    else if (key == "enable_purkinje") cfg.enable_purkinje = ParseBool(val);
    else if (key == "purkinje_network_path") cfg.purkinje_network_path = val;
    else if (key == "purkinje_cable_subdivision") cfg.purkinje_cable_subdivision = std::stoi(val);
    else if (key == "purkinje_dt_ms") cfg.purkinje_dt_ms = std::stod(val);
    else if (key == "purkinje_cm_uF_per_mm2") cfg.purkinje_cm_uF_per_mm2 = std::stod(val);
    else if (key == "purkinje_edge_g_mS_per_mm") cfg.purkinje_edge_g_mS_per_mm = std::stod(val);
    else if (key == "purkinje_leak_g_mS_per_uF") cfg.purkinje_leak_g_mS_per_uF = std::stod(val);
    else if (key == "purkinje_v_rest_mv") cfg.purkinje_v_rest_mv = std::stod(val);
    else if (key == "purkinje_stim_nodes") cfg.purkinje_stim_nodes = ParseIntList(val);
    else if (key == "purkinje_stim_start_ms") cfg.purkinje_stim_start_ms = std::stod(val);
    else if (key == "purkinje_stim_end_ms") cfg.purkinje_stim_end_ms = std::stod(val);
    else if (key == "purkinje_stim_amp_uA_per_uF") cfg.purkinje_stim_amp_uA_per_uF = std::stod(val);
    else if (key == "pvj_g_mS") cfg.pvj_g_mS = std::stod(val);
    else if (key == "pvj_max_dist_mm") cfg.pvj_max_dist_mm = std::stod(val);
    else if (key == "pvj_current_scale") cfg.pvj_current_scale = std::stod(val);
    else if (key == "pvj_delay_ms") cfg.pvj_delay_ms = std::stod(val);
    // Regional ionic
    else if (key == "enable_regional_ionic") cfg.enable_regional_ionic = ParseBool(val);
    else if (key == "atria_volume_attrs") cfg.atria_volume_attrs = ParseIntList(val);
    else if (key == "ventricle_volume_attrs") cfg.ventricle_volume_attrs = ParseIntList(val);
    else if (key == "av_delay_volume_attrs") cfg.av_delay_volume_attrs = ParseIntList(val);
    else if (key == "fibrosis_volume_attrs") cfg.fibrosis_volume_attrs = ParseIntList(val);
    else if (key == "av_delay_sigma_scale") cfg.av_delay_sigma_scale = std::stod(val);
    else if (key == "fibrosis_sigma_scale") cfg.fibrosis_sigma_scale = std::stod(val);
    else if (key == "av_delay_leak_g_mS_per_uF") cfg.av_delay_leak_g_mS_per_uF = std::stod(val);
    else if (key == "fibrosis_leak_g_mS_per_uF") cfg.fibrosis_leak_g_mS_per_uF = std::stod(val);
    else if (key == "passive_v_rest_mv") cfg.passive_v_rest_mv = std::stod(val);
    // Pseudo-ECG
    else if (key == "enable_pseudo_ecg") cfg.enable_pseudo_ecg = ParseBool(val);
    else if (key == "pseudo_ecg_sigma_i_mS_per_mm") cfg.pseudo_ecg_sigma_i_mS_per_mm = std::stod(val);
    else if (key == "pseudo_ecg_sigma_b_mS_per_mm") cfg.pseudo_ecg_sigma_b_mS_per_mm = std::stod(val);
    else if (key == "pseudo_ecg_stride") cfg.pseudo_ecg_stride = std::stoi(val);
    else if (key == "pseudo_ecg_csv") cfg.pseudo_ecg_csv = val;
    else if (key == "pseudo_ecg_probe") {
      const auto tokens = SplitCsv(val);
      if (tokens.size() != 4) {
        throw std::runtime_error(
            "pseudo_ecg_probe expects 4 fields name,x,y,z at line " +
            std::to_string(line_no));
      }
      SimulationConfig::EcgProbe p;
      p.name = tokens[0];
      p.x = std::stod(tokens[1]);
      p.y = std::stod(tokens[2]);
      p.z = std::stod(tokens[3]);
      cfg.pseudo_ecg_probes.push_back(p);
    }
    else {
      throw std::runtime_error("Unknown config key: " + key);
    }
  }

  if (cfg.dt_ode_ms <= 0.0 || cfg.dt_pde_ms <= 0.0) {
    throw std::runtime_error("dt_ode_ms and dt_pde_ms must be > 0");
  }
  if (cfg.dt_ode_ms > cfg.dt_pde_ms) {
    throw std::runtime_error("dt_ode_ms must be <= dt_pde_ms");
  }
  if (!cfg.use_conforming_wholebody && cfg.mesh_path.empty()) {
    throw std::runtime_error("mesh_path must be provided");
  }
  if (cfg.use_fiber_gf) {
    // Fiber tensor assembly requires all three orthogonal directions.
    if (cfg.fiber_f_path.empty() || cfg.fiber_s_path.empty() || cfg.fiber_n_path.empty()) {
      throw std::runtime_error("use_fiber_gf=1 requires fiber_f_path/fiber_s_path/fiber_n_path");
    }
  }
  if (cfg.use_conforming_wholebody && !cfg.enable_wholebody) {
    throw std::runtime_error("use_conforming_wholebody=1 requires enable_wholebody=1");
  }
  if (cfg.enable_wholebody) {
    if (cfg.use_conforming_wholebody) {
      if (cfg.wholebody_mesh_path.empty()) {
        throw std::runtime_error("use_conforming_wholebody=1 requires wholebody_mesh_path");
      }
      if (cfg.heart_volume_attrs.empty() || cfg.torso_volume_attrs.empty()) {
        throw std::runtime_error(
            "use_conforming_wholebody=1 requires heart_volume_attrs and torso_volume_attrs");
      }
    } else if (cfg.torso_mesh_path.empty()) {
      throw std::runtime_error("enable_wholebody=1 requires torso_mesh_path");
    }
    if (cfg.sigma_i_f_mS_per_mm <= 0.0 || cfg.sigma_i_s_mS_per_mm <= 0.0 ||
        cfg.sigma_i_n_mS_per_mm <= 0.0 || cfg.sigma_e_f_mS_per_mm <= 0.0 ||
        cfg.sigma_e_s_mS_per_mm <= 0.0 || cfg.sigma_e_n_mS_per_mm <= 0.0) {
      throw std::runtime_error("wholebody requires positive sigma_i_* and sigma_e_* values");
    }
    if (cfg.sigma_torso_mS_per_mm <= 0.0) {
      throw std::runtime_error("wholebody requires sigma_torso_mS_per_mm > 0");
    }
    if (cfg.interface_map_max_dist_mm <= 0.0) {
      throw std::runtime_error("wholebody requires interface_map_max_dist_mm > 0");
    }
    if (cfg.torso_dirichlet_penalty <= 0.0) {
      throw std::runtime_error("wholebody requires torso_dirichlet_penalty > 0");
    }
  }
  if (cfg.petsc_asm_nx <= 0 || cfg.petsc_asm_ny <= 0 || cfg.petsc_asm_nz <= 0) {
    throw std::runtime_error("petsc_asm_nx/petsc_asm_ny/petsc_asm_nz must be > 0");
  }
  if (cfg.enable_purkinje) {
    if (cfg.purkinje_network_path.empty()) {
      throw std::runtime_error("enable_purkinje=1 requires purkinje_network_path");
    }
    if (cfg.purkinje_dt_ms <= 0.0 || cfg.purkinje_dt_ms > cfg.dt_pde_ms) {
      throw std::runtime_error("purkinje_dt_ms must satisfy 0 < dt <= dt_pde_ms");
    }
    if (cfg.purkinje_cable_subdivision < 1) {
      throw std::runtime_error("purkinje_cable_subdivision must be >= 1");
    }
    if (cfg.pvj_g_mS < 0.0 || cfg.pvj_max_dist_mm <= 0.0) {
      throw std::runtime_error("pvj_g_mS must be >= 0 and pvj_max_dist_mm must be > 0");
    }
    if (cfg.pvj_delay_ms < 0.0) {
      throw std::runtime_error("pvj_delay_ms must be >= 0");
    }
  }
  return cfg;
}

void OverrideFromArgs(int argc, char* argv[], SimulationConfig& cfg) {
  // Lightweight CLI overrides for rapid solver studies.
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--use-petsc" && i + 1 < argc) {
      cfg.use_petsc = (std::stoi(argv[++i]) != 0);
    } else if (arg == "--t-end" && i + 1 < argc) {
      cfg.t_end_ms = std::stod(argv[++i]);
    } else if (arg == "--dt" && i + 1 < argc) {
      cfg.dt_pde_ms = std::stod(argv[++i]);
    }
  }
}

std::string ToString(const SimulationConfig& cfg) {
  // Startup log is intentionally compact; verbose diagnostics go to CSV files.
  std::ostringstream oss;
  oss << "mesh_path=" << cfg.mesh_path << "\n";
  oss << "use_fiber_gf=" << (cfg.use_fiber_gf ? 1 : 0) << "\n";
  oss << "enable_wholebody=" << (cfg.enable_wholebody ? 1 : 0) << "\n";
  oss << "use_conforming_wholebody=" << (cfg.use_conforming_wholebody ? 1 : 0) << "\n";
  oss << "wholebody_mesh_path=" << cfg.wholebody_mesh_path << "\n";
  oss << "torso_mesh_path=" << cfg.torso_mesh_path << "\n";
  oss << "fiber_f_path=" << cfg.fiber_f_path << "\n";
  oss << "fiber_s_path=" << cfg.fiber_s_path << "\n";
  oss << "fiber_n_path=" << cfg.fiber_n_path << "\n";
  oss << "sigma_i_f_mS_per_mm=" << cfg.sigma_i_f_mS_per_mm << "\n";
  oss << "sigma_i_s_mS_per_mm=" << cfg.sigma_i_s_mS_per_mm << "\n";
  oss << "sigma_i_n_mS_per_mm=" << cfg.sigma_i_n_mS_per_mm << "\n";
  oss << "sigma_e_f_mS_per_mm=" << cfg.sigma_e_f_mS_per_mm << "\n";
  oss << "sigma_e_s_mS_per_mm=" << cfg.sigma_e_s_mS_per_mm << "\n";
  oss << "sigma_e_n_mS_per_mm=" << cfg.sigma_e_n_mS_per_mm << "\n";
  oss << "sigma_torso_mS_per_mm=" << cfg.sigma_torso_mS_per_mm << "\n";
  oss << "interface_map_max_dist_mm=" << cfg.interface_map_max_dist_mm << "\n";
  oss << "torso_dirichlet_penalty=" << cfg.torso_dirichlet_penalty << "\n";
  oss << "heart_volume_attrs=" << cfg.heart_volume_attrs.size() << "\n";
  oss << "torso_volume_attrs=" << cfg.torso_volume_attrs.size() << "\n";
  oss << "heart_interface_bdr_attrs=" << cfg.heart_interface_bdr_attrs.size() << "\n";
  oss << "torso_interface_bdr_attrs=" << cfg.torso_interface_bdr_attrs.size() << "\n";
  oss << "dt_pde_ms=" << cfg.dt_pde_ms << "\n";
  oss << "dt_ode_ms=" << cfg.dt_ode_ms << "\n";
  oss << "t_end_ms=" << cfg.t_end_ms << "\n";
  oss << "stim_xmin_mm=" << cfg.stim_xmin_mm << "\n";
  oss << "stim_xmax_mm=" << cfg.stim_xmax_mm << "\n";
  oss << "stim_ymin_mm=" << cfg.stim_ymin_mm << "\n";
  oss << "stim_ymax_mm=" << cfg.stim_ymax_mm << "\n";
  oss << "stim_zmin_mm=" << cfg.stim_zmin_mm << "\n";
  oss << "stim_zmax_mm=" << cfg.stim_zmax_mm << "\n";
  oss << "stim_regions=" << cfg.stim_regions.size() << "\n";
  oss << "use_petsc=" << (cfg.use_petsc ? 1 : 0) << "\n";
  oss << "use_hypre_boomeramg=" << (cfg.use_hypre_boomeramg ? 1 : 0) << "\n";
  oss << "wholebody_solve_every_step=" << (cfg.wholebody_solve_every_step ? 1 : 0) << "\n";
  oss << "ksp_max_it=" << cfg.ksp_max_it << "\n";
  oss << "ksp_rtol=" << cfg.ksp_rtol << "\n";
  oss << "petsc_use_geometric_asm=" << (cfg.petsc_use_geometric_asm ? 1 : 0) << "\n";
  oss << "petsc_asm_nx=" << cfg.petsc_asm_nx << "\n";
  oss << "petsc_asm_ny=" << cfg.petsc_asm_ny << "\n";
  oss << "petsc_asm_nz=" << cfg.petsc_asm_nz << "\n";
  return oss.str();
}

}  // namespace mono
