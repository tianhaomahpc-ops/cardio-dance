// Idealized prolate-ellipsoidal LV mesh + helical fiber fields.
// Adopts the Land et al. (2015) "Verification of cardiac mechanics software:
// benchmark problems and solutions for testing active and passive material
// behaviour" geometry conventions:
//   * truncated prolate ellipsoid shell
//   * helical fibers, linear interpolation through wall from alpha_endo to alpha_epi
// Output:
//   <out-dir>/lv_ellipsoid.mesh
//   <out-dir>/fiber_f.gf, fiber_s.gf, fiber_n.gf
// Boundary attributes:
//   1 = base ring (truncation plane, top), used as Dirichlet clamp for mechanics
//   2 = endocardial surface (inner)
//   3 = epicardial  surface (outer)

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mfem.hpp"

namespace {

struct LvParams {
  // Land 2015 LV ellipsoid (units = mm).
  double r_short_endo = 7.0;
  double r_long_endo  = 17.0;
  double r_short_epi  = 10.0;
  double r_long_epi   = 20.0;
  // Truncation: base plane at z = z_base_mm; apex at z = -r_long_*.
  double z_base_mm    = 5.0;
  // Topological mesh size.
  int n_phi   = 32;   // circumferential
  int n_mu    = 16;   // longitudinal (base -> apex)
  int n_wall  = 4;    // transmural (endo -> epi)
  // Apex safety: stop short of mu = pi to keep elements non-degenerate.
  double mu_apex_tol_deg = 6.0;  // apex hole half-angle in degrees
  // Fiber helical angle at endo/epi (degrees), linearly interpolated.
  double alpha_endo_deg = 60.0;
  double alpha_epi_deg  = -60.0;
};

void WriteVectorField(const std::string& path, mfem::GridFunction& gf) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot open output file: " + path);
  }
  gf.Save(out);
}

inline int VIdx(int p, int m, int w, int n_phi_unused, int n_mu, int n_wall) {
  // p is already in [0, n_phi-1] (periodic wrap handled by caller).
  (void)n_phi_unused;
  return ((p * (n_mu + 1)) + m) * (n_wall + 1) + w;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  if (mfem::Mpi::WorldSize() != 1) {
    if (mfem::Mpi::WorldRank() == 0) {
      std::cerr << "generate_lv_ellipsoid_case must run with -np 1" << std::endl;
    }
    return 1;
  }

  try {
    LvParams p;
    std::filesystem::path out_dir = "benchmarks/lv_ellipsoid";

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto need_value = [&](const char* key) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error(std::string("Missing value for ") + key);
        }
        return std::string(argv[++i]);
      };
      if (arg == "--out-dir") {
        out_dir = need_value("--out-dir");
      } else if (arg == "--r-short-endo") {
        p.r_short_endo = std::stod(need_value("--r-short-endo"));
      } else if (arg == "--r-long-endo") {
        p.r_long_endo = std::stod(need_value("--r-long-endo"));
      } else if (arg == "--r-short-epi") {
        p.r_short_epi = std::stod(need_value("--r-short-epi"));
      } else if (arg == "--r-long-epi") {
        p.r_long_epi = std::stod(need_value("--r-long-epi"));
      } else if (arg == "--z-base-mm") {
        p.z_base_mm = std::stod(need_value("--z-base-mm"));
      } else if (arg == "--n-phi") {
        p.n_phi = std::stoi(need_value("--n-phi"));
      } else if (arg == "--n-mu") {
        p.n_mu = std::stoi(need_value("--n-mu"));
      } else if (arg == "--n-wall") {
        p.n_wall = std::stoi(need_value("--n-wall"));
      } else if (arg == "--apex-hole-deg") {
        p.mu_apex_tol_deg = std::stod(need_value("--apex-hole-deg"));
      } else if (arg == "--alpha-endo-deg") {
        p.alpha_endo_deg = std::stod(need_value("--alpha-endo-deg"));
      } else if (arg == "--alpha-epi-deg") {
        p.alpha_epi_deg = std::stod(need_value("--alpha-epi-deg"));
      } else {
        throw std::runtime_error("Unknown option: " + arg);
      }
    }

    if (p.n_phi < 4 || p.n_mu < 2 || p.n_wall < 1) {
      throw std::runtime_error("n_phi>=4, n_mu>=2, n_wall>=1 required");
    }
    if (!(p.r_short_endo > 0.0) || !(p.r_short_epi > p.r_short_endo)) {
      throw std::runtime_error("r_short_epi must exceed r_short_endo > 0");
    }
    if (!(p.r_long_endo > 0.0) || !(p.r_long_epi > p.r_long_endo)) {
      throw std::runtime_error("r_long_epi must exceed r_long_endo > 0");
    }
    if (!(p.z_base_mm < p.r_long_endo) || !(p.z_base_mm > -p.r_long_endo)) {
      throw std::runtime_error("z_base_mm must satisfy -r_long_endo < z < r_long_endo");
    }

    std::filesystem::create_directories(out_dir);

    // Parameter ranges.
    // mu in [mu_apex, mu_base] where mu_base is chosen so that the epicardial
    // ring sits at z = z_base. The endocardium is allowed to sit slightly
    // above that height (z_endo = r_long_endo * cos(mu_base)).
    const double mu_apex = M_PI - p.mu_apex_tol_deg * M_PI / 180.0;
    const double mu_base = std::acos(std::clamp(p.z_base_mm / p.r_long_epi, -0.99, 0.99));
    if (mu_apex <= mu_base) {
      throw std::runtime_error("apex angle must exceed base angle (check z_base_mm)");
    }

    const int n_phi = p.n_phi;
    const int n_mu = p.n_mu;
    const int n_wall = p.n_wall;
    const int n_vertices = n_phi * (n_mu + 1) * (n_wall + 1);
    const int n_elements = n_phi * n_mu * n_wall;
    // Boundary quads: base (n_phi * n_wall), endo (n_phi * n_mu), epi (n_phi * n_mu).
    const int n_bdr = n_phi * n_wall + 2 * n_phi * n_mu;

    mfem::Mesh mesh(3 /*dim*/, n_vertices, n_elements, n_bdr, 3 /*spaceDim*/);

    // Build vertices.
    for (int pi = 0; pi < n_phi; ++pi) {
      const double phi = 2.0 * M_PI * static_cast<double>(pi) / static_cast<double>(n_phi);
      for (int mi = 0; mi <= n_mu; ++mi) {
        // m=0 -> base ring; m=n_mu -> apex stop.
        const double s = static_cast<double>(mi) / static_cast<double>(n_mu);
        const double mu = mu_base + s * (mu_apex - mu_base);
        for (int wi = 0; wi <= n_wall; ++wi) {
          const double t = static_cast<double>(wi) / static_cast<double>(n_wall);
          const double rs = p.r_short_endo + t * (p.r_short_epi - p.r_short_endo);
          const double rl = p.r_long_endo  + t * (p.r_long_epi  - p.r_long_endo);
          const double sin_mu = std::sin(mu);
          const double cos_mu = std::cos(mu);
          const double x = rs * sin_mu * std::cos(phi);
          const double y = rs * sin_mu * std::sin(phi);
          const double z = rl * cos_mu;
          double xyz[3] = {x, y, z};
          mesh.AddVertex(xyz);
        }
      }
    }

    auto VID = [&](int pi, int mi, int wi) {
      const int ppi = ((pi % n_phi) + n_phi) % n_phi;
      return VIdx(ppi, mi, wi, n_phi, n_mu, n_wall);
    };

    // Build hex elements (attribute 1 = myocardium).
    for (int pi = 0; pi < n_phi; ++pi) {
      const int ppi = (pi + 1) % n_phi;
      for (int mi = 0; mi < n_mu; ++mi) {
        for (int wi = 0; wi < n_wall; ++wi) {
          // Hex8 connectivity following MFEM MakeCartesian3D convention
          // with local axes (mu, phi, w). With mu growing toward the apex
          // (z decreasing) and w growing toward the epicardium, the resulting
          // physical Jacobian is positive.
          const int v0 = VID(pi,  mi,     wi);
          const int v1 = VID(pi,  mi + 1, wi);
          const int v2 = VID(ppi, mi + 1, wi);
          const int v3 = VID(ppi, mi,     wi);
          const int v4 = VID(pi,  mi,     wi + 1);
          const int v5 = VID(pi,  mi + 1, wi + 1);
          const int v6 = VID(ppi, mi + 1, wi + 1);
          const int v7 = VID(ppi, mi,     wi + 1);
          mesh.AddHex(v0, v1, v2, v3, v4, v5, v6, v7, /*attr*/ 1);
        }
      }
    }

    // Boundary quads.
    // Base ring (mi = 0): attr = 1.
    for (int pi = 0; pi < n_phi; ++pi) {
      const int ppi = (pi + 1) % n_phi;
      for (int wi = 0; wi < n_wall; ++wi) {
        const int v0 = VID(pi,  0, wi);
        const int v1 = VID(ppi, 0, wi);
        const int v2 = VID(ppi, 0, wi + 1);
        const int v3 = VID(pi,  0, wi + 1);
        mesh.AddBdrQuad(v0, v1, v2, v3, /*attr*/ 1);
      }
    }
    // Endo (wi = 0): attr = 2.
    for (int pi = 0; pi < n_phi; ++pi) {
      const int ppi = (pi + 1) % n_phi;
      for (int mi = 0; mi < n_mu; ++mi) {
        // Orient so that normal points inward (toward cavity).
        const int v0 = VID(pi,  mi,     0);
        const int v1 = VID(pi,  mi + 1, 0);
        const int v2 = VID(ppi, mi + 1, 0);
        const int v3 = VID(ppi, mi,     0);
        mesh.AddBdrQuad(v0, v1, v2, v3, /*attr*/ 2);
      }
    }
    // Epi (wi = n_wall): attr = 3.
    for (int pi = 0; pi < n_phi; ++pi) {
      const int ppi = (pi + 1) % n_phi;
      for (int mi = 0; mi < n_mu; ++mi) {
        const int v0 = VID(pi,  mi,     n_wall);
        const int v1 = VID(ppi, mi,     n_wall);
        const int v2 = VID(ppi, mi + 1, n_wall);
        const int v3 = VID(pi,  mi + 1, n_wall);
        mesh.AddBdrQuad(v0, v1, v2, v3, /*attr*/ 3);
      }
    }

    mesh.FinalizeHexMesh(/*generate_edges*/ 1, /*refine*/ 0, /*fix_orientation*/ true);

    {
      const std::filesystem::path mesh_path = out_dir / "lv_ellipsoid.mesh";
      std::ofstream mesh_out(mesh_path);
      if (!mesh_out) {
        throw std::runtime_error("Cannot write mesh file: " + mesh_path.string());
      }
      mesh.Print(mesh_out);
    }

    // Helical fiber fields on a vector H1 space (vdim=3, byVDIM).
    const int dim = mesh.Dimension();
    mfem::H1_FECollection fec(1, dim);
    mfem::FiniteElementSpace vfes(&mesh, &fec, dim, mfem::Ordering::byVDIM);
    mfem::GridFunction fiber_f(&vfes);
    mfem::GridFunction fiber_s(&vfes);
    mfem::GridFunction fiber_n(&vfes);
    fiber_f = 0.0;
    fiber_s = 0.0;
    fiber_n = 0.0;

    // P1 nodal evaluation: assign fibers per vertex using the same parametric
    // map used to build vertices. Indexing matches VIdx.
    const double alpha_endo = p.alpha_endo_deg * M_PI / 180.0;
    const double alpha_epi  = p.alpha_epi_deg  * M_PI / 180.0;
    auto safe_normalize = [](mfem::Vector& v) {
      const double n = v.Norml2();
      if (n > 1e-14) {
        v /= n;
      } else {
        v = 0.0;
        v[0] = 1.0;
      }
    };

    for (int pi = 0; pi < n_phi; ++pi) {
      const double phi = 2.0 * M_PI * static_cast<double>(pi) / static_cast<double>(n_phi);
      for (int mi = 0; mi <= n_mu; ++mi) {
        const double s = static_cast<double>(mi) / static_cast<double>(n_mu);
        const double mu = mu_base + s * (mu_apex - mu_base);
        for (int wi = 0; wi <= n_wall; ++wi) {
          const double t = static_cast<double>(wi) / static_cast<double>(n_wall);
          const double rs = p.r_short_endo + t * (p.r_short_epi - p.r_short_endo);
          const double rl = p.r_long_endo  + t * (p.r_long_epi  - p.r_long_endo);
          const double sin_mu = std::sin(mu);
          const double cos_mu = std::cos(mu);
          const double x = rs * sin_mu * std::cos(phi);
          const double y = rs * sin_mu * std::sin(phi);
          const double z = rl * cos_mu;

          // Outward normal: gradient of (x/rs)^2 + (y/rs)^2 + (z/rl)^2.
          mfem::Vector e_n(3);
          e_n[0] = 2.0 * x / (rs * rs);
          e_n[1] = 2.0 * y / (rs * rs);
          e_n[2] = 2.0 * z / (rl * rl);
          safe_normalize(e_n);

          // Circumferential tangent.
          mfem::Vector e_phi(3);
          e_phi[0] = -std::sin(phi);
          e_phi[1] =  std::cos(phi);
          e_phi[2] =  0.0;
          // Project out any component along e_n for orthogonality.
          double dot = e_phi[0]*e_n[0] + e_phi[1]*e_n[1] + e_phi[2]*e_n[2];
          for (int k = 0; k < 3; ++k) e_phi[k] -= dot * e_n[k];
          safe_normalize(e_phi);

          // Longitudinal tangent (toward apex).
          mfem::Vector e_long(3);
          e_long[0] = e_n[1] * e_phi[2] - e_n[2] * e_phi[1];
          e_long[1] = e_n[2] * e_phi[0] - e_n[0] * e_phi[2];
          e_long[2] = e_n[0] * e_phi[1] - e_n[1] * e_phi[0];
          safe_normalize(e_long);

          const double alpha = alpha_endo + t * (alpha_epi - alpha_endo);
          mfem::Vector f(3), s_dir(3), n_vec(3);
          for (int k = 0; k < 3; ++k) {
            f[k]     =  std::cos(alpha) * e_phi[k] - std::sin(alpha) * e_long[k];
            s_dir[k] =  e_n[k];
          }
          n_vec[0] = s_dir[1] * f[2] - s_dir[2] * f[1];
          n_vec[1] = s_dir[2] * f[0] - s_dir[0] * f[2];
          n_vec[2] = s_dir[0] * f[1] - s_dir[1] * f[0];

          const int vid = VIdx(pi, mi, wi, n_phi, n_mu, n_wall);
          // byVDIM ordering: vdim block at offset = vid * dim.
          for (int k = 0; k < dim; ++k) {
            fiber_f[vid * dim + k] = f[k];
            fiber_s[vid * dim + k] = s_dir[k];
            fiber_n[vid * dim + k] = n_vec[k];
          }
        }
      }
    }

    WriteVectorField((out_dir / "fiber_f.gf").string(), fiber_f);
    WriteVectorField((out_dir / "fiber_s.gf").string(), fiber_s);
    WriteVectorField((out_dir / "fiber_n.gf").string(), fiber_n);

    if (mfem::Mpi::WorldRank() == 0) {
      std::cout << "Generated LV ellipsoid case in: " << out_dir << std::endl;
      std::cout << "  vertices=" << mesh.GetNV() << ", hex=" << mesh.GetNE()
                << ", bdr quads=" << mesh.GetNBE() << std::endl;
      std::cout << "  mu_base(deg)=" << mu_base * 180.0 / M_PI
                << ", mu_apex(deg)=" << mu_apex * 180.0 / M_PI << std::endl;
      std::cout << "  alpha_endo(deg)=" << p.alpha_endo_deg
                << ", alpha_epi(deg)=" << p.alpha_epi_deg << std::endl;
    }
  } catch (const std::exception& ex) {
    std::cerr << "generate_lv_ellipsoid_case failed: " << ex.what() << std::endl;
    return 2;
  }
  return 0;
}
