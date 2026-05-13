#include "ode/Land2017Model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace mono {

Land2017Model::Land2017Model(int n_local_true_dofs, const Params& params)
    : n_nodes_(n_local_true_dofs),
      params_(params),
      states_(static_cast<size_t>(n_local_true_dofs) * kNumStates, 0.0),
      lambda_(static_cast<size_t>(n_local_true_dofs), 1.0) {
  if (n_nodes_ < 0) {
    throw std::runtime_error("Land2017Model requires n_local_true_dofs >= 0");
  }
  InitializeRestState();
}

void Land2017Model::InitializeRestState() {
  for (int node = 0; node < n_nodes_; ++node) {
    double* s = StatePtr(node);
    s[0] = 0.0;     // CaTRPN
    s[1] = 1.0;     // B (all blocked at rest)
    s[2] = 0.0;     // S
    s[3] = 0.0;     // W
    s[4] = 0.0;     // zeta_s
    s[5] = 0.0;     // zeta_w
    s[6] = 0.0;     // Cd
    s[7] = 1.0;     // lam_int (sarcomere at rest stretch)
  }
  std::fill(lambda_.begin(), lambda_.end(), 1.0);
}

void Land2017Model::SetStretch(const mfem::Vector& lambda_true) {
  if (lambda_true.Size() != n_nodes_) {
    throw std::runtime_error("Land2017Model::SetStretch size mismatch");
  }
  for (int node = 0; node < n_nodes_; ++node) {
    lambda_[node] = lambda_true[node];
  }
}

double Land2017Model::LengthFactor(double lam) const {
  // Length-dependent activation factor, after Land 2017 eq (12).
  const double lam_c = std::min(std::max(lam, params_.lambda_min), params_.lambda_max);
  const double h = 1.0 + params_.beta_0 * (lam_c + std::min(lam_c, 1.2) - 1.87);
  return std::max(h, 0.0);
}

void Land2017Model::StepNode(int node, double dt_ms, double cai_uM) {
  double* s = StatePtr(node);
  const double CaTRPN  = s[0];
  const double B       = s[1];
  const double S       = s[2];
  const double W       = s[3];
  const double zeta_s  = s[4];
  const double zeta_w  = s[5];
  const double Cd      = s[6];
  const double lam_int = s[7];

  const double lam = lambda_[node];

  // 1) CaTRPN dynamics: Hill-type binding with rate k_trpn.
  const double ca_ratio = cai_uM / params_.Ca50_uM;
  const double ca_pow   = std::pow(std::max(ca_ratio, 0.0), params_.n_trpn);
  const double dCaTRPN  = params_.k_trpn * (ca_pow * (1.0 - CaTRPN) - CaTRPN);

  // 2) Tropomyosin blocked fraction B (1-B is available for binding).
  // Forward rate scaled by (CaTRPN/TRPN50)^n_tm * phi (cooperative unblock).
  const double trpn_norm = CaTRPN / std::max(params_.TRPN50, 1e-12);
  const double k_unb     = params_.k_tm_unb * params_.phi *
                           std::pow(std::max(trpn_norm, 0.0), params_.n_tm);
  const double k_block   = params_.k_tm_unb;
  const double dB        = -k_unb * B + k_block * (1.0 - B);

  // 3) Crossbridge cycling: U -> W -> S -> U.
  const double U = std::max(1.0 - B - S - W, 0.0);
  const double dW = params_.k_uw * U * (1.0 - B) - params_.k_ws * W - params_.k_su * W;
  const double dS = params_.k_ws * W - params_.k_su * S;

  // 4) Distortion dynamics zeta_s, zeta_w.
  // d(lam_int)/dt = (lam - lam_int) / lam_tau; dot_lam ~ d(lam_int)/dt
  const double dlam_int = (lam - lam_int) / std::max(params_.lam_tau, 1e-6);
  const double dot_lam  = dlam_int;
  const double dzeta_s  = params_.A_eff * dot_lam - params_.gamma_s * zeta_s;
  const double dzeta_w  = params_.A_eff * dot_lam - params_.gamma_w * zeta_w;

  // 5) Slow cooperativity factor Cd (low-pass of CaTRPN^n_tm).
  const double Cd_target = std::pow(std::max(trpn_norm, 0.0), params_.n_tm);
  const double dCd       = (Cd_target - Cd) / std::max(params_.cd_tau, 1e-6);

  // Forward Euler.
  s[0] = CaTRPN  + dt_ms * dCaTRPN;
  s[1] = std::clamp(B + dt_ms * dB, 0.0, 1.0);
  s[2] = std::max(S + dt_ms * dS, 0.0);
  s[3] = std::max(W + dt_ms * dW, 0.0);
  s[4] = zeta_s  + dt_ms * dzeta_s;
  s[5] = zeta_w  + dt_ms * dzeta_w;
  s[6] = std::clamp(Cd + dt_ms * dCd, 0.0, 1.0);
  s[7] = lam_int + dt_ms * dlam_int;

  // Numerical guard: ensure S+W <= 1-B-eps.
  const double bound = std::max(1.0 - s[1], 0.0);
  const double sw_sum = s[2] + s[3];
  if (sw_sum > bound && sw_sum > 0.0) {
    const double scale = bound / sw_sum;
    s[2] *= scale;
    s[3] *= scale;
  }
}

void Land2017Model::Advance(double dt_pde_ms, double dt_ode_ms,
                            const mfem::Vector& cai_true_mM) {
  if (cai_true_mM.Size() != n_nodes_) {
    throw std::runtime_error("Land2017Model::Advance: cai size mismatch");
  }
  const double dt_sub = std::min(std::max(dt_ode_ms, 1e-6), dt_pde_ms);
  const int n_sub = std::max(1, static_cast<int>(std::ceil(dt_pde_ms / dt_sub)));
  const double dt = dt_pde_ms / static_cast<double>(n_sub);

  for (int node = 0; node < n_nodes_; ++node) {
    // Convert mM -> uM.
    const double cai_uM = cai_true_mM[node] * 1000.0;
    for (int k = 0; k < n_sub; ++k) {
      StepNode(node, dt, cai_uM);
    }
  }
}

void Land2017Model::GetTension(mfem::Vector& ta_kPa_true) const {
  ta_kPa_true.SetSize(n_nodes_);
  const double inv_rs = 1.0 / std::max(params_.r_s, 1e-6);
  for (int node = 0; node < n_nodes_; ++node) {
    const double* s = StatePtr(node);
    const double S = s[2];
    const double W = s[3];
    const double zeta_s = s[4];
    const double zeta_w = s[5];
    const double lam = lambda_[node];
    const double h = LengthFactor(lam);
    // T_a = h(lam) * Tref / r_s * (S*(zeta_s+1) + W*zeta_w)
    const double Ta = h * params_.Tref * inv_rs *
                      (S * (zeta_s + 1.0) + W * zeta_w);
    ta_kPa_true[node] = std::max(Ta, 0.0);
  }
}

void Land2017Model::SaveState(std::ostream& os) const {
  const int32_t n = n_nodes_;
  const int32_t ns = kNumStates;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(&ns), sizeof(ns));
  os.write(reinterpret_cast<const char*>(states_.data()),
           static_cast<std::streamsize>(states_.size() * sizeof(double)));
  os.write(reinterpret_cast<const char*>(lambda_.data()),
           static_cast<std::streamsize>(lambda_.size() * sizeof(double)));
}

void Land2017Model::LoadState(std::istream& is) {
  int32_t n = 0, ns = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  is.read(reinterpret_cast<char*>(&ns), sizeof(ns));
  if (n != n_nodes_ || ns != kNumStates) {
    throw std::runtime_error("Land2017Model::LoadState header mismatch");
  }
  is.read(reinterpret_cast<char*>(states_.data()),
          static_cast<std::streamsize>(states_.size() * sizeof(double)));
  is.read(reinterpret_cast<char*>(lambda_.data()),
          static_cast<std::streamsize>(lambda_.size() * sizeof(double)));
  if (!is) {
    throw std::runtime_error("Land2017Model::LoadState read failed");
  }
}

}  // namespace mono
