#include "io/PseudoEcg.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <stdexcept>

namespace mono {

PseudoEcg::PseudoEcg(MPI_Comm comm,
                     const mfem::ParFiniteElementSpace& pfes,
                     const std::vector<Probe>& probes,
                     double sigma_i_mS_per_mm,
                     double sigma_b_mS_per_mm,
                     const std::string& csv_path)
    : comm_(comm),
      probes_(probes),
      sigma_i_(sigma_i_mS_per_mm),
      sigma_b_(sigma_b_mS_per_mm),
      pfes_(&pfes) {
  MPI_Comm_rank(comm_, &rank_);
  if (probes_.empty()) {
    throw std::runtime_error("PseudoEcg requires at least one probe");
  }
  if (sigma_b_ <= 0.0) {
    throw std::runtime_error("PseudoEcg sigma_b must be positive");
  }

  BuildElementCache(pfes);

  if (rank_ == 0) {
    std::filesystem::create_directories(
        std::filesystem::path(csv_path).parent_path());
    csv_.open(csv_path);
    if (!csv_) {
      throw std::runtime_error("PseudoEcg: cannot open " + csv_path);
    }
    csv_ << "t_ms";
    for (const auto& p : probes_) {
      csv_ << "," << p.name;
    }
    csv_ << "\n";
  }
}

PseudoEcg::~PseudoEcg() {
  if (csv_.is_open()) csv_.close();
}

void PseudoEcg::BuildElementCache(const mfem::ParFiniteElementSpace& pfes) {
  mfem::ParMesh* pmesh = pfes.GetParMesh();
  const int ne = pmesh->GetNE();
  elem_cache_.resize(ne);

  for (int e = 0; e < ne; ++e) {
    mfem::ElementTransformation* T = pmesh->GetElementTransformation(e);
    const mfem::FiniteElement* fe = pfes.GetFE(e);
    const mfem::IntegrationRule& ir =
        mfem::IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + 1);

    double vol = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      const double w = ip.weight * T->Weight();
      vol += w;
      mfem::Vector phys(3);
      T->Transform(ip, phys);
      cx += w * phys[0];
      cy += w * phys[1];
      cz += w * phys[2];
    }
    if (vol > 0.0) {
      cx /= vol;
      cy /= vol;
      cz /= vol;
    }
    elem_cache_[e] = {cx, cy, cz, vol};
  }
}

void PseudoEcg::Sample(double t_ms, const mfem::ParGridFunction& vm_gf) {
  // For each element, compute a single representative gradient at the
  // element centroid (P1 elements have constant gradient, so this is
  // exact for P1; for higher order it's an average).
  const int ne = pfes_->GetParMesh()->GetNE();
  std::vector<double> phi_local(probes_.size(), 0.0);

  mfem::Vector grad(3);
  for (int e = 0; e < ne; ++e) {
    mfem::ElementTransformation* T =
        pfes_->GetParMesh()->GetElementTransformation(e);
    const mfem::FiniteElement* fe = pfes_->GetFE(e);

    // Use the geometric center in reference coords; for P1 tets this
    // yields the (constant) element gradient.
    mfem::IntegrationPoint ip;
    ip.x = ip.y = ip.z = 0.25;
    ip.weight = 1.0;
    T->SetIntPoint(&ip);
    vm_gf.GetGradient(*T, grad);

    const auto& c = elem_cache_[e];
    const double vol = c.volume;

    for (size_t k = 0; k < probes_.size(); ++k) {
      const double rx = probes_[k].x - c.cx;
      const double ry = probes_[k].y - c.cy;
      const double rz = probes_[k].z - c.cz;
      const double r2 = rx * rx + ry * ry + rz * rz;
      if (r2 < 1e-12) continue;  // probe too close to element; skip
      const double r3 = r2 * std::sqrt(r2);
      const double dot = grad[0] * rx + grad[1] * ry + grad[2] * rz;
      phi_local[k] += vol * dot / r3;
      (void)fe;
    }
  }

  std::vector<double> phi_global(probes_.size(), 0.0);
  MPI_Allreduce(phi_local.data(), phi_global.data(),
                static_cast<int>(probes_.size()),
                MPI_DOUBLE, MPI_SUM, comm_);

  const double prefactor = sigma_i_ / (4.0 * M_PI * sigma_b_);

  if (rank_ == 0 && csv_.is_open()) {
    csv_ << std::setprecision(6) << t_ms;
    for (size_t k = 0; k < probes_.size(); ++k) {
      csv_ << "," << (prefactor * phi_global[k]);
    }
    csv_ << "\n";
    csv_.flush();
  }
}

}  // namespace mono
