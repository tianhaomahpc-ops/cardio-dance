#include "mechanics/CalciumDrivenElasticitySolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mono {

CalciumDrivenElasticitySolver::CalciumDrivenElasticitySolver(
    MPI_Comm comm,
    const mfem::ParMesh& reference_mesh,
    int expected_cai_true_size,
    const CalciumDrivenElasticityOptions& options)
    : comm_(comm), options_(options), expected_cai_true_size_(expected_cai_true_size) {
  if (expected_cai_true_size_ <= 0) {
    throw std::runtime_error("CalciumDrivenElasticitySolver requires expected_cai_true_size > 0");
  }
  if (options_.ca_half_mM <= 0.0) {
    throw std::runtime_error("CalciumDrivenElasticitySolver requires ca_half_mM > 0");
  }
  if (options_.ca_hill <= 0.0) {
    throw std::runtime_error("CalciumDrivenElasticitySolver requires ca_hill > 0");
  }
  if (options_.elasticity.order < 1) {
    throw std::runtime_error("CalciumDrivenElasticitySolver requires elasticity.order >= 1");
  }
  if (options_.elasticity.young_modulus <= 0.0) {
    throw std::runtime_error("CalciumDrivenElasticitySolver requires young_modulus > 0");
  }
  if (options_.elasticity.poisson_ratio <= -1.0 || options_.elasticity.poisson_ratio >= 0.5) {
    throw std::runtime_error("CalciumDrivenElasticitySolver requires poisson_ratio in (-1, 0.5)");
  }

  pmesh_ = std::make_unique<mfem::ParMesh>(reference_mesh);
  RelabelBoundaryByXAxis(*pmesh_, options_.elasticity.boundary_tolerance, 1, 2);

  const int dim = pmesh_->Dimension();
  fec_ = std::make_unique<mfem::H1_FECollection>(options_.elasticity.order, dim);
  fes_ = std::make_unique<mfem::ParFiniteElementSpace>(
      pmesh_.get(), fec_.get(), dim, mfem::Ordering::byVDIM);
  displacement_ = std::make_unique<mfem::ParGridFunction>(fes_.get());
  *displacement_ = 0.0;

  const double mu = options_.elasticity.young_modulus /
                    (2.0 * (1.0 + options_.elasticity.poisson_ratio));
  const double lambda = options_.elasticity.young_modulus * options_.elasticity.poisson_ratio /
                        ((1.0 + options_.elasticity.poisson_ratio) *
                         (1.0 - 2.0 * options_.elasticity.poisson_ratio));

  mfem::ConstantCoefficient lambda_coeff(lambda);
  mfem::ConstantCoefficient mu_coeff(mu);
  a_form_ = std::make_unique<mfem::ParBilinearForm>(fes_.get());
  a_form_->AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda_coeff, mu_coeff));
  a_form_->Assemble();
  a_form_->Finalize();

  int bdr_attr_max = 0;
  for (int be = 0; be < pmesh_->GetNBE(); ++be) {
    bdr_attr_max = std::max(bdr_attr_max, pmesh_->GetBdrAttribute(be));
  }
  if (bdr_attr_max < 2) {
    throw std::runtime_error("CalciumDrivenElasticitySolver requires at least two boundary attrs");
  }

  mfem::Array<int> ess_bdr(bdr_attr_max);
  ess_bdr = 0;
  ess_bdr[0] = 1; // attr=1 fixed
  fes_->GetEssentialTrueDofs(ess_bdr, ess_tdofs_);

  traction_bdr_.SetSize(bdr_attr_max);
  traction_bdr_ = 0;
  traction_bdr_[1] = 1; // attr=2 traction
}

CalciumDrivenElasticityStats CalciumDrivenElasticitySolver::SolveFromCytosolicCalcium(
    const mfem::Vector& cai_true) {
  if (cai_true.Size() != expected_cai_true_size_) {
    throw std::runtime_error("SolveFromCytosolicCalcium: cai_true size mismatch");
  }

  double local_sum = 0.0;
  long long local_count = 0;
  for (int i = 0; i < cai_true.Size(); ++i) {
    const double cai = std::isfinite(cai_true[i]) ? std::max(cai_true[i], 0.0) : 0.0;
    local_sum += cai;
    ++local_count;
  }

  double global_sum = 0.0;
  long long global_count = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm_);
  MPI_Allreduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, comm_);
  const double ca_mean = (global_count > 0) ? (global_sum / static_cast<double>(global_count)) : 0.0;

  const double ca = std::max(ca_mean, 0.0);
  const double ca_half = options_.ca_half_mM;
  const double hill = options_.ca_hill;
  const double ca_pow = std::pow(ca, hill);
  const double half_pow = std::pow(ca_half, hill);
  const double denom = ca_pow + half_pow;
  const double activation = (denom > 0.0) ? std::clamp(ca_pow / denom, 0.0, 1.0) : 0.0;

  const int dim = pmesh_->Dimension();
  mfem::Vector traction(dim);
  traction = 0.0;
  traction[0] = options_.elasticity.traction_x * activation;
  if (dim > 1) {
    traction[1] = options_.elasticity.traction_y * activation;
  }
  if (dim > 2) {
    traction[2] = options_.elasticity.traction_z * activation;
  }

  mfem::VectorConstantCoefficient traction_coeff(traction);
  mfem::ParLinearForm b(fes_.get());
  b.AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(traction_coeff), traction_bdr_);
  b.Assemble();

  mfem::OperatorPtr A;
  mfem::Vector X, B;
  a_form_->FormLinearSystem(ess_tdofs_, *displacement_, b, A, X, B);
  auto* A_hypre = dynamic_cast<mfem::HypreParMatrix*>(A.Ptr());
  if (A_hypre == nullptr) {
    throw std::runtime_error("CalciumDrivenElasticitySolver expected HypreParMatrix");
  }

  mfem::HypreBoomerAMG amg(*A_hypre);
  amg.SetPrintLevel(0);
  mfem::CGSolver cg(comm_);
  cg.SetOperator(*A_hypre);
  cg.SetPreconditioner(amg);
  cg.SetMaxIter(options_.elasticity.max_iter);
  cg.SetRelTol(options_.elasticity.rel_tol);
  cg.SetAbsTol(options_.elasticity.abs_tol);
  cg.SetPrintLevel(options_.elasticity.print_level);
  cg.Mult(B, X);

  a_form_->RecoverFEMSolution(X, b, *displacement_);

  mfem::Vector disp_true;
  displacement_->GetTrueDofs(disp_true);
  const double local_l2_sq = disp_true * disp_true;
  double global_l2_sq = 0.0;
  MPI_Allreduce(&local_l2_sq, &global_l2_sq, 1, MPI_DOUBLE, MPI_SUM, comm_);

  const double local_inf = disp_true.Normlinf();
  double global_inf = 0.0;
  MPI_Allreduce(&local_inf, &global_inf, 1, MPI_DOUBLE, MPI_MAX, comm_);

  CalciumDrivenElasticityStats stats;
  stats.num_cg_iterations = cg.GetNumIterations();
  stats.final_residual_norm = cg.GetFinalNorm();
  stats.displacement_l2_norm = std::sqrt(global_l2_sq);
  stats.displacement_max_abs = global_inf;
  stats.calcium_mean_mM = ca_mean;
  stats.activation = activation;
  stats.traction_x = traction[0];
  stats.traction_y = (dim > 1) ? traction[1] : 0.0;
  stats.traction_z = (dim > 2) ? traction[2] : 0.0;
  return stats;
}

void CalciumDrivenElasticitySolver::RelabelBoundaryByXAxis(mfem::ParMesh& mesh,
                                                           double rel_tol,
                                                           int fixed_attr,
                                                           int traction_attr) {
  if (mesh.GetNBE() <= 0) {
    throw std::runtime_error("CalciumDrivenElasticitySolver mesh has no boundary elements");
  }

  double x_min_local = std::numeric_limits<double>::infinity();
  double x_max_local = -std::numeric_limits<double>::infinity();
  for (int vi = 0; vi < mesh.GetNV(); ++vi) {
    const double* v = mesh.GetVertex(vi);
    x_min_local = std::min(x_min_local, v[0]);
    x_max_local = std::max(x_max_local, v[0]);
  }

  double x_min = 0.0;
  double x_max = 0.0;
  MPI_Allreduce(&x_min_local, &x_min, 1, MPI_DOUBLE, MPI_MIN, mesh.GetComm());
  MPI_Allreduce(&x_max_local, &x_max, 1, MPI_DOUBLE, MPI_MAX, mesh.GetComm());

  const double span = x_max - x_min;
  if (!(span > 0.0)) {
    throw std::runtime_error("CalciumDrivenElasticitySolver invalid x-span");
  }
  const double tol = std::max(std::abs(rel_tol), 1e-12) * span;

  int fixed_count_local = 0;
  int traction_count_local = 0;
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    mfem::Element* bdr = mesh.GetBdrElement(be);
    const int* verts = bdr->GetVertices();
    const int nverts = bdr->GetNVertices();
    if (nverts <= 0) {
      continue;
    }
    double x_avg = 0.0;
    for (int j = 0; j < nverts; ++j) {
      x_avg += mesh.GetVertex(verts[j])[0];
    }
    x_avg /= static_cast<double>(nverts);

    if (std::abs(x_avg - x_min) <= tol) {
      bdr->SetAttribute(fixed_attr);
      ++fixed_count_local;
    } else if (std::abs(x_avg - x_max) <= tol) {
      bdr->SetAttribute(traction_attr);
      ++traction_count_local;
    } else {
      bdr->SetAttribute(3);
    }
  }

  int fixed_count_global = 0;
  int traction_count_global = 0;
  MPI_Allreduce(&fixed_count_local, &fixed_count_global, 1, MPI_INT, MPI_SUM, mesh.GetComm());
  MPI_Allreduce(&traction_count_local, &traction_count_global, 1, MPI_INT, MPI_SUM, mesh.GetComm());
  if (fixed_count_global <= 0 || traction_count_global <= 0) {
    throw std::runtime_error(
        "CalciumDrivenElasticitySolver failed to detect x-min/x-max boundaries");
  }
}

}  // namespace mono

