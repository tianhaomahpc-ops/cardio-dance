#include "mechanics/ElasticitySolver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mono {

namespace {

std::string RankSuffixedPath(const std::string& prefix, int rank) {
  std::ostringstream os;
  os << prefix << "." << std::setfill('0') << std::setw(6) << rank;
  return os.str();
}

}  // namespace

ElasticitySolver::ElasticitySolver(MPI_Comm comm) : comm_(comm) {}

ElasticityResult ElasticitySolver::Solve(const std::string& mesh_path,
                                         const ElasticityOptions& options,
                                         const std::string& output_mesh_prefix,
                                         const std::string& output_disp_prefix) const {
  if (options.order < 1) {
    throw std::runtime_error("elasticity order must be >= 1");
  }
  if (options.young_modulus <= 0.0) {
    throw std::runtime_error("young_modulus must be > 0");
  }
  if (options.poisson_ratio <= -1.0 || options.poisson_ratio >= 0.5) {
    throw std::runtime_error("poisson_ratio must be in (-1, 0.5)");
  }

  mfem::Mesh serial_mesh(mesh_path.c_str(), 1, 1);
  RelabelBoundaryByXAxis(serial_mesh, options.boundary_tolerance, 1, 2);

  mfem::ParMesh pmesh(comm_, serial_mesh);
  const int dim = pmesh.Dimension();

  mfem::H1_FECollection fec(options.order, dim);
  mfem::ParFiniteElementSpace fes(&pmesh, &fec, dim, mfem::Ordering::byVDIM);

  const double mu = options.young_modulus / (2.0 * (1.0 + options.poisson_ratio));
  const double lambda =
      options.young_modulus * options.poisson_ratio /
      ((1.0 + options.poisson_ratio) * (1.0 - 2.0 * options.poisson_ratio));

  mfem::ConstantCoefficient lambda_coeff(lambda);
  mfem::ConstantCoefficient mu_coeff(mu);

  mfem::ParBilinearForm a(&fes);
  a.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda_coeff, mu_coeff));
  a.Assemble();

  const int bdr_attr_max = pmesh.bdr_attributes.Max();
  if (bdr_attr_max < 2) {
    throw std::runtime_error("elasticity mesh must expose at least two boundary attrs after relabel");
  }

  mfem::Array<int> ess_bdr(bdr_attr_max);
  ess_bdr = 0;
  ess_bdr[0] = 1;  // attr 1: fixed

  mfem::Array<int> traction_bdr(bdr_attr_max);
  traction_bdr = 0;
  traction_bdr[1] = 1;  // attr 2: traction

  mfem::Array<int> ess_tdofs;
  fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);

  mfem::Vector traction(dim);
  traction = 0.0;
  traction[0] = options.traction_x;
  if (dim > 1) {
    traction[1] = options.traction_y;
  }
  if (dim > 2) {
    traction[2] = options.traction_z;
  }

  mfem::VectorConstantCoefficient traction_coeff(traction);

  mfem::ParLinearForm b(&fes);
  b.AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(traction_coeff), traction_bdr);
  b.Assemble();

  mfem::ParGridFunction displacement(&fes);
  displacement = 0.0;

  mfem::OperatorPtr A;
  mfem::Vector X, B;
  a.FormLinearSystem(ess_tdofs, displacement, b, A, X, B);

  auto* A_hypre = dynamic_cast<mfem::HypreParMatrix*>(A.Ptr());
  if (A_hypre == nullptr) {
    throw std::runtime_error("elasticity solver expected HypreParMatrix from FormLinearSystem");
  }

  mfem::HypreBoomerAMG amg(*A_hypre);
  amg.SetPrintLevel(0);

  mfem::CGSolver cg(comm_);
  cg.SetOperator(*A_hypre);
  cg.SetPreconditioner(amg);
  cg.SetMaxIter(options.max_iter);
  cg.SetRelTol(options.rel_tol);
  cg.SetAbsTol(options.abs_tol);
  cg.SetPrintLevel(options.print_level);
  cg.Mult(B, X);

  a.RecoverFEMSolution(X, b, displacement);

  mfem::Vector disp_true;
  displacement.GetTrueDofs(disp_true);

  const double local_l2_sq = disp_true * disp_true;
  double global_l2_sq = 0.0;
  MPI_Allreduce(&local_l2_sq, &global_l2_sq, 1, MPI_DOUBLE, MPI_SUM, comm_);

  const double local_inf = disp_true.Normlinf();
  double global_inf = 0.0;
  MPI_Allreduce(&local_inf, &global_inf, 1, MPI_DOUBLE, MPI_MAX, comm_);

  int rank = 0;
  MPI_Comm_rank(comm_, &rank);
  if (!output_mesh_prefix.empty()) {
    std::ofstream mesh_out(RankSuffixedPath(output_mesh_prefix, rank));
    pmesh.Print(mesh_out);
  }
  if (!output_disp_prefix.empty()) {
    std::ofstream disp_out(RankSuffixedPath(output_disp_prefix, rank));
    displacement.Save(disp_out);
  }

  ElasticityResult result;
  result.dim = dim;
  result.num_elements = pmesh.GetNE();
  result.num_vertices = pmesh.GetNV();
  result.num_cg_iterations = cg.GetNumIterations();
  result.final_residual_norm = cg.GetFinalNorm();
  result.displacement_l2_norm = std::sqrt(global_l2_sq);
  result.displacement_max_abs = global_inf;
  return result;
}

void ElasticitySolver::RelabelBoundaryByXAxis(mfem::Mesh& mesh,
                                              double rel_tol,
                                              int fixed_attr,
                                              int traction_attr) {
  if (mesh.GetNBE() <= 0) {
    throw std::runtime_error("mesh has no boundary elements");
  }

  double xmin = std::numeric_limits<double>::infinity();
  double xmax = -std::numeric_limits<double>::infinity();
  for (int vi = 0; vi < mesh.GetNV(); ++vi) {
    const double* v = mesh.GetVertex(vi);
    xmin = std::min(xmin, v[0]);
    xmax = std::max(xmax, v[0]);
  }

  const double span = xmax - xmin;
  if (!(span > 0.0)) {
    throw std::runtime_error("invalid mesh x-range for boundary relabeling");
  }

  const double safe_rel = std::max(std::abs(rel_tol), 1e-12);
  const double tol = safe_rel * span;

  int fixed_count = 0;
  int traction_count = 0;
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    mfem::Element* bdr = mesh.GetBdrElement(be);
    const int* verts = bdr->GetVertices();
    const int nverts = bdr->GetNVertices();

    double x_avg = 0.0;
    for (int j = 0; j < nverts; ++j) {
      x_avg += mesh.GetVertex(verts[j])[0];
    }
    x_avg /= static_cast<double>(nverts);

    if (std::abs(x_avg - xmin) <= tol) {
      bdr->SetAttribute(fixed_attr);
      ++fixed_count;
    } else if (std::abs(x_avg - xmax) <= tol) {
      bdr->SetAttribute(traction_attr);
      ++traction_count;
    } else {
      bdr->SetAttribute(3);
    }
  }

  if (fixed_count == 0 || traction_count == 0) {
    throw std::runtime_error("failed to detect x-min/x-max boundaries for elasticity BCs");
  }

  mesh.SetAttributes();
}

}  // namespace mono
