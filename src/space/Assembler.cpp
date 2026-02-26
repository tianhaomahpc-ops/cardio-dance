#include "space/Assembler.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mono {

Assembler::Assembler(const SimulationConfig& cfg, MPI_Comm comm) : cfg_(cfg), comm_(comm) {
  serial_mesh_ = std::make_unique<mfem::Mesh>(cfg_.mesh_path.c_str(), 1, 1);
  pmesh_ = std::make_unique<mfem::ParMesh>(comm_, *serial_mesh_);

  const int dim = pmesh_->Dimension();

  fec_ = std::make_unique<mfem::H1_FECollection>(1, dim);
  pfes_ = std::make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), fec_.get());

  vm_ = std::make_unique<mfem::ParGridFunction>(pfes_.get());
  *vm_ = -85.23;

  // Conductivity tensor coefficient (constant or loaded fibers).
  InitializeFiberCoefficients(dim);

  // Assemble mass matrix M.
  m_form_ = std::make_unique<mfem::ParBilinearForm>(pfes_.get());
  m_form_->AddDomainIntegrator(new mfem::MassIntegrator());
  m_form_->Assemble();
  m_form_->Finalize();
  M_.reset(m_form_->ParallelAssemble());

  // Assemble diffusion matrix K with anisotropic conductivity.
  k_form_ = std::make_unique<mfem::ParBilinearForm>(pfes_.get());
  k_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*d_coeff_));
  k_form_->Assemble();
  k_form_->Finalize();
  K_.reset(k_form_->ParallelAssemble());

  BuildSystemMatrices(cfg_.dt_pde_ms);
}

Assembler::Assembler(const SimulationConfig& cfg,
                     MPI_Comm comm,
                     std::unique_ptr<mfem::ParMesh> pmesh_override)
    : cfg_(cfg), comm_(comm) {
  if (!pmesh_override) {
    throw std::runtime_error("Assembler pmesh_override cannot be null");
  }
  pmesh_ = std::move(pmesh_override);

  const int dim = pmesh_->Dimension();

  fec_ = std::make_unique<mfem::H1_FECollection>(1, dim);
  pfes_ = std::make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), fec_.get());

  vm_ = std::make_unique<mfem::ParGridFunction>(pfes_.get());
  *vm_ = -85.23;

  InitializeFiberCoefficients(dim);

  m_form_ = std::make_unique<mfem::ParBilinearForm>(pfes_.get());
  m_form_->AddDomainIntegrator(new mfem::MassIntegrator());
  m_form_->Assemble();
  m_form_->Finalize();
  M_.reset(m_form_->ParallelAssemble());

  k_form_ = std::make_unique<mfem::ParBilinearForm>(pfes_.get());
  k_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*d_coeff_));
  k_form_->Assemble();
  k_form_->Finalize();
  K_.reset(k_form_->ParallelAssemble());

  BuildSystemMatrices(cfg_.dt_pde_ms);
}

void Assembler::InitializeFiberCoefficients(int dim) {
  mfem::Vector f(dim), s(dim), n(dim);
  f = 0.0;
  s = 0.0;
  n = 0.0;
  if (dim > 0) f[0] = 1.0;
  if (dim > 1) s[1] = 1.0;
  if (dim > 2) n[2] = 1.0;

  if (!cfg_.use_fiber_gf) {
    // Default orthonormal basis for homogeneous benchmark cases.
    const_f_coeff_ = std::make_unique<mfem::VectorConstantCoefficient>(f);
    const_s_coeff_ = std::make_unique<mfem::VectorConstantCoefficient>(s);
    const_n_coeff_ = std::make_unique<mfem::VectorConstantCoefficient>(n);
    d_coeff_ = std::make_unique<FiberTensorCoefficient>(dim,
                                                         cfg_.sigma_f_mS_per_mm,
                                                         cfg_.sigma_s_mS_per_mm,
                                                         cfg_.sigma_n_mS_per_mm,
                                                         *const_f_coeff_,
                                                         *const_s_coeff_,
                                                         *const_n_coeff_);
    use_loaded_fibers_ = false;
    return;
  }

  fiber_fes_ = std::make_unique<mfem::ParFiniteElementSpace>(
      pmesh_.get(), fec_.get(), dim, mfem::Ordering::byVDIM);

  fiber_f_gf_ = std::make_unique<mfem::ParGridFunction>(fiber_fes_.get());
  fiber_s_gf_ = std::make_unique<mfem::ParGridFunction>(fiber_fes_.get());
  fiber_n_gf_ = std::make_unique<mfem::ParGridFunction>(fiber_fes_.get());

  auto load_fiber_field = [&](const std::string& path, mfem::ParGridFunction& dst, const char* label) {
    std::ifstream fin(path);
    if (!fin) {
      throw std::runtime_error(std::string("Failed to open ") + label + "_path: " + path);
    }

    // Read via source metadata and re-project to local target ordering.
    mfem::ParGridFunction src(pmesh_.get(), fin);
    if (src.ParFESpace()->GetVDim() != dim) {
      throw std::runtime_error(std::string("Invalid ") + label +
                               " vector dimension in file: " + path);
    }

    mfem::VectorGridFunctionCoefficient src_coeff(&src);
    dst.ProjectCoefficient(src_coeff);

    // Guard against broken field files that can inject NaNs.
    for (int i = 0; i < dst.Size(); ++i) {
      if (!std::isfinite(dst(i))) {
        throw std::runtime_error(std::string("Non-finite values detected in ") + label +
                                 " file: " + path);
      }
    }
  };

  load_fiber_field(cfg_.fiber_f_path, *fiber_f_gf_, "fiber_f");
  load_fiber_field(cfg_.fiber_s_path, *fiber_s_gf_, "fiber_s");
  load_fiber_field(cfg_.fiber_n_path, *fiber_n_gf_, "fiber_n");

  fiber_f_coeff_ = std::make_unique<mfem::VectorGridFunctionCoefficient>(fiber_f_gf_.get());
  fiber_s_coeff_ = std::make_unique<mfem::VectorGridFunctionCoefficient>(fiber_s_gf_.get());
  fiber_n_coeff_ = std::make_unique<mfem::VectorGridFunctionCoefficient>(fiber_n_gf_.get());

  d_coeff_ = std::make_unique<FiberTensorCoefficient>(dim,
                                                       cfg_.sigma_f_mS_per_mm,
                                                       cfg_.sigma_s_mS_per_mm,
                                                       cfg_.sigma_n_mS_per_mm,
                                                       *fiber_f_coeff_,
                                                       *fiber_s_coeff_,
                                                       *fiber_n_coeff_);
  use_loaded_fibers_ = true;
}

void Assembler::BuildSystemMatrices(double dt_pde_ms) {
  if (dt_pde_ms <= 0.0) {
    throw std::runtime_error("BuildSystemMatrices requires dt_pde_ms > 0");
  }

  // Split-step linear systems:
  // A = alpha M + 0.5 K,  B = alpha M - 0.5 K, alpha = chi*Cm/dt.
  const double alpha = (cfg_.chi_per_mm * cfg_.cm_uF_per_mm2) / dt_pde_ms;
  A_.reset(mfem::Add(alpha, *M_, 0.5, *K_));
  B_.reset(mfem::Add(alpha, *M_, -0.5, *K_));
}

}  // namespace mono
