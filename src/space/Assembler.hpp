#pragma once

#include <memory>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "space/FiberTensorCoefficient.hpp"

namespace mono {

// Builds FE spaces and monodomain system matrices:
//   M mass matrix
//   K anisotropic diffusion matrix
//   A/B split-step system matrices for Crank-Nicolson diffusion update.
class Assembler {
 public:
  Assembler(const SimulationConfig& cfg, MPI_Comm comm);
  Assembler(const SimulationConfig& cfg, MPI_Comm comm, std::unique_ptr<mfem::ParMesh> pmesh_override);

  // Rebuild A,B when dt_pde changes.
  void BuildSystemMatrices(double dt_pde_ms);

  mfem::ParFiniteElementSpace& PFES() { return *pfes_; }
  const mfem::ParFiniteElementSpace& PFES() const { return *pfes_; }
  mfem::ParGridFunction& Vm() { return *vm_; }
  const mfem::ParGridFunction& Vm() const { return *vm_; }

  const mfem::HypreParMatrix& M() const { return *M_; }
  const mfem::HypreParMatrix& K() const { return *K_; }
  const mfem::HypreParMatrix& A() const { return *A_; }
  const mfem::HypreParMatrix& B() const { return *B_; }

  int TrueVSize() const { return pfes_->GetTrueVSize(); }
  bool HasFiberFields() const { return use_loaded_fibers_; }
  mfem::ParGridFunction* FiberF() { return fiber_f_gf_.get(); }
  mfem::ParGridFunction* FiberS() { return fiber_s_gf_.get(); }
  mfem::ParGridFunction* FiberN() { return fiber_n_gf_.get(); }
  const mfem::ParGridFunction* FiberF() const { return fiber_f_gf_.get(); }
  const mfem::ParGridFunction* FiberS() const { return fiber_s_gf_.get(); }
  const mfem::ParGridFunction* FiberN() const { return fiber_n_gf_.get(); }

 private:
  void InitializeFiberCoefficients(int dim);

  const SimulationConfig& cfg_;
  MPI_Comm comm_;

  std::unique_ptr<mfem::Mesh> serial_mesh_;
  std::unique_ptr<mfem::ParMesh> pmesh_;
  std::unique_ptr<mfem::H1_FECollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> pfes_;
  std::unique_ptr<mfem::ParFiniteElementSpace> fiber_fes_;
  std::unique_ptr<mfem::ParGridFunction> vm_;
  std::unique_ptr<mfem::ParGridFunction> fiber_f_gf_;
  std::unique_ptr<mfem::ParGridFunction> fiber_s_gf_;
  std::unique_ptr<mfem::ParGridFunction> fiber_n_gf_;
  bool use_loaded_fibers_ = false;

  std::unique_ptr<mfem::ParBilinearForm> m_form_;
  std::unique_ptr<mfem::ParBilinearForm> k_form_;

  std::unique_ptr<mfem::VectorGridFunctionCoefficient> fiber_f_coeff_;
  std::unique_ptr<mfem::VectorGridFunctionCoefficient> fiber_s_coeff_;
  std::unique_ptr<mfem::VectorGridFunctionCoefficient> fiber_n_coeff_;
  std::unique_ptr<mfem::VectorConstantCoefficient> const_f_coeff_;
  std::unique_ptr<mfem::VectorConstantCoefficient> const_s_coeff_;
  std::unique_ptr<mfem::VectorConstantCoefficient> const_n_coeff_;
  std::unique_ptr<FiberTensorCoefficient> d_coeff_;

  std::unique_ptr<mfem::HypreParMatrix> M_;
  std::unique_ptr<mfem::HypreParMatrix> K_;
  std::unique_ptr<mfem::HypreParMatrix> A_;
  std::unique_ptr<mfem::HypreParMatrix> B_;
};

}  // namespace mono
