#include "solver/LinearSolverFactory.hpp"

#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MFEM_USE_PETSC
#include <petscksp.h>
#endif

namespace mono {

#ifdef MFEM_USE_PETSC
namespace {

// 将 PETSc 错误码统一转成 C++ 异常，便于上层按同一方式处理错误。
void CheckPetscError(PetscErrorCode ierr, const char* where) {
  if (ierr == 0) {
    return;
  }
  std::ostringstream oss;
  oss << where << " failed with PETSc error code " << ierr;
  throw std::runtime_error(oss.str());
}

PetscInt ToPetscInt(HYPRE_BigInt value, const char* what) {
  const long long v = static_cast<long long>(value);
  const long long lo = static_cast<long long>(std::numeric_limits<PetscInt>::min());
  const long long hi = static_cast<long long>(std::numeric_limits<PetscInt>::max());
  if (v < lo || v > hi) {
    std::ostringstream oss;
    oss << what << " out of PetscInt range: " << v;
    throw std::runtime_error(oss.str());
  }
  return static_cast<PetscInt>(value);
}

// 直接用 Hypre 的 diag/offd 稀疏块构造 PETSc MPIAIJ。
// 这样可避免 MatComputeOperator() 的额外开销，且对 ASM 更友好。
std::unique_ptr<mfem::PetscParMatrix> BuildPetscAijFromHypre(const mfem::HypreParMatrix& A) {
  mfem::SparseMatrix diag;
  mfem::SparseMatrix offd;
  HYPRE_BigInt* cmap = nullptr;
  A.GetDiag(diag);
  A.GetOffd(offd, cmap);

  const int local_rows = A.GetNumRows();
  const int local_cols = A.GetNumCols();
  const HYPRE_BigInt* row_starts = A.GetRowStarts();
  const HYPRE_BigInt* col_starts = A.GetColStarts();

  const PetscInt row_start = ToPetscInt(row_starts[0], "row_starts[0]");
  const PetscInt col_start = ToPetscInt(col_starts[0], "col_starts[0]");
  const PetscInt global_rows = ToPetscInt(A.GetGlobalNumRows(), "global rows");
  const PetscInt global_cols = ToPetscInt(A.GetGlobalNumCols(), "global cols");

  const auto m_local = static_cast<PetscInt>(local_rows);
  const auto n_local = static_cast<PetscInt>(local_cols);

  const int* d_i = diag.HostReadI();
  const int* d_j = diag.HostReadJ();
  const mfem::real_t* d_data = diag.HostReadData();
  const int* o_i = offd.HostReadI();
  const int* o_j = offd.HostReadJ();
  const mfem::real_t* o_data = offd.HostReadData();
  const int offd_width = offd.Width();

  std::vector<PetscInt> d_nnz(local_rows, 0);
  std::vector<PetscInt> o_nnz(local_rows, 0);
  for (int r = 0; r < local_rows; ++r) {
    d_nnz[r] = static_cast<PetscInt>(d_i[r + 1] - d_i[r]);
    o_nnz[r] = static_cast<PetscInt>(o_i[r + 1] - o_i[r]);
  }

  mfem::petsc::Mat mat = nullptr;
  try {
    CheckPetscError(MatCreateAIJ(A.GetComm(),
                                 m_local,
                                 n_local,
                                 global_rows,
                                 global_cols,
                                 0,
                                 d_nnz.data(),
                                 0,
                                 o_nnz.data(),
                                 &mat),
                    "MatCreateAIJ");

    std::vector<PetscInt> cols;
    std::vector<PetscScalar> vals;
    for (int r = 0; r < local_rows; ++r) {
      cols.clear();
      vals.clear();
      cols.reserve(static_cast<size_t>(d_nnz[r] + o_nnz[r]));
      vals.reserve(static_cast<size_t>(d_nnz[r] + o_nnz[r]));

      for (int p = d_i[r]; p < d_i[r + 1]; ++p) {
        cols.push_back(col_start + static_cast<PetscInt>(d_j[p]));
        vals.push_back(static_cast<PetscScalar>(d_data[p]));
      }

      for (int p = o_i[r]; p < o_i[r + 1]; ++p) {
        if (cmap == nullptr) {
          throw std::runtime_error("Hypre offd block has entries but col_map is null");
        }
        const int k = o_j[p];
        if (k < 0 || k >= offd_width) {
          std::ostringstream oss;
          oss << "Invalid offd column index " << k << ", width=" << offd_width;
          throw std::runtime_error(oss.str());
        }
        cols.push_back(ToPetscInt(cmap[k], "offd cmap"));
        vals.push_back(static_cast<PetscScalar>(o_data[p]));
      }

      if (!cols.empty()) {
        const PetscInt row = row_start + static_cast<PetscInt>(r);
        const PetscInt nnz = static_cast<PetscInt>(cols.size());
        CheckPetscError(MatSetValues(mat, 1, &row, nnz, cols.data(), vals.data(), INSERT_VALUES),
                        "MatSetValues");
      }
    }

    CheckPetscError(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY), "MatAssemblyBegin");
    CheckPetscError(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY), "MatAssemblyEnd");

    return std::make_unique<mfem::PetscParMatrix>(mat, false);
  } catch (...) {
    if (mat != nullptr) {
      MatDestroy(&mat);
    }
    throw;
  }
}

void DestroyISVector(std::vector<IS>& is_vec) {
  for (auto& is : is_vec) {
    if (is != nullptr) {
      ISDestroy(&is);
    }
  }
  is_vec.clear();
}

void ConfigurePartitionSubdomains(std::vector<IS>& is_store, mfem::petsc::KSP ksp) {
  mfem::petsc::PC pc = nullptr;
  CheckPetscError(KSPGetPC(ksp, &pc), "KSPGetPC");

  const char* pc_type = nullptr;
  CheckPetscError(PCGetType(pc, &pc_type), "PCGetType");
  if (pc_type == nullptr) {
    return;
  }
  // 仅在 ASM / GASM 下显式设置子域；其它 PC 类型直接返回。
  const bool is_asm = (std::strcmp(pc_type, PCASM) == 0);
  const bool is_gasm = (std::strcmp(pc_type, PCGASM) == 0);
  if (!is_asm && !is_gasm) {
    return;
  }

  mfem::petsc::Mat mat = nullptr;
  CheckPetscError(KSPGetOperators(ksp, &mat, nullptr), "KSPGetOperators");
  // 使用矩阵 ownership range 构造本 rank 子域：
  // 每个 MPI rank 一个连续行区间子域。
  PetscInt row_begin = 0;
  PetscInt row_end = 0;
  CheckPetscError(MatGetOwnershipRange(mat, &row_begin, &row_end), "MatGetOwnershipRange");
  const PetscInt n_local_rows = row_end - row_begin;
  if (n_local_rows <= 0) {
    return;
  }

  DestroyISVector(is_store);
  IS is = nullptr;
  CheckPetscError(ISCreateStride(PETSC_COMM_SELF, n_local_rows, row_begin, 1, &is),
                  "ISCreateStride");
  is_store.push_back(is);
  const PetscInt n_sub = 1;
  if (is_asm) {
    CheckPetscError(PCASMSetLocalSubdomains(pc, n_sub, is_store.data(), is_store.data()),
                    "PCASMSetLocalSubdomains");
  } else {
    CheckPetscError(PCGASMSetSubdomains(pc, n_sub, is_store.data(), is_store.data()),
                    "PCGASMSetSubdomains");
  }
}

}  // namespace
#endif

LinearSystemSolver::LinearSystemSolver(const SimulationConfig& cfg,
                                       MPI_Comm comm,
                                       const mfem::ParFiniteElementSpace* pfes)
    : use_petsc_(cfg.use_petsc),
      comm_(comm),
      pfes_(pfes),
      max_it_(cfg.ksp_max_it),
      rtol_(cfg.ksp_rtol),
      use_hypre_boomeramg_(cfg.use_hypre_boomeramg),
      petsc_use_geometric_asm_(cfg.petsc_use_geometric_asm),
      petsc_asm_nx_(cfg.petsc_asm_nx),
      petsc_asm_ny_(cfg.petsc_asm_ny),
      petsc_asm_nz_(cfg.petsc_asm_nz) {
#ifdef MFEM_USE_PETSC
  if (use_petsc_) {
    // PETSc 路径下，延迟到 SetOperator() 再创建 KSP 与 AIJ 矩阵。
    return;
  }
#else
  if (use_petsc_) {
    throw std::runtime_error("Config asks use_petsc=1 but MFEM was built without PETSc support.");
  }
#endif

  // 非 PETSc 路径：默认使用 MFEM CG。
  cg_ = std::make_unique<mfem::CGSolver>(comm_);
  // 热启动：保留传入 x 作为初值，而不是每步强制从 0 开始。
  cg_->iterative_mode = true;
  cg_->SetRelTol(rtol_);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(max_it_);
  cg_->SetPrintLevel(0);

  if (use_hypre_boomeramg_) {
    // BoomerAMG 需要绑定具体算子 A，故在 SetOperator(A) 中再创建。
  }
}

LinearSystemSolver::~LinearSystemSolver() {
#ifdef MFEM_USE_PETSC
  DestroyISVector(petsc_asm_subdomains_);
#endif
}

void LinearSystemSolver::InvalidatePetscOperator() {
#ifdef MFEM_USE_PETSC
  petsc_A_.reset();
#endif
}

void LinearSystemSolver::SetOperator(const mfem::HypreParMatrix& A) {
#ifdef MFEM_USE_PETSC
  if (use_petsc_) {
    if (!petsc_ksp_) {
      // iter_mode=true：PETSc 也采用热启动初值。
      petsc_ksp_ = std::make_unique<mfem::PetscLinearSolver>(comm_, "mono_", false, true);
      petsc_ksp_->SetRelTol(rtol_);
      petsc_ksp_->SetAbsTol(0.0);
      petsc_ksp_->SetMaxIter(max_it_);
      petsc_ksp_->SetPrintLevel(0);
    }

    if (!petsc_A_) {
      // 现有流程中 A 矩阵不随步变化，构造一次后复用。
      petsc_A_ = BuildPetscAijFromHypre(A);
    }
    petsc_ksp_->SetOperator(*petsc_A_);

    mfem::petsc::KSP ksp = *petsc_ksp_;
    mfem::petsc::PC pc = nullptr;
    // 先设默认 PCNONE，再让 KSPSetFromOptions 接管最终 PC/KSP 选择。
    CheckPetscError(KSPGetPC(ksp, &pc), "KSPGetPC");
    CheckPetscError(KSPSetType(ksp, KSPCG), "KSPSetType");
    CheckPetscError(
        KSPSetInitialGuessNonzero(ksp, PETSC_TRUE), "KSPSetInitialGuessNonzero");
    CheckPetscError(PCSetType(pc, PCNONE), "PCSetType");
    CheckPetscError(KSPSetFromOptions(ksp), "KSPSetFromOptions");
    if (petsc_use_geometric_asm_) {
      // 若最终 PC 是 ASM/GASM，则按分区 ownership 显式设置子域。
      ConfigurePartitionSubdomains(petsc_asm_subdomains_, ksp);
    }
    return;
  }
#endif
  // 非 PETSc 路径：MFEM CG，可选 BoomerAMG 预条件。
  if (use_hypre_boomeramg_) {
    // 按 MFEM 官方示例方式：由算子 A 构造 BoomerAMG。
    amg_ = std::make_unique<mfem::HypreBoomerAMG>(A);
    amg_->SetPrintLevel(0);
    cg_->SetPreconditioner(*amg_);
  }
  cg_->SetOperator(A);
}

void LinearSystemSolver::Solve(const mfem::Vector& rhs, mfem::Vector& x) {
#ifdef MFEM_USE_PETSC
  if (use_petsc_) {
    // 缓存每步求解统计，供主程序写入 ksp_history.csv。
    petsc_ksp_->Mult(rhs, x);
    last_num_iterations_ = petsc_ksp_->GetNumIterations();
    last_final_norm_ = petsc_ksp_->GetFinalNorm();
    return;
  }
#endif
  cg_->Mult(rhs, x);
  last_num_iterations_ = cg_->GetNumIterations();
  last_final_norm_ = cg_->GetFinalNorm();
}

}  // namespace mono
