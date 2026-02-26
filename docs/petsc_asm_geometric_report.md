# PETSc ASM/GASM 分区分解实现报告

## 结论

当前代码已经显式实现并调用了**基于 MFEM 分区所有权的 ASM/GASM 子域分解**：

1. 线性系统矩阵来自 MFEM/Hypre，但在 PETSc 路径会转换为 MPIAIJ。
2. `KSPSetFromOptions` 后读取 `PC` 类型；仅当 `PCASM` 或 `PCGASM` 时进入分区子域配置。
3. 读取 PETSc 矩阵当前 rank 的 ownership range（对应 MFEM true dof 分区所有权）。
4. 为该 rank 构造一个连续 stride `IS` 子域，并调用：
   - `PCASMSetLocalSubdomains(...)` 或
   - `PCGASMSetSubdomains(...)`

因此，子域直接对齐 MFEM/ParMETIS 的并行分区，不依赖坐标分桶。

## 关键实现位置

- `/Users/tianhaoma/Documents/New project/src/solver/LinearSolverFactory.cpp`
  - `ConfigurePartitionSubdomains(...)`
  - `LinearSystemSolver::SetOperator(...)` 中 `KSPSetFromOptions` 之后调用分区子域配置
- `/Users/tianhaoma/Documents/New project/src/solver/LinearSolverFactory.hpp`
  - 保存 PETSc 子域 `IS` 并在析构时销毁
- `/Users/tianhaoma/Documents/New project/src/config/SimulationConfig.hpp`
- `/Users/tianhaoma/Documents/New project/src/config/SimulationConfig.cpp`
  - 配置项：
    - `petsc_use_geometric_asm`（历史命名，当前用于开启显式 ASM/GASM 子域设置）

## 为什么这能保证“按分区分解”

实现链路中，子域集合不是 PETSc 默认自动策略，而是代码按矩阵 ownership（即 MFEM 分区后的 true dof 所有权）显式传给 ASM/GASM。  
只要 `petsc_use_geometric_asm=1` 且 `PC` 实际类型为 `asm/gasm`，就会使用该分区子域。

## 运行时验证方法（已实测）

示例环境变量（注意求解器前缀 `mono_`）：

```bash
export PETSC_OPTIONS='-mono_pc_type asm -mono_sub_pc_type jacobi -mono_ksp_view'
```

短程运行后应在日志看到类似：

- `PC Object: (mono_) ... type: asm`
- `total subdomain blocks = 2`（例如 `-np 2` 时）

子域数与 MPI rank 数一致，代表“每个分区一个 ASM 子域”。

## 已知边界

1. 当前是一层分区 ASM（每 rank 一个子域），并非多层/重叠优化版本。
2. 仍依赖 `KSP` 的前缀选项机制；若误用无前缀选项（如 `-pc_type asm`），PETSc 会提示未使用。
