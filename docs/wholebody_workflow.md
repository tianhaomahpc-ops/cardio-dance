# Whole-body 工作流与对比验证指南

## 1. 目标

在现有 monodomain（心脏）基础上，增加：

1. 心脏 extracellular recovery（`ue`）。
2. 躯干电位求解（`uT`）。
3. 双网格与 conforming 单网格两种耦合路径。

## 2. 代码入口

- 主程序调度：`src/main.cpp`
- KSP2：`src/solver/ExtracellularRecoverySolver.*`
- KSP3：`src/solver/TorsoPotentialSolver.*`
- 映射：`src/coupling/InterfaceMapper.*`
- 输出：`src/io/OutputManager.*`
- torso 网格生成：`tools/generate_torso_box.cpp`
- conforming 网格生成：`tools/generate_conforming_wholebody_case.cpp`

## 3. 快速复现

### 3.1 生成 torso 包围盒网格

```bash
mpirun -np 1 ./build/generate_torso_box \
  --heart-mesh benchmarks/niederer/niederer_benchmark.mesh \
  --out-mesh benchmarks/torso_box/torso_box.mesh \
  --padding-mm 8.0 \
  --h-mm 1.0
```

### 3.2 运行 whole-body

```bash
mpirun -np 1 ./build/monodomain --config config/wholebody_default.options
```

MPI smoke:

```bash
mpirun -np 4 ./build/monodomain --config config/wholebody_default.options --t-end 0.05
```

### 3.3 Conforming 单网格模式（推荐）

生成单一 whole-body 网格（心脏/躯干用 volume attributes 区分）：

```bash
mpirun -np 1 ./build/generate_conforming_wholebody_case \
  --heart-mesh benchmarks/niederer/niederer_benchmark.mesh \
  --out-mesh benchmarks/wholebody/wholebody_conforming.mesh \
  --padding-mm 8.0 \
  --h-mm 1.0 \
  --heart-attr 1 \
  --torso-attr 2
```

运行 conforming 配置：

```bash
mpirun -np 1 ./build/monodomain --config config/wholebody_conforming.options
```

## 4. 文献对比建议流程

### 4.1 心脏内对比（先锁定）

先用 Niederer benchmark 固定心脏参数，比较：

1. 激动到达时间（建议使用 `--benchmark-probes 1`）。
2. `Vm` 峰值与激动时序。
3. 网格尺度变化下的收敛趋势（h=0.5/0.2/0.1 mm）。

如果心脏内传播不对，不要进入 torso 对比。

当前仓库内可复现实测（`config/niederer_50ms.options`）：

- 50 ms 末端角点未完全激活（`P8=NA`）。
- 延长到 80 ms 时 `P8=62.58 ms`，明显慢于公开基准常见量级（约 40+ ms）。

因此默认参数对该基准偏慢，需要迭代。

### 4.2 躯干场对比（再锁定）

在心脏部分稳定后，比较：

1. `uT` 空间分布形态。
2. 指定体表点（electrode）时间序列波形（振幅与相位）。
3. 多个时刻的等势面与文献图对照。

### 4.3 迭代优先级

出现偏差时建议按顺序迭代：

1. `sigma_i/sigma_e/sigma_torso` 参数与单位。
2. heart->torso 映射距离阈值 `interface_map_max_dist_mm`。
3. torso 网格分辨率（`--h-mm`）。
4. 时间步（`dt_pde_ms`, `dt_ode_ms`）。
5. 再考虑更换界面施加策略与网格组织方式。

本轮已完成一次参数迭代示例（`config/niederer_70ms_litfit.options`）：

- `sigma_f/s/n` 在保持比例不变情况下整体提高。
- 实测 `P8=40.75 ms`，进入文献常见范围量级（更接近公开 benchmark 结果）。

## 5. 当前方案的边界

当前实现支持双路径：

1. 双独立网格 + 最近邻映射 + penalty 约束。
2. conforming 单网格 + 子域抽取（`ParSubMesh`）+ penalty 约束。

仍有的局限：

1. penalty 约束不是严格强制 Dirichlet。
2. conforming 路线已是 parent-id 直连映射，但仍使用 `Allgatherv` 同步接口值（可继续优化通信量）。
3. ECG 发表级精度还需要标准化电极/导联系统。

## 6. 更优实现路线（推荐）

若目标是“大规模并行 + 高可信 ECG 对比”，下一步建议：

1. **保留 conforming whole-body mesh**，继续把 parent-id 直连映射做成更低通信开销版本。
2. **严格界面条件施加**（对称消元/拉格朗日乘子/Nitsche）。
3. **电极与导联系统标准化**（12 导联、参考电极、滤波流程）。
4. **参数反演或文献标定表**（统一电导参数与单位体系）。
5. **并行优化**（块预条件、分层网格、I/O 降频与异步输出）。

在该路线下，心脏-躯干接口不再需要最近邻映射，数值误差和并行通信复杂度都会更可控。
