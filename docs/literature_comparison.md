# Niederer Benchmark Literature Comparison

## 1. References

Primary references used for target range:

1. Niederer SA, et al. *Verification of cardiac tissue electrophysiology simulators using an N-version benchmark*. Philos Trans A (2011).
2. Rosi G, et al. *LifeX: an open source finite element library for cardiac electrophysiology* (benchmark discussion and convergence notes, open access):
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC8078854/

Secondary implementation reference (for probe-table style comparisons):

3. FEniCS beat Niederer benchmark demo:
   - https://finsberg.github.io/fenics-beat/demos/niederer_benchmark.html

## 2. Literature target (qualitative)

From open-access benchmark discussion (Ref 2):

- the original N-version benchmark reports a spread in activation times across implementations;
- for the highly resolved setup, last activation is reported around low-40 ms (roughly 42-43 ms range).

This is used here as a sanity target for propagation speed order-of-magnitude.

## 3. Our runs and comparison

### 3.1 Baseline config (`config/niederer_50ms.options`)

- 50 ms run: far probes not activated (`P8=NA`).
- 80 ms run: `P8=62.58 ms`, significantly slower than low-40 ms target.

Conclusion: baseline parameters are too slow for this benchmark target.

### 3.2 Iterated fit config (`config/niederer_70ms_litfit.options`)

- kept anisotropy ratio (`sigma_f:sigma_s:sigma_n`) unchanged.
- scaled all three monodomain conductivities up.
- obtained `P8=40.75 ms`.

Conclusion: propagation speed enters the expected benchmark order-of-magnitude.

## 4. Notes

- This comparison is a **sanity check**, not yet a publication-grade validation.
- For final claims, compare against exactly matched benchmark protocol
  (mesh resolution, ionic model setup, stimulus definition, and post-processing threshold).
