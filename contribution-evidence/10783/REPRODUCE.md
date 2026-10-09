# Reproduce the comparison

Use two clean source trees at the reported baseline and PR revisions. Install each build into its own prefix. The ROCm toolchain and BLAS reference dependencies must be identical for both builds.

From each checkout's `projects/rocblas` directory:

```bash
python rmake.py -a gfx1100 -c --ninja -j 8 \
  --build_dir "$BUILD_DIR" \
  --cmake-darg LINK_BLIS=OFF \
  --cmake-darg ROCBLAS_CLIENTS_REFERENCE_ILP64=ON \
  --cmake-darg Tensile_CPU_THREADS=8 \
  --cmake-darg CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
cmake --install "$BUILD_DIR/release" --prefix "$INSTALL_PREFIX"
```

Use a fresh build directory so the generated Tensile library includes the added logic files. Confirm that each installed library matches its build output. Do not use the distribution's prebuilt rocBLAS package as the baseline for this source comparison.

## Performance

The saved `benchmarks/*/input.yaml` files already specify data types, shapes, leading dimensions, strides, batch counts, iterations, warmups, and cache settings. With `BASELINE` and `CANDIDATE` set to the corresponding installation prefixes, run from this report directory:

```bash
unset TENSILE_DB ROCBLAS_LAYER ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH
unset ROCBLAS_GEMM_OVERRIDE_PATH ROCBLAS_TENSILE_LIBPATH
export ROCBLAS_USE_HIPBLASLT=0
mkdir -p rerun

for repeat in 0 1 2; do
  if [ "$repeat" -eq 1 ]; then
    order="candidate baseline"
  else
    order="baseline candidate"
  fi
  for label in $order; do
    if [ "$label" = baseline ]; then prefix="$BASELINE"; else prefix="$CANDIDATE"; fi
    LD_LIBRARY_PATH="$prefix/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib" \
      "$prefix/bin/rocblas-bench" \
      --yaml benchmarks/stable-squares-hot-pair/input.yaml \
      > "rerun/hot-$repeat-$label.csv"
  done
done
```

Repeat with the rotating input for the other cache configuration. Match the complete case identity before comparing median times. Keep other GPU workloads stopped during measurement. The published ratios are `baseline_us / candidate_us`.

`measurements.csv` contains the preserved per-repetition times used to produce the published medians. Groups marked `fresh_on_final_candidate=false` retain earlier evidence only for selections or fallback paths shown unchanged by the transition audit. Those rows must not be described as fresh runs of the final candidate.

## Host tests

From the checkout root, with the component's pinned googletest dependency available:

```bash
cmake -S shared/tensile/HostLibraryTests -B "$HOST_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_CXX_FLAGS=-D__HIP_HCC_COMPAT_MODE__=1 \
  -DGPU_TARGETS=gfx1100 -DTensile_ARCHITECTURE=gfx1100 \
  -DTensile_CPU_THREADS=8 \
  -DTensile_ROOT="$PWD/shared/tensile/Tensile" \
  -DTENSILE_USE_HIP=ON -DTENSILE_USE_LLVM=ON \
  -DTENSILE_BUILD_CLIENT=ON -DTENSILE_USE_MSGPACK=ON \
  -DTENSILE_USE_OPENMP=ON \
  -DLLVM_DIR=/opt/rocm/lib/llvm/lib/cmake/llvm
cmake --build "$HOST_BUILD" --parallel 8
```

This revision generates single-architecture fixture artifacts under `library/gfx1100`, while several legacy tests load flat paths. The local run exposes those generated files at the expected paths using build-only symlinks. No source fixture is substituted. The following reproduces that build-directory adaptation; the evidence includes the target hashes:

```bash
python3 - "$HOST_BUILD" <<'PY'
from pathlib import Path
import sys
build = Path(sys.argv[1]).resolve()
for directory in sorted((build / 'data').glob('test_*/library/gfx1100')):
    for target in directory.iterdir():
        if target.suffix not in ('.dat', '.yaml', '.co', '.hsaco'):
            continue
        link = directory.parent / target.name
        if not link.exists():
            link.symlink_to(Path('gfx1100') / target.name)
        assert link.read_bytes() == target.read_bytes()
PY
```

The repaired tests can be run separately with:

```bash
cd "$HOST_BUILD"
./TensileTests --gtest_filter='EqualityMatchingTest.*:MultipleSolutionsPerSize.*:HipSolutionAdapterTest.BetaOnlyKernel_*'
```

The final host-suite invocation has no test filter or shard configuration and writes JUnit XML. Its case inventory is compared with the earlier complete 3,513-case run.

Set `OMP_THREAD_LIMIT=8` for this container when running the full suite. Some reference code explicitly requests more threads, so `OMP_NUM_THREADS=8` alone does not bound the CPU work. This setting changes execution resources, not the selected tests or numerical criteria.

## Component checks

Tensile's guide requires `tox run -m precommit` and `tox run -m prepr`. The recorded `prepr` invocation selects every environment in that label. Its explicit overrides only constrain workers/build resources, preserve XML, and configure/build/run the full host suite. The default `hostlibtest` tox command builds but does not run the tests; the recorded command executes them as well. See the invocation JSON for the exact overrides and the result JSON for each environment's exit status.

The native rocBLAS run selects the project's BF16-input/output quick/pre-checkin GEMM tests, bad-argument tests, and get-solutions tests. Coverage is checked against the native test inventory, including missing and unexpected case detection. Supplemental layout tests use the project's existing numerical reference and error criteria.
