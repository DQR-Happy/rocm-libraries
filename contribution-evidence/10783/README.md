# gfx1100 BF16 GEMM validation

This report accompanies the contribution for [ROCm/rocm-libraries#10783](https://github.com/ROCm/rocm-libraries/issues/10783). The profiling in that issue and vlluvia's earlier [#10786](https://github.com/ROCm/rocm-libraries/pull/10786) provided the starting point. The measurements here use newly built baseline and candidate libraries, not the earlier PR's API solution indices.

Status: all local submission checks completed successfully. Upstream CI and maintainer review remain separate.

## Environment and source

| Item | Configuration |
|---|---|
| GPU | W7900D, gfx1100, 96 CUs, 48 GiB |
| CPU and RAM limits | 16-CPU quota, 64 GiB |
| OS | Ubuntu 22.04.5 container; host kernel 6.8.0-79 |
| ROCm | 7.2.4.70204-93, HIP 7.2.53211, AMD Clang 22 |
| Python and CMake | Python 3.10, CMake 3.31.6 |
| Baseline sources | `930b79499cb88cd76971c0c2002b374112809abb` |
| Submission base | `6d8061a84c301acc95c01a529ff1bc00133130f6` |
| Submitted source | `fee38d4d5cad3d43dd8270378cb5e561cd0758a4` |
| rocBLAS backend | Tensile, `ROCBLAS_USE_HIPBLASLT=0` |
| Reference | Project test clients and ILP64 OpenBLAS |
| GPU settings | Existing clock and power settings; no overclocking or power-limit changes |

The PR was prepared on a newer `develop`. The upstream Tensile tree, rocBLAS runtime sources, client headers, and BF16 test sources are unchanged from the tested baseline. Later upstream changes add general version-string tests and modify coverage-only build paths; the recorded rocBLAS build has `BUILD_CODE_COVERAGE=OFF`. Those new auxiliary tests are not claimed as executed here. All ten submitted files match the validated working tree. The beta-only fixture repair affects HostLibraryTests; the production lookup and GEMM logic files are unchanged from the measured candidate. The final source certificate records the revisions, tree identities, file hashes, and review of these upstream changes.

## Selection and scope

The candidate adds four native Equality logic files. Their keys are M/N/batch/K (`indexOrder=[2,3,0,1]`). The existing Euclidean tables are byte-identical to the baseline. Each record reuses a measured existing solution; leading dimensions remain subject to that solution's predicates.

The solution numbers below refer to the original source logic tables, not to the public API's generated indices.

| Path | Transposes | M=N=K | Batch count -> source solution |
|---|---|---:|---|
| Ordinary / strided | T/N | 512 | 1, 3 -> 138 |
| Ordinary / strided | T/N | 1024 | 1 -> 145 |
| Ordinary / strided | T/N | 1536 | 1 -> 58 |
| Ordinary / strided | T/N | 2048 | 1 -> 71 |
| Ordinary / strided | T/N | 4096 | 1, 3 -> 71 |
| Ordinary / strided | T/N | 6144 | 1 -> 44 |
| Ordinary / strided | N/N | 2048 | 1 -> 81 |
| Ordinary / strided | N/N | 8192 | 1, 3, 8 -> 80 |
| Pointer-batched | T/N | 512 | 3 -> 138 |
| Pointer-batched | T/N | 4096 | 8 -> 71 |
| Pointer-batched | N/N | 8192 | 8 -> 80 |

There are 15 records and ten solution configurations after deduplication within each file. Batch one covers both `gemm_ex` and `gemm_strided_batched_ex`. Unlisted shapes and batch counts fall back to the original library.

An earlier experiment added square points to the Euclidean tables. It made 25 of 144 nearby rectangular cases more than 10% slower, which motivated limiting the final change to exact keys. The rejected candidate's [measurements](evidence/results/initial-catchment-pair/comparison.json) are retained separately and are not included in the final candidate's performance table.

## Performance

The benchmark runner alternates AB/BA order, checks full case identity, and compares the median of repeated samples. `measurements.csv` includes every sample, leading dimension, stride, batch count, cache group, and whether the result was freshly measured on the final candidate. Individual input YAML and comparison JSON files are in `benchmarks/`.

The hot configuration sets `flush_memory_size=0`. The rotating configuration requests a 256 MiB working set; this does not guarantee multiple buffers for matrices larger than that set. The four final single-batch groups use three repetitions. The focused T/N 2048 comparison uses five repetitions and 300 iterations. Exact invocation details are retained with the evidence.

| Group | Cases | Geometric-mean speedup | Minimum speedup | Fresh on final candidate |
|---|---:|---:|---:|---|
| Square shapes, hot, including controls | 32 | 1.0901x | 0.9932x | Yes |
| Square shapes, rotating, including controls | 32 | 1.0891x | 0.9910x | Yes |
| Additional strides | 24 | 1.1091x | 0.9888x | Yes |
| Strided batch one | 40 | 1.1158x | 0.9935x | Yes |
| Affected T/N 2048, hot | 10 | 1.0986x | 1.0313x | Yes |
| Affected T/N 2048, rotating | 10 | 1.1018x | 1.0313x | Yes |
| Batched shapes | 64 | 1.0351x | 0.9890x | No; unchanged selections |
| Batched strides | 39 | 1.0995x | 0.9944x | No; unchanged selections |
| Neighboring rectangles | 144 | 1.0002x | 0.9876x | No; unchanged fallback |
| Broad grid | 1,024 | 0.9998x | 0.9186x | No; unchanged fallback |

The selected square subset contains 16 cases per cache configuration: eight shapes, tight and padded. Its geometric-mean speedups are 1.188334x hot and 1.188090x rotating. See [the eight tight cases](selected-square-table.md) for individual baseline/candidate times. Maximum speedup is approximately 2.253x.

Two short grid measurements exceeded the 5% diagnostic threshold. Repeating those cases seven times with 1,000 iterations gave speedups of 1.00763x and 1.00131x. The original observations remain in the dataset. Neither 5% nor 10% is presented as an upstream acceptance threshold.

## Correctness and regression checks

The final rocBLAS candidate freshly passes 27,457 native BF16 quick/pre-checkin, bad-argument, and get-solutions cases, plus 12 affected semantic/layout cases: 27,469 total, no failures or skips. All 115 target dispatch checks match; all 45 affected fallback comparisons retain the baseline kernel sequence.

Another 126 semantic/layout cases and the complete 2,419-case fallback trace retain results from the preceding four-key candidate. The final refinement changed only T/N 2048 batch one from solution 139 to 71. The other 14 records and all fallback semantics are unchanged, and the rocBLAS shared library, benchmark/test executables, and GPU code packages are byte-identical between those two candidates. `stable-transition.json` documents this boundary. The 45 fresh fallback checks overlap the earlier trace and are not added to it as independent coverage.

The Equality regression tests reproduce rejected-duplicate failures and an empty-table crash before the fix. The repaired lookup passes all six added cases and the existing matching test. The beta-only tests also failed in the preserved baseline; their independent Source fixture now generates the missing kernel and both original tests pass without changing their assertions.

Pre-fix reproductions are retained under `evidence/results/equality-red/` and `evidence/results/baseline-host-failures/`. They document the bugs being fixed; final acceptance results are under `evidence/results/submission-prepr/`.

`stable-summary.json` predates the beta-only fixture repair and retains the earlier host-suite failures. The final `submission-prepr` evidence supersedes that host-test entry; the GEMM measurements and production code are unchanged.

Tensile's complete `prepr` label passes with the recorded build/resource overrides. All seven environments succeed: format, lint, isort, security, unittest, ci, and hostlibtest.

| Suite | Passed | Upstream skips | Failures |
|---|---:|---:|---:|
| Full Debug HostLibraryTests | 3,513 | 0 | 0 |
| Python unit tests | 5,485 | 3 | 0 |
| Pre-checkin configurations | 105 | 37 | 0 |

The host suite runs every case without filtering or sharding, including all six Equality regressions and both repaired BetaOnly tests. The 37 pre-checkin and three Python skips match the unchanged upstream definitions; their reasons are documented in the report. Both CMake entry points also build the new embedded fixture.

The previous complete pre-checkin run skipped 37 YAML configurations: 36 carry an upstream `skip-gfx1100` mark, and `direct_to_vgpr/dtv_sgemm_lite.yaml` carries an unconditional `skip`. Three Python unit tests are also explicitly disabled upstream: `test_WriteClientLibraryFromSolutions` (assembler discovery), `test_filterProcessingErrors` (debugging function), and `test_pad_4x1` (convolution padding). The skip definitions and test collector are unchanged. These cases are not counted as passed; the final run is checked against the same inventory. The source audit is included with the submission evidence.

## Build and test notes

The full Debug host suite is built for gfx1100 with the project's HostLibraryTests CMake entry point and HIP/LLVM/MessagePack enabled. The second, `next-cmake` entry point was configured independently and its new embedded beta-only fixture object was compiled successfully.

The `prepr` run uses the repository's complete environment label. Recorded command-line overrides bound CI to four workers with two compiler threads each and an OpenMP thread limit of eight, save JUnit XML, and replace the default build-only host command with explicit configure/build/full-suite execution. No test filter or expected-result threshold is weakened. The legacy tests expect flat fixture paths, so the already generated gfx1100 fixture files are exposed through build-directory symlinks; these links are not part of the source change.

The root pre-commit command was run on the changed files. Its configured exclusions and file types skipped all hooks for this change. Component clang-format and Tensile's own format/lint/security/unit checks provide the applicable checks.

## Limits

These are GEMM microbenchmarks on one W7900D and the recorded driver/user-space combination. Other gfx1100 devices, vLLM end-to-end performance, and upstream CI need their respective validation. Maintainer review and repository CI determine whether the PR is accepted.
