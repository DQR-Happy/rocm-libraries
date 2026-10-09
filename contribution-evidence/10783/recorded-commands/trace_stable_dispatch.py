"""Verify every final selection and API against the actual emitted kernel trace."""

import fcntl
import json
import os
from pathlib import Path
import subprocess

import yaml

from trace_fallback import parse_sequences


root = Path("/mnt/data/rocm-10783")
source = root / "artifacts/stable-selection"
output = root / "results/stable-target-dispatch"
output.mkdir(exist_ok=False)
document = yaml.safe_load((source / "trace.yaml").read_text())
expected = json.loads((source / "trace-expected.json").read_text())
mapping = json.loads((root / "artifacts/baseline-solution-map.json").read_text())
for case in document["Tests"]:
    case.update(unit_check=0, timing=1, iters=1, cold_iters=0,
                flush_memory_size=0, flush_batch_count=1)
suite = output / "input.yaml"
suite.write_text(yaml.safe_dump(document, sort_keys=False))
prefix = root / "candidate-stable"
env = dict(os.environ)
for key in ("ROCBLAS_LAYER", "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH",
            "ROCBLAS_GEMM_OVERRIDE_PATH", "ROCBLAS_TENSILE_LIBPATH"):
    env.pop(key, None)
env.update(ROCBLAS_USE_HIPBLASLT="0", TENSILE_DB="0x8000",
           LD_LIBRARY_PATH=f"{prefix}/lib:/opt/rocm-7.2.4/lib:/opt/rocm-7.2.4/lib/llvm/lib")
command = [str(prefix / "bin/rocblas-bench"), "--yaml", str(suite)]
(output / "invocation.json").write_text(json.dumps({"command": command, "environment": {
    key: env[key] for key in ("ROCBLAS_USE_HIPBLASLT", "TENSILE_DB", "LD_LIBRARY_PATH")}}, indent=2)+"\n")
with (root / "environment/gpu-experiment.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (output / "trace.log").open("wb") as stream:
        subprocess.run(command, env=env, cwd=prefix / "bin", stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=1800)
records = parse_sequences((output / "trace.log").read_text())
assert len(records) == len(expected) == len(document["Tests"])
rows = []
for case, selection, record in zip(document["Tests"], expected, records):
    assert record["dimensions"] == [case[key] for key in ("M", "N", "K")]
    layout = "Alik" if selection["transA"] == "T" else "Ailk"
    suffix = "_GB" if selection["variant"] == "gb" else ""
    table = f"navi31_Cijk_{layout}_Bljk_BBS_BH{suffix}.yaml"
    kernel = next(row["kernel_name"] for row in mapping
                  if row["source_table"] == table and row["logic_index"] == selection["logic_index"])
    rows.append({"case": case, "expected_kernel": kernel, "actual_kernels": record["kernels"],
                 "matched": kernel in record["kernels"]})
summary = {"case_count": len(rows), "matched": sum(row["matched"] for row in rows), "cases": rows}
(output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps({key: value for key, value in summary.items() if key != "cases"}), flush=True)
if not all(row["matched"] for row in rows):
    raise RuntimeError("Final default dispatch does not match measured selections")
