"""Run the project's BF16-output GEMM quick/pre-checkin regression tests."""

import argparse
import datetime
import fcntl
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


TYPE = "bf16_rbf16_rbf16_rbf16_rf32_r"
PATTERNS = [
    f"_/gemm{batch}_ex.*/{category}_*{TYPE}_*"
    for batch in ("", "_batched", "_strided_batched")
    for category in ("quick", "pre_checkin")
] + [
    f"_/gemm{batch}_ex.*/quick_*bad_arg_bf16_r_*"
    for batch in ("", "_batched", "_strided_batched")
] + [f"_/gemm_ex_get_solutions.*{TYPE}_*"]


def selected_inventory(path):
    names = set()
    suite = ""
    for line in path.read_text().splitlines():
        if line.startswith("_/") and line.endswith("."):
            suite = line
        elif line.startswith("  ") and suite:
            name = suite + line.strip().split("#", 1)[0].strip()
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in PATTERNS):
                names.add(name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("inventory", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty; choose a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    expected = selected_inventory(args.inventory)
    if not expected:
        raise ValueError("Native regression filter selected no inventoried tests")
    (args.output / "expected-tests.txt").write_text("\n".join(sorted(expected)) + "\n")
    env = dict(os.environ)
    for key in ("TENSILE_DB", "ROCBLAS_LAYER", "ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH", "ROCBLAS_GEMM_OVERRIDE_PATH", "ROCBLAS_TENSILE_LIBPATH"):
        env.pop(key, None)
    env.update(ROCBLAS_USE_HIPBLASLT="0", LD_LIBRARY_PATH=f"{args.prefix}/lib:/opt/rocm-7.2.4/lib:/opt/rocm-7.2.4/lib/llvm/lib")
    xml = (args.output / "gtest.xml").resolve()
    command = [str(args.prefix / "bin/rocblas-test"), "--gtest_filter=" + ":".join(PATTERNS), f"--gtest_output=xml:{xml}"]
    event = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": command, "expected_tests": len(expected), "environment": {k: env[k] for k in ("ROCBLAS_USE_HIPBLASLT", "LD_LIBRARY_PATH")}}
    (args.output / "invocation.json").write_text(json.dumps(event, indent=2) + "\n")
    print(json.dumps(event), flush=True)
    with Path("/mnt/data/rocm-10783/environment/gpu-experiment.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.output / "gtest.log").open("wb") as log:
            result = subprocess.run(command, env=env, cwd=args.prefix / "bin", stdout=log, stderr=subprocess.STDOUT, timeout=14400, check=False)
    root = ET.parse(xml).getroot()
    failures = []
    skipped = []
    actual = set()
    for test in root.iter("testcase"):
        name = test.attrib["classname"] + "." + test.attrib["name"]
        actual.add(name)
        if test.find("failure") is not None or test.find("error") is not None:
            failures.append({"name": name, "details": [node.attrib for node in test]})
        if test.find("skipped") is not None:
            skipped.append({"name": name, "details": [node.attrib for node in test]})
    summary = {"returncode": result.returncode, "gtest": root.attrib, "failures": failures, "skipped": skipped, "missing": sorted(expected - actual), "unexpected": sorted(actual - expected)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("failures", "skipped")}), flush=True)
    # Preserve any baseline failures for comparison instead of discarding the
    # remaining experiment. Infrastructure and coverage errors still abort.
    if result.returncode not in (0, 1) or summary["missing"] or summary["unexpected"]:
        raise RuntimeError("Native test execution or coverage was incomplete")


if __name__ == "__main__":
    main()
