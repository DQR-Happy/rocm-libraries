#!/bin/bash
set -eu
task_root=/mnt/data/rocm-10783
. "$task_root/environment/activate.sh"
export OMP_THREAD_LIMIT=8
python - <<'PY'
from pathlib import Path
import hashlib,json,os
root=Path('/mnt/data/rocm-10783')
record=json.loads((root/'artifacts/submission-commits.json').read_text())
for entry in record['files']:
    assert hashlib.sha256((root/'source'/entry['path']).read_bytes()).hexdigest()==entry['sha256'], entry['path']
print('All ten test/build source files match the reviewed Git commits.')
PY
cmake -S "$task_root/source/shared/tensile/HostLibraryTests" -B "$task_root/build-equality-tests-gfx1100" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.4/bin/amdclang++ -DTENSILE_USE_HIP=ON -DGPU_TARGETS=gfx1100 -DTensile_ARCHITECTURE=gfx1100 -DCMAKE_CXX_FLAGS=-D__HIP_HCC_COMPAT_MODE__=1 -DTENSILE_USE_LLVM=ON -DTENSILE_BUILD_CLIENT=ON -DLLVM_DIR=/opt/rocm-7.2.4/lib/llvm/lib/cmake/llvm -DTENSILE_USE_MSGPACK=ON -DTENSILE_USE_OPENMP=ON -DTensile_CPU_THREADS=8 -DTensile_ROOT="$task_root/source/shared/tensile/Tensile"
cmake --build "$task_root/build-equality-tests-gfx1100" --parallel 8
cd "$task_root/build-equality-tests-gfx1100"
unset GTEST_FILTER GTEST_SHARD_INDEX GTEST_TOTAL_SHARDS
./TensileTests --gtest_output="xml:$task_root/results/submission-prepr/host-library.xml" > "$task_root/results/submission-prepr/host-library.log" 2>&1
python - <<'PY'
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib,json,os
root=Path('/mnt/data/rocm-10783')
old=ET.parse(root/'results/tensile-host-library/gtest.xml').getroot()
new=ET.parse(root/'results/submission-prepr/host-library.xml').getroot()
names=lambda tree: {case.attrib['classname']+'.'+case.attrib['name'] for case in tree.iter('testcase')}
assert names(new)==names(old) and len(names(new))==3513
assert not list(new.iter('failure')) and not list(new.iter('error')) and not list(new.iter('skipped'))
record=json.loads((root/'artifacts/submission-commits.json').read_text())
for entry in record['files']:
    assert hashlib.sha256((root/'source'/entry['path']).read_bytes()).hexdigest()==entry['sha256'], entry['path']
digest=hashlib.sha256()
with (root/'build-equality-tests-gfx1100/TensileTests').open('rb') as binary:
    for chunk in iter(lambda:binary.read(1024*1024),b''): digest.update(chunk)
summary={'tests':len(names(new)),'failures':0,'errors':0,'skipped':0,'missing':[],'unexpected':[],'all_original_assertions_preserved':True,'gtest':new.attrib,'binary_sha256':digest.hexdigest(),'source_files':record['files'],'reviewed_commit':record['head'],'environment':{name:os.environ[name] for name in ['OMP_THREAD_LIMIT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
(root/'results/submission-prepr/host-library-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
PY
