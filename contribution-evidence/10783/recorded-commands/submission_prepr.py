"""Run the project's complete prepr label with recorded hardware build overrides."""

import json
import os
from pathlib import Path
import subprocess


root = Path('/mnt/data/rocm-10783')
source = root / 'source/shared/tensile'
result = root / 'results/submission-prepr'
result.mkdir(exist_ok=False)
command = [
    'tox', '--workdir', str(root / 'tox'), 'run', '-m', 'prepr',
    '--result-json', str(result / 'tox-result.json'),
    '-x', 'testenv:hostlibtest.allowlist_externals=/bin/bash',
    '-x', 'testenv:hostlibtest.commands=/bin/bash ' + str(root / 'environment/submission-hostlibtest.sh'),
    '-x', 'testenv:ci.commands=invoke build-client {env:TENSILE_CLIENT_ARGS}\n'
          'py.test {toxinidir}/Tensile/Tests {[vars]options} -m pre_checkin '
          '--numprocesses=4 --global-parameters=CpuThreads=2 '
          '--basetemp=' + str(root / 'submission-ci-temp') + ' '
          '--junitxml=' + str(result / 'pre-checkin.xml'),
    '-x', 'testenv:unittest.commands=coverage erase\n'
          'pytest {toxinidir}/Tensile/Tests {[vars]options} -m unit '
          '--cov-append --cov=Tensile --cov-report=term '
          '--junitxml=' + str(result / 'unit.xml'),
]
command += ['-x', 'testenv:ci.pass_env+=OMP_THREAD_LIMIT']
env = dict(os.environ, TENSILE_CLIENT_ARGS='--gpu-targets=gfx1100 --no-build', OMP_THREAD_LIMIT='8')
(result / 'invocation.json').write_text(json.dumps({
    'command': command,
    'cwd': str(source),
    'environment': {key: env[key] for key in ('TENSILE_CLIENT_ARGS', 'OMP_THREAD_LIMIT')},
    'override_reason': 'Bound single-GPU build resources, select gfx1100, retain XML, '
                       'and actually execute the full Debug host suite; the default '
                       'hostlibtest tox command only builds and omits --run.',
    'test_filters_changed': False,
}, indent=2) + '\n')
completed = subprocess.run(command, cwd=source, env=env, check=False)
(result / 'exit-code.txt').write_text(str(completed.returncode) + '\n')
raise SystemExit(completed.returncode)
