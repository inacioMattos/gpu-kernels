#!/usr/bin/env python3
"""Run from the repository root after `make benchmark`."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=Path('build/matmul-results'))
parser.add_argument('--rounds', type=int, default=7)
parser.add_argument('--iterations', type=int, default=10)
args = parser.parse_args()
if args.rounds < 1 or args.iterations < 1:
    parser.error('rounds and iterations must be positive')
args.out.mkdir(parents=True, exist_ok=True)

def query_gpu():
    return subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu',
        '--format=csv'], text=True).strip()

sources = ['matmul/benchmark.cu', 'matmul/matmul.cu', 'matmul/k10_experiment.cuh', 'common/check.cuh', 'Makefile']
metadata = {
    'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'platform': platform.system() + ' ' + platform.release(),
    'nvcc': subprocess.check_output(['nvcc', '--version'], text=True).strip(),
    'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
    'gpu_before': query_gpu(),
    'timing': 'CUDA events; median of batch means; 3 warmups before every batch; kernel order rotates each round',
    'transfers_and_allocation_timed': False,
    'runs': [],
}
for n in [2048, 4096]:
    print(f'Benchmarking {n} x {n}...', flush=True)
    command = ['./build/benchmark', str(n), str(args.rounds), str(args.iterations), str(args.out / f'{n}-samples.csv')]
    with (args.out / f'{n}-summary.csv').open('w') as out, (args.out / f'{n}-checks.txt').open('w') as err:
        subprocess.run(command, stdout=out, stderr=err, check=True)
    metadata['runs'].append({'command': command, 'gpu_after': query_gpu()})
metadata['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(args.out / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(f'Results saved to {args.out}', flush=True)
