import subprocess
import os, subprocess, resource
import argparse
from enum import Enum

class BenchType(str, Enum):
    small = "small"
    size = "size"
    table4 = "table4"
    table3 = "table3"
    variance = "variance"
    lambda_ = "lambda"   # "lambda" is a Python keyword
    all = "all"

BENCH_CHOICES = [e.value for e in BenchType]

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--simd',   dest='s', type = int, required = True)
parser.add_argument('-r', '--repetitions',   dest='r', type = int, required = False)
parser.add_argument('-m', '--mal-sec',   dest='m', action='store_true')
parser.add_argument('-n', '--network',   dest='n', action='store_true')
parser.add_argument('-b', '--build',  action='store_true', dest='b')
parser.add_argument('-d', '--debug',  action='store_true', dest='d')
parser.add_argument('-t', '--trace',  action='store_true', dest='t')
parser.add_argument("bench", choices=BENCH_CHOICES, help="Benchmark suite to run")



args = parser.parse_args()

STACK_BYTES = 256 * 1024 * 1024  # 256 MiB
def set_stack():
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    new_soft = min(STACK_BYTES, hard if hard != resource.RLIM_INFINITY else STACK_BYTES)
    resource.setrlimit(resource.RLIMIT_STACK, (new_soft, hard))

def test(exe):
    procs = []
    env = os.environ.copy()
    if args.t:
        env["RUST_LOG"] = "lut_sampler=trace"
    for i in range(3):
        source = "release" if not args.d else "debug"
        cmd = [
            f"./target/{source}/{exe}",
            "--config",
            f"p{i + 1}.toml",
            "--simd", 
            f"{args.s}"
        ]
        
        if args.m:
            cmd.append("--mal-sec")
        if args.n:
            cmd.append("--network")
        if args.d:
            cmd.append("--debug")
        if args.r:
            cmd.append("--rep")
            cmd.append(f"{args.r}")
        cmd.append("--bench")
        cmd.append(f"{args.bench}")
        # print(cmd)
        redir = None if i == 0 else subprocess.DEVNULL
        procs.append(subprocess.Popen(
            cmd, 
            stdout=redir, 
            stderr=redir,
            env=env,
            preexec_fn=set_stack
        ))
    for p in procs:
        p.wait()

exe = "sampler"

# test()
if args.b:
    if not args.d:
        ret = subprocess.Popen(f'RUSTFLAGS="-C target-cpu=native " cargo build --release --bin {exe} --features clmul', shell=True).wait()
    else:
        ret = subprocess.Popen(f'RUSTFLAGS="-C target-cpu=native " cargo build --bin {exe} --features clmul', shell=True).wait()
    if ret != 0:
        raise RuntimeError("Build failed")
test(exe)
