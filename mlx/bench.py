#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Measures commit, prove and verify over repeated runs, from a work directory of mlx/demo.sh, and prints the
median, the lowest and the highest of each figure. Only measured numbers: every run must succeed, and every
proof must be accepted, or the script stops. A commitment takes a fresh seed in each run, so its nonce search
differs. The machine must stay awake: run under caffeinate.

usage: caffeinate -dimsu mlx/bench.py WORK_DIR [commits] [proofs]
"""
import os, re, shutil, statistics, subprocess, sys, tempfile

TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = "prover/examples/target_small_f2_roomy"
W = os.path.abspath(sys.argv[1])
NC = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 7
ID = re.search(r"commitment ID: (\w+)", open(W + "/commit.log").read()).group(1)
T = tempfile.mkdtemp(prefix="bench", dir=W)


def timed(cmd, name):
    tf = T + "/time"
    with open(T + "/" + name + ".log", "w") as log:
        r = subprocess.run(["/usr/bin/time", "-l", "-o", tf] + cmd, cwd=TOP, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"{name}: exit {r.returncode}")
    t = open(tf).read()
    m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", t)
    return {"real s": float(m[1]), "user s": float(m[2]), "sys s": float(m[3]), "footprint GB": int(re.search(r"(\d+)\s+peak memory footprint", t)[1]) / 1e9}


rows = {"commit": [], "prove": [], "verify": []}
sizes = []
for i in range(NC):
    d = f"{T}/c{i}"
    os.makedirs(d)
    rows["commit"].append(timed(["mlx/build/fp_commit", K, W + "/ref/matrix.i8", d + "/ivk", d + "/seed"], f"commit{i}"))
    print("commit", i, rows["commit"][-1], flush=True)
    shutil.rmtree(d)
for i in range(NP):
    p = f"{T}/p{i}"
    rows["prove"].append(timed(["mlx/build/fp_prove", K, W + "/ivk", W + "/ref", W + "/isk/seed", W + "/request.req", "auto", ID, p, "mlx/build/fp_companion"], f"prove{i}"))
    sizes.append(os.path.getsize(p))
    rows["verify"].append(timed(["bin/cp_verify", K, W + "/ivk", W + "/request.req", p + ".work/y", p, "-id", ID], f"verify{i}"))
    if open(f"{T}/verify{i}.log").read().strip().splitlines()[-1] != "YES":
        sys.exit("a proof was not accepted")
    print("prove", i, rows["prove"][-1], flush=True)
print(f"\n| Step | runs | real s (median, low-high) | user s | sys s | peak footprint GB |\n|---|---|---|---|---|---|")
for k, v in rows.items():
    c = [f"{statistics.median(x[f] for x in v):.2f} ({min(x[f] for x in v):.2f}-{max(x[f] for x in v):.2f})" for f in ("real s", "user s", "sys s", "footprint GB")]
    print(f"| {k} | {len(v)} | " + " | ".join(c) + " |")
print(f"\nproof size: {min(sizes):,} to {max(sizes):,} bytes over {len(sizes)} proofs; all accepted")
shutil.rmtree(T)
