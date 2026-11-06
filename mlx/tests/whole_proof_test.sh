#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# A whole proof by the kernels against one by the plain host paths, under fixed coins: every byte must be
# equal. A planted fault in one product level must be seen. fp_prove_fixed reuses its masks and is a test
# program only. The host paths are slow: this takes some minutes.
# usage: mlx/tests/whole_proof_test.sh <work directory of mlx/demo.sh>
set -uo pipefail
cd "$(dirname "$0")/../.."
W=$1; K=prover/examples/target_small_f2_roomy
ID=$(sed -n 's/^commitment ID: //p' "$W/commit.log"); T=$W/whole; mkdir -p "$T"
cp "$W/isk/seed" "$T/seed"; printf '%020d\n' 0 > "$T/seed.spent"
one() { # name, then environment
  local n=$1; shift
  env "$@" mlx/build/fp_prove_fixed "$K" "$W/ivk" "$W/ref" "$T/seed" "$W/request.req" auto "$ID" "$T/$n.proof" mlx/build/fp_companion > "$T/$n.log" 2>&1 || { echo "FAILED: $n exit $?"; tail -2 "$T/$n.log"; exit 1; }
  grep -E '^proof:' "$T/$n.log" | sed "s/^/$n: /"
}
one kernels A=1
one again A=1
one host FP_TEST_REFERENCE=1
one fault FP_TEST_FAULT=20
cmp "$T/kernels.proof" "$T/again.proof" && echo "ok: two runs of the kernels give the same bytes" || { echo "FAILED: fixed coins do not fix the proof"; exit 1; }
cmp "$T/kernels.proof" "$T/host.proof" && echo "ok: the kernels and the host paths give the same bytes" || { echo "FAILED: the kernels differ from the host paths"; exit 1; }
if cmp -s "$T/kernels.proof" "$T/fault.proof"; then echo "FAILED: a planted fault was not seen"; exit 1; fi
echo "ok: the planted fault changes the proof"
bin/cp_verify "$K" "$W/ivk" "$W/request.req" "$T/fault.proof.work/y" "$T/fault.proof" -id "$ID" > "$T/v.txt" 2>&1; c=$?
echo "cp_verify on the faulty proof: exit $c, $(tail -1 "$T/v.txt")"
[ $c -ne 0 ] || { echo "FAILED: the faulty proof was accepted"; exit 1; }
echo "whole_proof_test: all hold"
