#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Requests beyond the demo's: e < T, r = T - 1, positions that break the policy, a malformed range; then a
# clipped gap, with a token above C, and with two where one is allowed.
# usage: mlx/tests/paths_test.sh <work directory of mlx/demo.sh> <empty work directory>
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH
W=$1; G=$2; T=$G/t; mkdir -p "$T"; bad=0
K=prover/examples/target_small_f2_roomy; KG=prover/examples/target_small_f2_gap
GAP=clipped-gap,9244130,323343,1,16,8
ID=$(sed -n 's/^commitment ID: //p' "$W/commit.log")
python3 - "$W/request.req" "$T" <<'P'
import struct, sys
b = open(sys.argv[1], 'rb').read(); n = len(b) // 4; w = list(struct.unpack('<%dI' % n, b)); T = n - 2
def put(name, x): open(sys.argv[2] + '/' + name + '.req', 'wb').write(struct.pack('<%dI' % n, *x))
for name, r, e in (('e32', 15, 32), ('e16', 15, 16), ('r63', 63, 64), ('policy', 0, 64), ('range', 20, 20)):
    x = w[:]; x[T] = r; x[T + 1] = e; put(name, x)
P
try() { # want(yes|exit code) name config dir id [policy]
  local want=$1 n=$2 k=$3 d=$4 id=$5 pol=${6:-}
  mlx/fpmlx prove "$k" "$d" -id "$id" -out "$T/$n.proof" -request "$T/$n.req" > "$T/$n.out" 2>&1; local c=$?
  if [ "$want" != yes ]; then
    echo "$n: prove exit $c: $(grep -hE 'fp_prove:' "$T/$n.out" | tail -1 | cut -c1-100)"
    [ "$c" = "$want" ] || { echo "FAILED: $n: wanted exit $want"; bad=1; }
    return
  fi
  [ $c -eq 0 ] || { echo "FAILED: $n: prove exit $c"; bad=1; return; }
  mlx/fpmlx verify "$k" "$d/ivk" "$T/$n.req" "$T/$n.proof.y" "$T/$n.proof" -id "$id" ${pol:+-policy "$pol"} > "$T/$n.v" 2>&1; c=$?
  echo "$n: verify exit $c: $(tail -1 "$T/$n.v" | cut -c1-70); $(grep -h 'positions above C' "$T/$n.proof.log")"
  [ $c -eq 0 ] || bad=1
}
try yes e32 "$K" "$W" "$ID"
try yes e16 "$K" "$W" "$ID"
try yes r63 "$K" "$W" "$ID"
try 3 policy "$K" "$W" "$ID"
try 2 range "$K" "$W" "$ID"
mlx/fpmlx prepare "$KG" "$G" -tokens "$K/own_choices.req" > "$G/prepare.out" 2>&1 || { echo "FAILED: prepare"; exit 1; }
[ -s "$G/commit.log" ] || mlx/fpmlx commit "$KG" "$G" > /dev/null || { echo "FAILED: commit"; exit 1; }
IG=$(sed -n 's/^commitment ID: //p' "$G/commit.log")
python3 - "$G/request.req" "$T" <<'P'
import struct, sys
b = open(sys.argv[1], 'rb').read(); n = len(b) // 4; w = list(struct.unpack('<%dI' % n, b)); T = n - 2
def put(name, x): open(sys.argv[2] + '/' + name + '.req', 'wb').write(struct.pack('<%dI' % n, *x))
put('gap', w)
x = w[:]; x[T - 1] = (x[T - 1] + 137) % 512; put('gap_one', x)
x[T - 2] = (x[T - 2] + 211) % 512; put('gap_two', x)
P
try yes gap "$KG" "$G" "$IG" "$GAP"
try yes gap_one "$KG" "$G" "$IG" "$GAP"
grep -q 'positions above C: 1' "$T/gap_one.proof.log" || { echo "FAILED: gap_one has no token above C"; bad=1; }
try 3 gap_two "$KG" "$G" "$IG"
[ $bad -eq 0 ] && echo "paths_test: all hold" || { echo "paths_test: FAILED"; exit 1; }
