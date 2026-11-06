#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Milestone 1 end to end: synthetic weights -> commitment -> proof on this Mac -> the stock cp_verify in
# a separate process, which gets the commitment ID and the public files alone, then rejected changes.
# usage: mlx/demo.sh [work directory]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH
W=${1:-$HOME/fpwork/demo}; K=prover/examples/target_small_f2_roomy
mkdir -p "$W"; PY=.venv/bin/python
[ -x bin/cp_verify ] || verifier/build.sh
[ -x mlx/build/fp_commit ] || mlx/build_commit.sh
[ -x mlx/build/fp_prove ] || mlx/build_prove.sh
[ -x mlx/build/fp_companion ] || mlx/build_companion.sh
mlx/fpmlx prepare "$K" "$W" || exit 1
[ -f "$W/commit.log" ] || mlx/fpmlx commit "$K" "$W" > /dev/null || exit 1
ID=$(sed -n 's/^commitment ID: //p' "$W/commit.log")
echo "model:       synthetic weights of $K"
echo "commitment:  $ID"
mlx/fpmlx prove "$K" "$W" -id "$ID" -out "$W/a.proof" -time "$W/prove.time" > "$W/prove.out" 2>&1
c=$?; [ $c -eq 0 ] || { tail -5 "$W/prove.out" "$W/prove.time"; echo "FAILED: prove exit $c"; exit 1; }
Y="$W/a.proof.y"
echo "next token:  $($PY -c "import struct,sys;print(struct.unpack('<I',open(sys.argv[1],'rb').read())[0])" "$Y")"
grep -E '^proof: ' "$W/prove.out"
echo "peak memory: $(awk '/peak memory footprint/{printf "%.2f GB\n",$1/1e9}' "$W/prove.time")"
s=$(date +%s.%N); mlx/fpmlx verify "$K" "$W/ivk" "$W/request.req" "$Y" "$W/a.proof" -id "$ID" > "$W/verify.txt" 2>&1; c=$?; e=$(date +%s.%N)
echo "verify:      exit $c, last line: $(tail -1 "$W/verify.txt"), $(echo "$e - $s" | bc) s"
[ $c -eq 0 ] || { echo "FAILED"; exit 1; }
$PY - "$W" "$Y" <<'PY'
import struct,sys
W,Y=sys.argv[1:3]
y=struct.unpack("<I",open(Y,"rb").read())[0]
open(W+"/y.bad","wb").write(struct.pack("<I",(y+1)%512))
r=bytearray(open(W+"/request.req","rb").read()); r[0]^=1; open(W+"/request.bad","wb").write(r)
p=bytearray(open(W+"/a.proof","rb").read()); p[len(p)//2]^=1; open(W+"/a.bad","wb").write(p)
PY
bad=0
no() { mlx/fpmlx verify "$K" "$W/ivk" "$2" "$3" "$4" -id "$5" > "$W/neg.txt" 2>&1; c=$?; echo "rejected?    $1: exit $c, $(tail -1 "$W/neg.txt")"; [ $c -eq 1 ] || bad=1; }
no "changed output token" "$W/request.req" "$W/y.bad" "$W/a.proof" "$ID"
no "changed request" "$W/request.bad" "$Y" "$W/a.proof" "$ID"
no "changed proof byte" "$W/request.req" "$Y" "$W/a.bad" "$ID"
no "other commitment" "$W/request.req" "$Y" "$W/a.proof" "${ID%?}$([ "${ID: -1}" = 0 ] && echo 1 || echo 0)"
[ $bad -eq 0 ] && echo "SUCCESS: accepted the honest proof, rejected every change" || { echo "FAILED: a change was not rejected"; exit 1; }
