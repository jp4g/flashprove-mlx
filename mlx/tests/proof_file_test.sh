#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# The proof as a file: the stock verifier reads every byte of it, so it must refuse the file with one byte
# more or less, with parts moved, and with a bit flipped anywhere it is tried. A file written and read back
# stays accepted, and a second proof of the request is accepted too and differs.
# usage: mlx/tests/proof_file_test.sh <work directory of mlx/demo.sh>
set -uo pipefail
cd "$(dirname "$0")/../.."
W=$1; K=prover/examples/target_small_f2_roomy; T=$W/file; mkdir -p "$T"; bad=0
ID=$(sed -n 's/^commitment ID: //p' "$W/commit.log"); Y=$W/a.proof.y
v() { bin/cp_verify "$K" "$W/ivk" "$W/request.req" "$2" "$1" -id "$ID" > "$T/v.txt" 2>&1; echo $?; }
no() { local c; c=$(v "$T/x.proof" "$Y"); echo "$1: exit $c, $(tail -1 "$T/v.txt")"; [ "$c" = 1 ] || bad=1; }
S=$(wc -c < "$W/a.proof" | tr -d " ")
cp "$W/a.proof" "$T/copy.proof"; c=$(v "$T/copy.proof" "$Y"); echo "a copy: exit $c, $(tail -1 "$T/v.txt")"; [ "$c" = 0 ] || bad=1
head -c $((S - 1)) "$W/a.proof" > "$T/x.proof"; no "a byte less"
head -c 13 "$W/a.proof" > "$T/x.proof"; no "the head alone"
{ cat "$W/a.proof"; printf '\0'; } > "$T/x.proof"; no "a byte more"
python3 - "$W/a.proof" "$T/x.proof" <<'P'
import sys
b = open(sys.argv[1], 'rb').read(); h = len(b) // 2
open(sys.argv[2], 'wb').write(b[:64] + b[h:h + 4096] + b[64 + 4096:h] + b[64:64 + 4096] + b[h + 4096:])
P
no "two parts swapped"
for at in 0 12 13 100 $((S / 4)) $((S / 2)) $((3 * S / 4)) $((S - 1)); do
  python3 - "$W/a.proof" "$T/x.proof" "$at" <<'P'
import sys
b = bytearray(open(sys.argv[1], 'rb').read()); b[int(sys.argv[3])] ^= 1; open(sys.argv[2], 'wb').write(b)
P
  no "a bit flipped at $at"
done
mlx/fpmlx prove "$K" "$W" -id "$ID" -out "$T/b.proof" > /dev/null 2>&1 || { echo "FAILED: the second proof"; exit 1; }
c=$(v "$T/b.proof" "$T/b.proof.y"); echo "a second proof: exit $c, $(tail -1 "$T/v.txt")"; [ "$c" = 0 ] || bad=1
cmp -s "$W/a.proof" "$T/b.proof" && { echo "FAILED: two proofs are equal: the masks repeat"; bad=1; }
[ $bad -eq 0 ] && echo "proof_file_test: all hold" || { echo "proof_file_test: FAILED"; exit 1; }
