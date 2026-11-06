#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Every test of the port, one line each; exit 1 if any fails. About half an hour.
# usage: mlx/run_tests.sh [empty work directory]
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH
W=${1:-$HOME/fpwork/tests}; mkdir -p "$W"; K=prover/examples/target_small_f2_roomy; fails=0
mlx/build_commit.sh && mlx/build_prove.sh && mlx/build_companion.sh > /dev/null || { echo "FAIL build"; exit 1; }
t() { # name, command...
  local n=$1; shift
  # A pass is exit 0 and a last line that says so: a script that dies on a shell error must not pass.
  if "$@" > "$W/$n.log" 2>&1 && ! grep -qE 'unbound variable|command not found|syntax error' "$W/$n.log" && [ -n "$(tail -1 "$W/$n.log")" ]; then echo "PASS $n: $(tail -1 "$W/$n.log" | cut -c1-90)"; else echo "FAIL($?) $n: $(tail -1 "$W/$n.log" | cut -c1-90)"; fails=$((fails + 1)); fi
}
for n in field_test engine_test psc_test psc_gpu_test form_gpu_test coins_gpu_test kernels_gpu_test ledger_test forward_test ff_keys_test attn_keys_test head_keys_test toy_cpu toy_mlx segments_cpu segments_mlx; do
  mlx/build.sh $n > "$W/build_$n.log" 2>&1 || { echo "FAIL build of $n"; fails=$((fails + 1)); }
done
for n in field_test engine_test psc_test psc_gpu_test form_gpu_test coins_gpu_test kernels_gpu_test ledger_test; do t $n mlx/build/$n; done
t demo mlx/demo.sh "$W/demo"
R=$W/demo/ref
t forward_test mlx/build/forward_test "$R"
t ff_keys_test mlx/build/ff_keys_test "$R" $K/policy.txt
t attn_keys_test mlx/build/attn_keys_test "$R" $K/policy.txt
t head_keys_test mlx/build/head_keys_test "$R" $K/policy.txt "$W/demo/request.req" "$W/head_check.req"
same() { local a b; a=$(mlx/build/$1 8 42 | grep -i digest); b=$(mlx/build/$2 8 42 | grep -i digest); echo "$a"; [ -n "$a" ] && [ "$a" = "$b" ]; }
t toy_digests same toy_cpu toy_mlx
t segment_digests same segments_cpu segments_mlx
commit_again() { # the same seed gives the same commitment, byte for byte
  local C=$W/again; mkdir -p "$C"; cp "$W/demo/isk/seed" "$C/seed"
  mlx/build/fp_commit $K "$R/matrix.i8" "$C/ivk" "$C/seed" > "$C/log" && diff -r "$C/ivk" "$W/demo/ivk" && echo "the same files"
}
t commit_again commit_again
hidden() { # a wrong hidden value breaks the segment's equation, and the stock verifier refuses the proof
  local id; id=$(sed -n 's/^commitment ID: //p' "$W/demo/commit.log")
  FP_SELF_CHECKS=1 FP_TAMPER_HIDDEN=3 mlx/fpmlx prove $K "$W/demo" -id "$id" -out "$W/tampered.proof" > "$W/tampered.out" 2>&1
  grep -qE 'DOES NOT HOLD' "$W/tampered.proof.log" "$W/tampered.out" || { echo "the equation test saw nothing"; return 1; }
  mlx/fpmlx verify $K "$W/demo/ivk" "$W/demo/request.req" "$W/tampered.proof.y" "$W/tampered.proof" -id "$id" > "$W/tampered.v" 2>&1; local c=$?
  echo "the equation does not hold; cp_verify exit $c, $(tail -1 "$W/tampered.v")"; [ $c -eq 1 ]
}
t tampered_hidden hidden
t proof_file mlx/tests/proof_file_test.sh "$W/demo"
t paths mlx/tests/paths_test.sh "$W/demo" "$W/gap"
t whole_proof mlx/tests/whole_proof_test.sh "$W/demo"
t library sh -c "mlx/run_library_tests.sh > '$W/library.txt' 2>&1; n=\$(grep -c '^PASS' '$W/library.txt'); m=\$(grep -vc '^PASS' '$W/library.txt'); echo \"\$n passed, \$m others\"; [ \"\$m\" = 0 ]"
[ $fails -eq 0 ] && echo "ALL PASS" || { echo "$fails FAILED"; exit 1; }
