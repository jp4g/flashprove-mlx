#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Builds cp_prove and cp_commit for Apple's GPUs from the unchanged CUDA sources: see
# prover/metal/README.md. Needs Homebrew's llvm (clang, llvm-objcopy) and the Command Line Tools.
# usage: prover/metal/build.sh [serving|tests] [cp_prove|cp_commit|all]
# Output: prover/build/metal/<serving|tests>/{cp_prove,cp_commit,cp_verify}.
set -euo pipefail
export PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/opt/findutils/libexec/gnubin:$PATH
cd "$(dirname "$0")/.."
form=${1:-serving} what=${2:-all}
out=build/metal/$form
mkdir -p "$out"
LLVM=${LLVM:-/opt/homebrew/opt/llvm}
metal/build_runtime.sh build/metal
flags=(-DMTL_BUILD=1)
[[ $form == serving ]] && flags+=(-DSERVING_BUILD=1)
fw=(-framework Metal -framework Foundation -framework Accelerate -Wl,-stack_size,0x4000000)
keeper() {
  metal/build_unit.sh "$out/hf_keeper_c.o" hash_commitment/hf_keeper.cu "${flags[@]}" -DCHAIN_HOST_ENGINE=1
  local door
  door=$(nm "$out/hf_keeper_c.o" | awk '$2 == "T" && $3 ~ /^__ZN2hf14hf_keeper_unitE/ { print $3 }')
  ld -r -exported_symbol "$door" -o "$out/hf_keeper.o" "$out/hf_keeper_c.o"
  for n in 2 3; do
    "$LLVM/bin/llvm-objcopy" --redefine-sym "$door=${door/14hf_keeper_unit/16hf_keeper_unit_$n}" "$out/hf_keeper.o" "$out/hf_keeper_$n.o"
  done
}
prove() {
  keeper &
  local k=$!
  metal/build_unit.sh "$out/cp_prove.o" prover/cp_prove.cu "${flags[@]}" -DCHAIN_HOST_ENGINE=1 &
  local c=$!
  wait "$k"; wait "$c"
  clang++ -o "$out/cp_prove" "$out/cp_prove.o" "$out"/hf_keeper{,_2,_3}.o build/metal/libmetal_runtime.a "${fw[@]}"
}
commit() {
  # cp_commit holds the verifier's code too: only its entry point stays visible. It states the same
  # source digest as ../bin/cp_verify, which verifier/build.sh computed from these sources.
  local word
  word=$(cd ../verifier/src && export LC_ALL=C && eval "$(grep '^build_word=' ../build.sh)" && echo "$build_word")
  ( cd ../verifier/src && "$LLVM/bin/clang++" -O2 -std=c++17 -pthread -w -I. -I../../blocks -I../../prover -I../../prover/metal/include/compat -include ../../prover/metal/include/metal_compat.h -DBLK_FIELD=2 -DSERVING_BUILD=1 -DDIGEST_WORDS=8 -DVF_SOURCE_DIGEST="\"$word\"" -DCP_VERIFY_AS_A_UNIT -isysroot "$(xcrun --show-sdk-path)" -c cp_verify.cpp -o "../../prover/$out/cp_configured_c.o" )
  local door
  door=$(nm "$out/cp_configured_c.o" | awk '$2 == "T" && $3 ~ /^__Z18cp_with_configured/ { print $3 }')
  ld -r -exported_symbol "$door" -o "$out/cp_configured.o" "$out/cp_configured_c.o"
  metal/build_unit.sh "$out/cp_commit.o" hash_commitment/hf_commit.cu "${flags[@]}" -I../verifier/src -DHC_RANGE_N_MOST=40
  clang++ -o "$out/cp_commit" "$out/cp_commit.o" "$out/cp_configured.o" build/metal/libmetal_runtime.a "${fw[@]}"
}
case $what in
  link)
    clang++ -o "$out/cp_prove" "$out/cp_prove.o" "$out"/hf_keeper{,_2,_3}.o build/metal/libmetal_runtime.a "${fw[@]}"
    [[ -f $out/cp_commit.o ]] && clang++ -o "$out/cp_commit" "$out/cp_commit.o" "$out/cp_configured.o" build/metal/libmetal_runtime.a "${fw[@]}"
    ;;
  cp_prove) prove ;;
  cp_commit) commit ;;
  all) prove & p=$!; commit & c=$!; wait $p; wait $c ;;
esac
cp -f ../bin/cp_verify "$out/cp_verify"
echo "built $out"
