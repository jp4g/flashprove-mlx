#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# usage: build_unit.sh <out.o> <source.cu> [clang flags...]
# One translation unit for the Metal backend: device pass to PTX, ptx2msl.py to a pack, host pass
# with the pack embedded as the unit's GPU binary. Run from prover/.
set -euo pipefail
out=$1 src=$2; shift 2
LLVM=${LLVM:-/opt/homebrew/opt/llvm}
SDK=$(xcrun --show-sdk-path)
common=(-x cuda -nocudainc -nocudalib --cuda-gpu-arch=sm_90a -std=c++17 -O3 -I. -Imetal/include -Imetal/include/compat -I../blocks
  -DBLK_FIELD=2 -DDIGEST_WORDS=8 -DCCCL_DISABLE_NVTX -include cuda_runtime.h -isysroot "$SDK" -Wno-unknown-cuda-version -w
  -Xclang -target-sdk-version=13.2 --cuda-feature=+ptx80 -fno-jump-tables -ffp-contract=off "$@")
ptx=$out.ptx pack=$out.pack
"$LLVM/bin/clang++" "${common[@]}" --cuda-device-only -S "$src" -o "$ptx"
python3 metal/ptx2msl.py "$ptx" "$pack"
"$LLVM/bin/clang++" "${common[@]}" --cuda-host-only -Xclang -fcuda-include-gpubinary -Xclang "$pack" -c "$src" -o "$out"
