#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Builds mlx/build/fp_prove. build_commit.sh makes cp_configured.o and mlx_kernels.o.
set -euo pipefail
cd "$(dirname "$0")/.."
M=$(.venv/bin/python -c 'import mlx.core,os;print(os.path.dirname(mlx.core.__file__))')
O=mlx/build
for f in fp_prove integer_forward; do
  clang++ -std=c++17 -O2 -w -pthread -Imlx/src -Iblocks -Iverifier/src -I"$M/include" -include mlx/src/macos_compat.h -DBLK_FIELD=2 -DDIGEST_WORDS=8 -DFPMLX_KERNEL_DIR="\"$PWD/mlx/kernels\"" -c mlx/src/$f.cpp -o $O/$f.o &
done
clang++ -std=c++17 -O2 -w -pthread -Imlx/src -Iblocks -Iverifier/src -I"$M/include" -include mlx/src/macos_compat.h -DBLK_FIELD=2 -DDIGEST_WORDS=8 -DFPMLX_TEST_FIXED_COINS -DFPMLX_KERNEL_DIR="\"$PWD/mlx/kernels\"" -c mlx/src/fp_prove.cpp -o $O/fp_prove_fixed.o &
wait
# fp_prove_fixed: a test's program, with the same masks in every proof. Never for real weights.
clang++ -o $O/fp_prove_fixed $O/fp_prove_fixed.o $O/integer_forward.o $O/mlx_kernels.o $O/cp_configured.o -L"$M/lib" -lmlx -Wl,-rpath,"$M/lib" -framework Metal -framework Foundation -framework Accelerate
clang++ -o $O/fp_prove $O/fp_prove.o $O/integer_forward.o $O/mlx_kernels.o $O/cp_configured.o -L"$M/lib" -lmlx -Wl,-rpath,"$M/lib" -framework Metal -framework Foundation -framework Accelerate
