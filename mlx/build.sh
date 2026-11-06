#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# usage: mlx/build.sh <test name>: builds mlx/tests/<name>.cpp against MLX and blocks/ into mlx/build/.
set -euo pipefail
cd "$(dirname "$0")/.."
M=$(.venv/bin/python -c 'import mlx.core,os;print(os.path.dirname(mlx.core.__file__))')
mkdir -p mlx/build
clang++ -std=c++17 -O2 -w -Imlx/src -Iblocks -I"$M/include" -DBLK_FIELD=2 -include mlx/src/macos_compat.h -DFPMLX_KERNEL_DIR="\"$PWD/mlx/kernels\"" \
  "mlx/tests/$1.cpp" mlx/src/mlx_kernels.cpp mlx/src/integer_forward.cpp -L"$M/lib" -lmlx -Wl,-rpath,"$M/lib" -framework Metal -framework Foundation -framework Accelerate -o "mlx/build/$1"
