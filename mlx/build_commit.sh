#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Builds mlx/build/fp_commit. It uses cp_verify's code as a unit, for the public files' plans.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/opt/homebrew/opt/coreutils/libexec/gnubin:/opt/homebrew/opt/findutils/libexec/gnubin:$PATH
M=$(.venv/bin/python -c 'import mlx.core,os;print(os.path.dirname(mlx.core.__file__))')
O=mlx/build; mkdir -p $O
word=$(cd verifier/src && export LC_ALL=C && eval "$(grep '^build_word=' ../build.sh)" && echo "$build_word")
if [[ ! -f $O/cp_configured.o || verifier/src/cp_verify.cpp -nt $O/cp_configured.o ]]; then
  (cd verifier/src && clang++ -O2 -std=c++17 -pthread -w -I. -I../../blocks -I../../prover -include ../../mlx/src/macos_compat.h -DBLK_FIELD=2 -DSERVING_BUILD=1 -DDIGEST_WORDS=8 -DVF_SOURCE_DIGEST="\"$word\"" -DCP_VERIFY_AS_A_UNIT -c cp_verify.cpp -o ../../$O/cp_configured_c.o)
  door=$(nm $O/cp_configured_c.o | awk '$2 == "T" && $3 ~ /^__Z18cp_with_configured/ { print $3 }')
  ld -r -exported_symbol "$door" -o $O/cp_configured.o $O/cp_configured_c.o
fi
for f in mlx_kernels fp_commit; do
  clang++ -std=c++17 -O2 -w -pthread -Imlx/src -Iblocks -Iverifier/src -I"$M/include" -include mlx/src/macos_compat.h -DBLK_FIELD=2 -DDIGEST_WORDS=8 -DFPMLX_KERNEL_DIR="\"$PWD/mlx/kernels\"" -c mlx/src/$f.cpp -o $O/$f.o &
done
wait
clang++ -o $O/fp_commit $O/fp_commit.o $O/mlx_kernels.o $O/cp_configured.o -L"$M/lib" -lmlx -Wl,-rpath,"$M/lib" -framework Metal -framework Foundation -framework Accelerate
