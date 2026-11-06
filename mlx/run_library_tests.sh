#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# Builds the proof library's tests that use EngineCpu with EngineMlx in its place, runs each, and
# prints one line per test. usage: mlx/run_library_tests.sh [substring of a test's path]
cd "$(dirname "$0")/.."
M=$(.venv/bin/python -c 'import mlx.core,os;print(os.path.dirname(mlx.core.__file__))')
O=mlx/build/lib; mkdir -p "$O"
[[ -f $O/k.o ]] && [[ $O/k.o -nt mlx/src/mlx_kernels.cpp ]] || clang++ -std=c++17 -O2 -w -Imlx/src -Iblocks -I"$M/include" -include mlx/src/macos_compat.h -DBLK_FIELD=2 -DFPMLX_KERNEL_DIR="\"$PWD/mlx/kernels\"" -c mlx/src/mlx_kernels.cpp -o "$O/k.o"
tests=$(grep -rlE 'EngineCpu|engine_cpu\.h|_world\.h|whole_runs\.h|req_cases' blocks --include='*_test.cpp' | grep -v must_not_build | grep -v exact_test | sort)
one() {
  t=$1; n=$(echo "$t" | tr '/.' '__')
  if ! clang++ -std=c++17 -O2 -w -Imlx/tests/as_cpu -Imlx/src -Iblocks -I"$M/include" -include mlx/src/macos_compat.h -DBLK_FIELD=2 -DFPMLX_KERNEL_DIR="\"$PWD/mlx/kernels\"" "$t" "$O/k.o" -L"$M/lib" -lmlx -Wl,-rpath,"$M/lib" -framework Metal -framework Foundation -framework Accelerate -o "$O/$n" 2>"$O/$n.err"; then
    echo "NOBUILD $t: $(grep -m1 error "$O/$n.err" | cut -c1-120)"; return; fi
  (cd blocks && timeout 300 "../$O/$n" >"../$O/$n.out" 2>&1); s=$?
  [ $s = 0 ] && echo "PASS $t" || echo "FAIL($s) $t: $(tail -1 "$O/$n.out" | cut -c1-120)"
}
export -f one; export M O
echo "$tests" | grep -- "${1:-}" | xargs -P 8 -I{} bash -c 'one {}'
