#!/bin/bash
# usage: metal/run_test.sh <name of metal/tests/*.cu> [args]: builds and runs one GPU test.
set -euo pipefail
cd "$(dirname "$0")/.."
n=$1; shift
metal/build_runtime.sh build/metal
mkdir -p build/metal/tests
metal/build_unit.sh "build/metal/tests/$n.o" "metal/tests/$n.cu" >/dev/null
clang++ -o "build/metal/tests/$n" "build/metal/tests/$n.o" build/metal/libmetal_runtime.a -framework Metal -framework Foundation -framework Accelerate
"build/metal/tests/$n" "$@"
