#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
# usage: build_runtime.sh <out dir>. Builds libmetal_runtime.a. Run from prover/.
set -euo pipefail
o=$1; mkdir -p "$o"
LLVM=${LLVM:-/opt/homebrew/opt/llvm}
python3 - "$o" <<'PY'
import sys
o=sys.argv[1]
def emit(name,path):
    b=open(path,'rb').read()+b'\0'
    return f'extern const char {name}[]={{'+','.join(str(x) for x in b)+'};\n'
open(o+'/mtl_texts.cpp','w').write('extern "C++" {\n'+'extern const char mtl_prelude_text[];extern const char mtl_builtin_text[];\n'+emit('mtl_prelude_text','metal/metal_prelude.metal')+emit('mtl_builtin_text','metal/runtime/builtin.metal')+'}\n')
PY
clang++ -std=c++17 -O2 -c "$o/mtl_texts.cpp" -o "$o/mtl_texts.o"
clang++ -std=c++17 -O2 -fobjc-arc -w -Imetal/include -Imetal/include/compat -Imetal/runtime -x objective-c++ -include cuda_runtime.h -D__CUDA_METAL_HOST_ONLY__ -c metal/runtime/metal_runtime.mm -o "$o/metal_runtime.o"
rm -f "$o/libmetal_runtime.a"
ar rcs "$o/libmetal_runtime.a" "$o/metal_runtime.o" "$o/mtl_texts.o"
