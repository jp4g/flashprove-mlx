# FlashProve on Apple silicon

Proves a request on the bundled synthetic model on a Mac, and checks it with the stock `cp_verify`:

```
mlx/demo.sh ~/fpwork/demo
```

It needs `.venv` with `mlx`, Homebrew's `gcc`, `llvm` and GNU tools. See `MLX_PORT_STATUS.md` for what
was measured and for the limits: the work that grows with the model runs on Metal, and witness assembly is still host code.

# FlashProve on MLX: reproducing the results

The port is unfinished: see [MLX_PORT_STATUS.md](MLX_PORT_STATUS.md). These commands reproduce what
exists. It cannot yet commit to or prove a model.

## Setup (Apple silicon)

Xcode is not needed: Metal kernels compile at run time.

```sh
brew install gcc coreutils findutils gnu-sed gawk grep
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx mlx-lm numpy torch
```

`verifier/build.sh` needs GNU tools and g++ first on `PATH`, under the names `g++` and `gcc`.

## Verifier

```sh
verifier/build.sh
bin/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk examples/accepted-proof/example.req examples/accepted-proof/example.y examples/accepted-proof/example.proof -id "$(cat examples/accepted-proof/ID.txt)"
```

## Tests

```sh
mlx/build.sh field_test && mlx/build/field_test
mlx/build.sh engine_test && mlx/build/engine_test
for t in toy_cpu toy_mlx segments_cpu segments_mlx; do mlx/build.sh $t && mlx/build/$t 8 42; done
.venv/bin/python mlx/tests/make_reference.py prover/examples/target_small_f2_roomy /tmp/ref
mlx/build.sh forward_test && mlx/build/forward_test /tmp/ref
for t in psc_gpu_test form_gpu_test coins_gpu_test kernels_gpu_test; do mlx/build.sh $t && mlx/build/$t; done
mlx/run_library_tests.sh
```

With the same seed, `toy_cpu` and `toy_mlx` print the same proof digest, as do `segments_cpu` and
`segments_mlx`.
