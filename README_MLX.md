# FlashProve on Apple silicon

Proves a request on a bundled synthetic model on a Mac, on the GPU through MLX and Metal, and checks it
with the stock `cp_verify`. What was measured, and the limits: [MLX_PORT_STATUS.md](MLX_PORT_STATUS.md).
Use it on synthetic or public weights only.

## Setup

Xcode is not needed: Metal kernels compile at run time.

```sh
brew install gcc coreutils findutils gnu-sed gawk grep
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx mlx-lm numpy torch
verifier/build.sh
```

`verifier/build.sh` needs GNU tools and g++ first on `PATH`, under the names `g++` and `gcc`.

## Everything at once

```sh
mlx/demo.sh ~/fpwork/demo
```

From synthetic weights: reference, commitment, proof, the stock `cp_verify` in a separate process, and
four rejected changes. It ends with `SUCCESS` and exit 0.

## Step by step

```sh
K=prover/examples/target_small_f2_roomy; W=~/fpwork/mine
mlx/fpmlx prepare $K $W                     # the reference, the matrix of cells, a request
mlx/fpmlx commit  $K $W                     # prints: commitment ID: ...
mlx/fpmlx prove   $K $W -id ID -out $W/a.proof
mlx/fpmlx verify  $K $W/ivk $W/request.req $W/a.proof.y $W/a.proof -id ID
```

`verify` is the stock `bin/cp_verify`. Its exit code and last line are the answer, `YES` here. Pass the ID
that `commit` printed, or one from a party you trust; never one taken from files that came with a proof.
Under a clipped-gap kit such as `target_small_f2_gap`, also pass the policy you expect:
`-policy clipped-gap,C,THETA,CAP,S,SH`, and `prepare` needs `-tokens FILE`, since that kit has no
`own_choices.req`. A `YES` here says nothing of a provider's model, and it does not show that weights are
hidden: see `guide/what-is-proved.md` and `guide/weight-hiding.md`.

`DIR/isk` is secret. `seed` re-derives the random entries of the tables. `seed.spent` counts the columns
that proofs have opened: see the limits in the status file. Do not copy or restore that directory: a
restored count lets proofs pass the table's limit. This `isk` is not the directory of `cp_commit`, whose
`isk.txt` and stored tables `prover/` reads. It holds what `fp_prove` needs to rebuild a leaf table.

## Tests

```sh
mlx/run_tests.sh ~/fpwork/tests             # all of them, one line each; ends with ALL PASS
```

`mlx/run_tests.sh` builds and runs every test below; its last line says how many failed.
