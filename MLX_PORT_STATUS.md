# MLX port: status

Branch `mlx-port`. **The chain works: a request on a bundled synthetic model is committed and proved on the
Apple GPU through MLX and Metal, and the unmodified `bin/cp_verify` answers `YES`.** A proof takes 13 s, with
about 3.6 s of user CPU. What stays on the host is listed under the limits. SmolLM2 is no longer the end goal,
by the owner's decision of 2026-11-05, and nothing of it is built. Everything below is measured on an Apple
M5 Pro (64 GB, macOS 27), MLX 0.32.4.

## The result

`mlx/run_tests.sh` passes all 21 groups (`ALL PASS`, exit 0). Its `mlx/demo.sh` ran from an empty directory:
reference, placing, commitment, proof, the stock `cp_verify` in a separate process, four rejected changes.
Commitment ID `77de866cd5750b23a288196ca878e7dc2638c23a84c1f3bc2718766c458ecf4f`, exit 0, last line `YES`.

`mlx/bench.py`, on `target_small_f2_roomy`, median (lowest-highest):

| Step | Runs | Real s | User s | Sys s | Peak footprint GB |
|---|---|---|---|---|---|
| commit | 3 | 110.75 (79.40-175.70) | 5.44 (5.36-5.63) | 4.40 (4.18-4.55) | 50.00 (49.33-50.00) |
| prove | 7 | 12.99 (12.29-14.64) | 3.60 (3.51-3.66) | 2.66 (2.56-2.67) | 31.09 (31.09-31.27) |
| verify | 7 | 0.59 (0.58-0.69) | 1.11 (1.01-1.21) | 0.02 (0.01-0.02) | 0.01 (0.01-0.01) |

Proof size: 4,147,588 to 4,161,260 bytes over 7 proofs, all accepted. Each commit has a fresh seed, and its
time is mostly the nonce search, which is why it varies. Before the move to Metal, in single runs: commit about
4 min, prove 51.0 s with 124.2 s of user CPU, peak footprint 63.4 GB.

**Measure with the machine awake.** Left idle, this Mac goes into `Maintenance Sleep` and GPU work stalls: one
commit took 439 s with the GPU at 2%. Run benchmarks under `caffeinate -dimsu`.

`fp_prove` and `fp_commit` limit MLX's cache of freed buffers to 4 GB, and the encoding is evaluated every four
layers. The peak of `fp_commit` is mostly two tables of 19 GB each, which `make_entry` needs at once.

`cp_verify`'s lines on that run: `covered: r=15 e=64 T=64`, `commitment: 49 closeness rounds of the one-time
files rest on another source than the bound within a third of the distance; the table has a record: the binding
of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken as given`, `proof of work: up
to 32 bits counted in 38 rounds; with it the least round has 128.00 bits, without it 96.23 bits`, these bits being
no security level (`guide/known-limits.md`). This `YES` is `cp_verify`'s, on a small model with synthetic weights.
It says nothing of a provider's model, and it does not show that the weights are hidden
(`guide/weight-hiding.md`).

Refused, each exit 1: a changed token (`NO: reply-differs`), a changed request, changed proof bytes, a cut or
extended file, swapped parts (`NO: proof-refused`), another commitment ID (`NO: model-statement`), a wrong hidden
value and a proof made with a faulty kernel (`NO: proof-refused`). Two proofs of one request are both accepted and
differ.

## GPU and CPU agreement

- **A whole proof:** under fixed coins, the kernels and the plain host paths give the same 4.1 MB of proof,
  byte for byte, and a planted fault in one product level changes the proof, which `cp_verify` refuses
  (`whole_proof_test.sh`, with the test program `fp_prove_fixed`). Encoding, the trees and the nonce search
  stay on the GPU in both runs; `engine_test` compares those with `EngineCpu`.
- **A commitment:** with one seed, the Metal `fp_commit` writes every file byte for byte as the CPU path did
  (`diff -r` of `ivk`, same commitment ID).
- **Each kernel:** against plain host code, see below.

## Tests against the brief

`mlx/run_tests.sh` builds and runs all of these; `paths_test.sh`, `proof_file_test.sh` and
`whole_proof_test.sh` are in `mlx/tests/`.

| The brief asks for | Test | What it shows |
|---|---|---|
| finite-field operations | `field_test` | add, sub, mul, E mul equal `blocks/field` on about 2^20 inputs each |
| integer matrix operations | `forward_test`, `kernels_gpu_test` | 191 tensors, 5,246,656 values equal `model.py`; combinations, column sums, `g^T M` equal host code |
| GPU/CPU agreement | `whole_proof`, `commit_again`, `psc_gpu_test`, `form_gpu_test`, `coins_gpu_test`, `kernels_gpu_test`, `ff_keys_test`, `attn_keys_test`, `head_keys_test`, `engine_test`, `toy_digests`, `segment_digests` | word for word: 288 sumcheck runs, 36 forms, 9 coin draws, 29 kernel checks, every witness cell and key |
| proof serialization | `proof_file_test.sh` | stock `cp_verify` reads the file whole: a copy passes; a byte less or more, the head alone, swapped parts and bits flipped at 8 places are refused |
| commitment consistency | `commit_again`, `demo`, `paths` | one seed, the same files and ID; the proof fits its commitment and no other |
| honest acceptance | `demo`, `paths` | `YES`; `e < T`, `r = T - 1`; `WITHIN LIMITS` under a clipped gap, with one token above C |
| invalid rejection | `demo`, `proof_file`, `tampered_hidden`, `paths` | changed token, request, ID or proof; a wrong hidden value; a policy broken; two tokens above C where one is allowed; `r = e` |
| repeated proof generation | `proof_file_test.sh`, `ledger_test` | a second proof is accepted and differs; the table's count survives runs and stops at its limit |
| library regression | `library` | 37 test programs of `blocks/` with `EngineMlx` in place of `EngineCpu` |

`docs/format/read_proof.py` and `read_key.py` were not run: they need the plan files that
`blocks/frame/handover_*` writes, which `fp_commit` and `fp_prove` do not.

## Limits of this result

- **On the GPU:** the integer forward products; the witnesses of all three kinds of segment, with their
  cells, keys and norm rows; table layout; the encoding; the Merkle trees; the nonce search; the folds and
  their coefficient vectors; sumcheck rounds from 2^13 values up; product trees, leaves and multiplicities
  of lookups; the smallness test; `g^T M`; long coin draws; and the commitment's keys and lookups.
- **Still host code, by design:** the Fiat-Shamir transcript, proof orchestration and arithmetic on a few
  scalars. These are sequential and tiny.
- **Still host code, small:** sumcheck rounds below 2^13 values (no faster on the GPU, #6); the member's
  sumcheck over 2^11 columns; small maps of a few thousand values (`Z`, the rope's table, the head's); the
  head's budget, a few words per position. By profile the host is 13% of the main thread's time, most of it
  those short rounds, the transcript, and copies: `blk::Table` is a host struct of `blocks/`, and MLX copies an
  array that it is given, so a table crosses that line once each way. The rest is waiting for the GPU.
- **The companion is permanent:** `fp_prove` runs a patched build of the verifier beside it, for each
  segment's equation and public parts. They come from about 4,000 lines of walker in `verifier/src/`
  (`mlx/DESIGN_PROVER.md`). Reusing that code means that the prover and the verifier cannot drift apart;
  a rewrite would add a second copy of it, and with it a way to disagree. The companion decides nothing:
  the judge is the stock `cp_verify`, built without those patches.
- **The table's budget:** a leaf table hides 65,535 opened columns, and a proof opens 51, so it allows
  1,285 batches, `make_entry`'s among them. The count is in `<seed file>.spent`, raised and flushed before
  any column is opened, under a lock. `fp_prove` refuses if that file is missing or damaged, and at the
  limit (exit 3). Deleting or restoring the file defeats it: nothing but the operator's care protects it.
- **`isk`:** `fpmlx commit` writes `DIR/isk` with the secret seed and the count. It is not `cp_commit`'s
  directory: `prover/` reads `isk.txt` and stored tables, while `fp_prove` rebuilds the leaf table from the
  seed (about 5 s), so the seed replaces them.
- **Refused, not tested:** `fp_prove` refuses `v >= V` padding, a head matrix in several parts and limbs
  other than five (exit 2; `FP_UNTESTED_PATHS=1` tries them). No bundled kit small enough for this Mac
  reaches them. The full-size kits were not run here.
- **Fixed coins** exist only in the test program `fp_prove_fixed`. Never use it on real weights.

## Design

Native port: `mlx/` holds C++ on `mlx::core` arrays with hand-written Metal kernels
(`mx::fast::metal_kernel`). `blocks/` (proof library) and `verifier/` are used as they are, apart
from the macOS link flags in `verifier/build.sh`. `prover/` (CUDA) is not compiled.

The branch `cuda-shim-parked` holds an earlier CUDA-to-Metal compatibility layer. It is not this
port and nothing here depends on it.

## Done and tested

| Piece | Where | Result |
|---|---|---|
| `cp_verify` on macOS | `verifier/build.sh` | builds with g++ 16; stored example gives `YES` |
| F and E arithmetic | `mlx/kernels/field.h.metal` | `field_test`: add, sub, mul, E mul equal to `blocks/field` on about 2^20 inputs each |
| `EngineMlx`: encoding, BLAKE2b leaves and tree, evaluation, folds | `mlx/src/engine_mlx.h`, `mlx_kernels.cpp` | `engine_test`: root, openings, evaluation, fold rounds, folded rows equal to `EngineCpu` |
| Whole toy protocols | `mlx/tests/toy_mlx.cpp`, `segments_mlx.cpp` | accepted, proof digests byte-identical to the CPU engine at seed 42 |
| Native commitment | `mlx/src/fp_commit.cpp`, `index_mlx.h`, `mlx/tools/place_weights.py` | on `target_small_f2_roomy`: key and leaf files pass the unmodified `cp_verify` (`PART PASSED`); 4 tampered files refused (`key-refused`, `entry-refused`). On Metal since; see the table above |
| Verifier companion | `mlx/build_companion.sh`, `mlx/tools/make_shadow.py`, `mlx/tests/stream_test.py` | the stored example proof, streamed 4 KB at a time through shared memory: `YES`, 5 segment equations exported |
| Masked sumcheck prover | `mlx/src/psc_prover.h`, `mlx/tests/psc_test.cpp` | 66 configurations (equality, plain, mixed rounds, shown values, bit orders): the claim `Left(C; S0; K) = 0` holds |
| Integer forward pass, with experts | `mlx/src/integer_forward.cpp` | `forward_test`: 191 tensors, 5,246,656 values equal to `model.py` on `target_small_f2_roomy` synthetic weights; one altered reference value makes it fail |

## Library tests with EngineMlx in place of EngineCpu (`mlx/run_library_tests.sh`)

37 test programs, all pass, rerun after the move to Metal. `EngineMlx` runs on the GPU in the parent process. Tests that `fork()`
to check that a misuse ends the process run the engine's host path in the child (Metal does not
survive a fork), so those children exercise the engine's guards, not its kernels.

## Not done

- SmolLM2: dropped as the end goal. A separate verifier variant would be needed for it.
- The full-size kits (`target_full_f2`, d = 5120, 48 layers) were not run.
