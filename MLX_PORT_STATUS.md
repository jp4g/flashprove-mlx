# MLX port: status

Branch `mlx-port`. **Milestone 1's chain works: a request on the bundled synthetic model is proved by
the native prover and the unmodified `bin/cp_verify` answers `YES`. The heavy proof loops now run on
Metal, but not all work that grows with the model does: witness assembly is still host code. A proof
takes 21 s, with about 11 s of user CPU time.** What is still host code is listed under the limits. SmolLM2 is no longer the end goal, by the owner's decision of 2026-11-05, and
nothing of it is built. Everything below is measured on an Apple M5 Pro (64 GB, macOS 27), MLX 0.32.4.

## The result

`mlx/demo.sh` runs it all from synthetic weights: reference, placing, commitment, proof, the stock
`cp_verify` in a separate process, four rejected changes. Last run, from an empty directory, commitment
ID `4c317f4c8a3a9da11e9ceb0e288aab5cd435f0f3e43582d1c025a3dff9efee1e`:

| Step | Before the move to Metal | Now |
|---|---|---|
| Commitment (`fp_commit`) | about 4 min | 87 s and 118 s in two runs: the key file took 54 s and 91 s, with different seeds; much of the key file is the nonce search |
| Proof (`fp_prove`, leaf table rebuilt, equation test on) | 51.0 s | 21.3 s |
| CPU time of a proof | 124.2 s user, 10.6 s system | 10.8 s user, 6.1 s system |
| Segments | G0 9.2 s, A0 4.8 s, G1 10.3 s, A1 5.6 s, H 6.6 s | G0 3.7 s, A0 1.6 s, G1 3.8 s, A1 1.5 s, H 1.9 s |
| Proof size | 4,158,700 bytes | 4,153,500 bytes (it differs from proof to proof) |
| `cp_verify` | 0.71 s, exit 0, `YES` | 0.69 s, exit 0, `YES` |
| Memory | peak footprint 63.4 GB | peak footprint 37.6 GB; MLX's peak in use 29.2 GB |

These are single runs, not averages. The old footprint was mostly MLX's cache of freed buffers (34 GB at
the end of a proof); `fp_prove` and `fp_commit` now limit that cache to 4 GB (`FP_MLX_CACHE_GB` for
`fp_prove`). `fp_commit` still peaked at 61 GB in the one run measured.

`cp_verify`'s lines on the last run: `covered: r=15 e=64 T=64`, `commitment: 49 closeness rounds of the
one-time files rest on another source than the bound within a third of the distance; the table has a
record: the binding of the weights rests on the theorem on codes (ePrint 2025/2055, Theorem 4.6), taken
as given`, `proof of work: up to 32 bits counted in 38 rounds; with it the least round has 128.00 bits,
without it 96.23 bits`, these bits being no security level (`guide/known-limits.md`). This `YES` is
`cp_verify`'s, on a small model with synthetic weights. It says nothing of a provider's model, and it
does not show that the weights are hidden (`guide/weight-hiding.md`).

Refused in the demo, each exit 1: a changed token (`NO: reply-differs`), a changed request and a changed
proof byte (`NO: proof-refused`), another commitment ID (`NO: model-statement`). By hand on the same
proof, also exit 1 and `NO: proof-refused`: one bit flipped early, in the middle and late, and a cut
proof. A second proof of the same request is accepted too and differs from byte 13 on.

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
| GPU/CPU agreement | `whole_proof`, `commit_again`, `psc_gpu_test`, `form_gpu_test`, `coins_gpu_test`, `kernels_gpu_test`, `ff_keys_test`, `attn_keys_test`, `head_keys_test`, `engine_test`, `toy_digests`, `segment_digests` | word for word: 288 sumcheck runs, 36 forms, 9 coin draws, 21 kernels, every witness cell and key |
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
- **Still host code, small:** sumcheck rounds below 2^13 values (#6: no faster on the GPU); the member's
  sumcheck over 2^11 columns; `FFProver`'s and the head's small maps, 14 jobs of at most 2^18 values and two
  of 2^13 values in 107 maps, against device jobs of up to 2^24; the head's budget, a few words per position.
  `blk::Table` is a host struct of `blocks/`, so each laid-out table is copied back once.
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
