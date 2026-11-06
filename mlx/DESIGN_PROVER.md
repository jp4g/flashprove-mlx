# Native prover on MLX: design

How `fp_prove` makes a request's proof without `prover/`'s CUDA. The sources of truth are
`docs/format/11-*.md`, `blocks/iop/req.h` and `verifier/src/`.

## Split of the work

`blk::req_prove()` (`blocks/iop/req.h`) runs the frame: per segment `first`, `later`, `close`, then
`finish`. It calls a `ReqSubProver`: `tables(link, moment)`, `mask_row(link)`, `plain(link)`,
`at_start()`, `run(P, frame, moment, sent)`. Beyond the first chain: `member_first_prove`, the seal,
`lin_shared_*`, `lin_second_*` (see `blocks/frame/check.h` for the order, and
`prover/prover/request_part.h`, `batch_opening.h`).

What the sub-prover must produce per segment: the witness arrays (mask row, `M`mu, `U3`, `ZZ`, `BZ`)
and every scalar (hidden values, sumcheck messages). Nothing is tested during the rounds, and one
equation per segment closes it.

## The verifier as a companion

A segment's equation (`blk::ReqEquation`: rank-one terms on arrays, coefficients on the mask row,
target) is a function of public data and the transcript. It is built by about 4,000 lines of walker
in `verifier/src/` (`vf_layers_check.inc`, `vf_attention_check.h`, `vf_output_head_check.inc`,
`vf_smallness_test_members.inc`). The prover does not rewrite them.

`fp_prove` starts a companion: the unmodified `verifier/src/` sources built with shadow copies of
`blocks/iop/message.h` and `blocks/iop/req.h` placed first on the include path.

- Shadow `Reader`: in a marked file, `take(n)` waits until the prover has written that far in a
  shared mapping. So the companion walks the proof as it is being written.
- Shadow `ReqVerifier::close()`: writes the equation to the shared mapping, for the prover's frame.
- Key and leaf stages of the companion run normally.

The companion only supplies the equation. The final proof is checked by a separate run of the stock
`bin/cp_verify`, built without the shadow headers.

Debugging: `HFork::valuesF` (`vf_subproof_segment.h`, `verifier/tests/claims_report.h`) tells which
claim fails on the prover's data.

## Sumcheck with a hiding sum (11.1)

Scalars j0..: `hs`, then round coefficients, then end values. Mask entries `mask(j0 + pos(l, deg))`
are the hiding polynomial's coefficients, NOT masks of the scalars sent under those numbers.
`pos(l, deg) = sk(l) + deg - 1` for l < NV - 1, `sk(NV-1) + deg` for the last variable, which alone
has a constant. U = sum_l u_l(X_l) prod_{i>l} psi_i(X_i), psi = 1 (equality, Y) or X (plain).
hs = phi applied to U. Round l: g = U + eps H' with earlier variables bound and later ones summed.
H' = H with val_i -> val_i + Y mask(jE + i), plus (pub(C) - C) prod psi_i(X_i). End value i =
val_i(point) + beta mask(jE + i). Equality round: constant not sent. Plain: g_1 not sent. Y: g_0 not
sent.

## Order of work

1. Generic masked sumcheck, product level, lookup, smallness test (CPU first, against the companion).
2. Segments: G (11.2), A (11.3), H (11.4).
3. Frame beyond the first chain.
4. Move heavy loops to Metal through MLX, by profile.
