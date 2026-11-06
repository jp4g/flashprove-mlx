// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// What a segment's prover does around its sub-proof: the mask row, the main tables of a moment, and
// the equation that closes it.
#pragma once
#include "companion_link.h"
#include "iop/send.h"
#include "seg_common.h"
namespace fpmlx {
struct SegMask {
  Table table;
  std::vector<E> v; // entry j masks scalar j
};
inline SegMask make_mask(ProverContext &P, const ReqShape &shape, size_t link) {
  const Codes &codes = P.st.codes();
  for (const ReqMessage &msg : shape.link[link].messages)
    if (msg.shape.mask_rows) {
      BLK_REQUIRE(msg.shape.mask_rows == 1, "fp_prove: a mask of more than one row");
      const size_t ell = codes.mask[msg.shape.mask_code].ell;
      SegMask s{Table(E_DEGREE, 1, ell), std::vector<E>(ell)};
      std::vector<F> r(ell * E_DEGREE);
      P.coins.fill(r.data(), r.size());
      for (size_t i = 0; i < ell; i++) s.v[i] = E{{r[3 * i], r[3 * i + 1], r[3 * i + 2]}};
      for (const auto &pad : Send::pad(codes, msg.shape)) s.v[pad.index] = E::zero();
      for (size_t i = 0; i < ell; i++) s.table.set(0, i, s.v[i]);
      return s;
    }
  BLK_REQUIRE(false, "fp_prove: a segment without a mask row");
  return {};
}
using Arrays = std::map<std::string, ArrayData>;
inline std::vector<TableId> place_tables(ProverContext &P, const ReqLayout &lay, const ReqShape &shape, size_t link, unsigned moment, const Arrays &arrays) {
  const Codes &codes = P.st.codes();
  std::vector<TableId> ids;
  const auto &msgs = shape.link[link].messages;
  for (size_t mi = 0; mi < msgs.size(); mi++) {
    if (msgs[mi].moment != moment) continue;
    const OracleShape &o = msgs[mi].shape;
    if (!o.main_rows) { ids.push_back(NO_ID); continue; }
    Table t(o.main_deg, o.main_rows, codes.base[o.code].ell);
    // The table's words are written on the GPU, block by block, and read back once.
    mx::array words(mx::allocator::malloc(t.a.size() * 8), {(int)t.a.size()}, mx::uint64);
    memset(words.data<uint64_t>(), 0, t.a.size() * 8);
    std::vector<mx::array> done;
    for (const ReqArray &a : lay.link[link]) {
      if (a.message != mi) continue;
      auto it = arrays.find(a.name);
      if (it == arrays.end()) throw std::runtime_error("no array named " + a.name + " at link " + std::to_string(link));
      const ArrayData &d = it->second;
      if (d.over_e && t.deg == 1) throw std::runtime_error("an array over E in a table over F");
      const u64 *from = d.over_e ? reinterpret_cast<const u64 *>(d.e.data()) : reinterpret_cast<const u64 *>(d.f.data());
      const size_t count = (d.over_e ? 3 : 1) * d.rows * d.columns;
      std::vector<u64> pad(from, from + count);
      pad.resize(std::max<size_t>(count, 8), 0);
      const mx::array src(reinterpret_cast<const uint64_t *>(pad.data()), {(int)pad.size()}, mx::uint64);
      for (const LaidBlock &b : a.laid.blocks) {
        const u64 n = u64(1) << (b.bits_i + b.bits_c), place_mask = (u64(1) << b.place_bits) - 1;
        BLK_REQUIRE(b.first_row + ((n - 1) >> b.place_bits) < t.rows && b.first_place + std::min(place_mask, n - 1) < t.len, "table: no such entry (rows beyond the real ones are zero and cannot be set)");
        const u64 words8[8] = {b.first_i, b.first_c, b.first_row, b.first_place, b.bits_i, b.bits_c, b.place_bits, b.i_is_low};
        done.push_back(fpmlx::lay_block(words, t.deg, t.len, src, d.over_e ? 3 : 1, d.columns, words8));
      }
    }
    mx::eval(done);
    memcpy((void *)t.a.data(), words.data<uint64_t>(), t.a.size() * 8);
    ids.push_back(P.engine.put(t));
  }
  return ids;
}
// A weight of IndexWeights: bit t of the index picks lo[t] or hi[t].
inline std::vector<E> weights_of(const IndexWeights &w, u64 n) {
  std::vector<E> v(n, E::one());
  if (w.values) { for (u64 i = 0; i < n; i++) v[i] = i < w.values->size() ? (*w.values)[i] : E::zero(); return v; }
  for (u64 i = 0; i < n; i++) for (size_t t = 0; t < w.lo.size(); t++) v[i] = v[i] * (((i >> t) & 1) ? w.hi[t] : w.lo[t]);
  return v;
}
// All arrays of the request so far, by segment and name: what the equations read.
using HeldArrays = std::map<std::pair<size_t, std::string>, ArrayData>;
// The left side minus the target of a segment's equation, on the prover's own data: 0 if the segment's
// claims hold. A test of this program only; the verifier forms and tests the equation by itself.
inline E equation_gap(const ReqEquation &eq, const HeldArrays &held, const std::vector<E> &mask) {
  E sum = E::zero() - eq.target;
  for (const ReqTerm &t : eq.terms) {
    auto it = held.find({t.born, t.array});
    if (it == held.end()) throw std::runtime_error("the equation reads an array that is not held: " + t.array);
    const ArrayData &a = it->second;
    const std::vector<E> wr = weights_of(t.on_rows, a.rows), wc = weights_of(t.on_columns, a.columns);
    E acc = E::zero();
    for (u64 r = 0; r < a.rows; r++) {
      E row = E::zero();
      for (u64 c = 0; c < a.columns; c++) row = row + wc[c] * a.at(r, c);
      acc = acc + wr[r] * row;
    }
    sum = sum + t.scale * acc;
  }
  for (size_t i = 0; i < eq.on_masks.size(); i++) sum = sum + eq.on_masks[i] * mask[i];
  return sum;
}
// Takes the segment's equation from the companion and closes the segment.
inline void close_segment(ProverContext &P, ReqProver &frame, ReqRows<ProverContext> &rows, Companion &io, const HeldArrays *held = nullptr, const std::vector<E> *mask = nullptr) {
  const ReqEquation eq = read_equation(io.take(fpstream::EQUATION));
  if (held) {
    const E gap = equation_gap(eq, *held, *mask);
    printf("segment %zu: %zu terms, the equation %s\n", frame.link(), eq.terms.size(), gap == E::zero() ? "HOLDS on the prover's data" : "DOES NOT HOLD");
  }
  P.st.add_row(frame.group(), rows.row(eq));
  frame.close(false);
}
}
