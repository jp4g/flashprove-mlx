// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// What the sub-proofs of a request share on the prover's side: the channel, matrices of cells and their
// packed arrays, the placing of arrays in an oracle message's main table, and the smallness test
// (docs/format/11-1-machinery.md, third piece). See mlx/DESIGN_PROVER.md.
#pragma once
#include <functional>
#include <map>
#include <string>
#include "iop/context.h"
#include "chain/rounds.h"
#include "iop/req_rows.h"
#include "lookup_prover.h"
#include "numbers/round_names.h"
namespace fpmlx {
// FP_SELF_CHECKS: the prover also tests its own sums and each segment's equation, on the host. No proof
// needs them; they find a wrong witness early. Off by default.
inline bool self_checks() { static const bool on = getenv("FP_SELF_CHECKS") != nullptr; return on; }
// The channel as psc_run() and lookup_run() use it.
struct Chan {
  ProverRounds &r;
  // pub(C), which the verifier's walk computes; null: the caller's own value stands.
  std::function<E()> next_public;
  E public_part(const E &x) { return next_public ? next_public() : x; }
  void send(const E &x) { r.send(x); }
  void send_hiding_sum(const E &x) { r.send_hiding_sum(x); }
  E epsilon() { return r.close_epsilon(R_PSC_EPSILON); }
  E round() { return r.close_algebraic(R_PSC_ROUND).draw(); }
  E y() { return r.close_algebraic(R_PSC_Y).draw_nonzero(); }
  void values() { r.close_empty(R_PSC_VALUES); }
  E shift() { return r.close_algebraic(R_LKP_SHIFT).draw(); }
  E alpha() { return r.close_algebraic(R_LKP).draw(); }
  E line() { return r.close_algebraic(R_LINE).draw(); }
  std::vector<E> point(const char *round, size_t n) {
    auto d = r.close_algebraic(round);
    std::vector<E> v(n);
    for (E &x : v) x = d.draw();
    return v;
  }
  std::vector<E> point_join(size_t n) { return point(R_LKP_PLANES, n); }
  std::vector<E> point_zero(size_t n) { return point(R_ZERO, n); }
  E batch(u64 rows) { auto d = r.close_algebraic(R_BAT_SIGMA); d.require_rows(rows); return d.draw(); }
  // The segment's last batching: the number of its rows comes from the verifier's walk.
  std::function<u64()> next_rows;
  E batch_open() {
    auto d = r.close_algebraic(R_BAT_SIGMA);
    if (next_rows) d.require_rows(next_rows());
    return d.draw();
  }
};
// eq(z, i), the coordinate of the highest bit first.
inline E eq_at(const std::vector<E> &z, u64 i) {
  E w = E::one();
  const size_t n = z.size();
  for (size_t s = 0; s < n; s++) w = w * (((i >> (n - 1 - s)) & 1) ? z[s] : E::one() - z[s]);
  return w;
}
// All 2^n weights.
inline std::vector<E> eq_all(const std::vector<E> &z) {
  std::vector<E> w(1, E::one());
  for (const E &c : z) {
    std::vector<E> n(2 * w.size());
    for (size_t i = 0; i < w.size(); i++) n[2 * i] = w[i] * (E::one() - c), n[2 * i + 1] = w[i] * c;
    w = std::move(n);
  }
  return w;
}
// A matrix of small integers: rows of cells, with k columns. Packed as 11.1 says.
struct CellMatrix2 {
  u64 m = 0, k = 0;
  std::vector<int32_t> cell; // cell[i * k + c]
  CellMatrix2() = default;
  CellMatrix2(u64 m_, u64 k_) : m(m_), k(k_), cell(m_ * k_, 0) {}
  int32_t &at(u64 i, u64 c) { return cell[i * k + c]; }
  int32_t at(u64 i, u64 c) const { return cell[i * k + c]; }
  u64 packed_rows() const { return (m + 2) / 3; }
  F packed(u64 i, u64 c) const {
    long long v = 0;
    for (int q = 2; q >= 0; q--) { const u64 r = 3 * i + q; v = (long long)((__int128)v * (1 << 20)) + (r < m ? at(r, c) : 0); }
    return from_i64(v);
  }
};
// An array of a segment, as the layout names it: rows x columns of elements of F or E.
struct ArrayData {
  u64 rows = 0, columns = 0;
  bool over_e = false;
  std::vector<F> f; // f[r * columns + c]
  std::vector<E> e;
  E at(u64 r, u64 c) const { return over_e ? e[r * columns + c] : E::from(f[r * columns + c]); }
};
inline ArrayData array_of(const CellMatrix2 &M) {
  ArrayData a;
  a.rows = M.packed_rows(), a.columns = M.k;
  a.f.resize(a.rows * a.columns);
  const Buf w = pack_cells(M.cell.data(), M.m, M.k);
  memcpy((void *)a.f.data(), w.data(), a.f.size() * 8);
  return a;
}
// Places arrays in the main table of an oracle message, by the layout's blocks.
inline void lay_block(Table &t, const LaidBlock &b, const ArrayData &a) {
  const u64 n = u64(1) << (b.bits_i + b.bits_c);
  for (u64 x = 0; x < n; x++) {
    u64 i, c;
    if (b.i_is_low) i = x & ((u64(1) << b.bits_i) - 1), c = x >> b.bits_i;
    else c = x & ((u64(1) << b.bits_c) - 1), i = x >> b.bits_c;
    const u64 row = b.first_row + (x >> b.place_bits), place = b.first_place + (x & ((u64(1) << b.place_bits) - 1));
    t.set(row, place, a.at(b.first_i + i, b.first_c + c));
  }
}
// One part of U3: a matrix of the list and a weight vector of len entries.
struct U3Part {
  unsigned mat = 0;
  std::vector<E> phi;
};
// The smallness test of one segment.
class Smallness {
public:
  std::vector<const CellMatrix2 *> mats; // the segment's list, those of earlier segments among them
  std::vector<U3Part> parts;
  unsigned tau = 0, lidb = 0;
  // Entry at(f) + c nb(f) + b: the sum over r < len(f) of phi_f[r] cell(mat(f), b len(f) + r, c).
  std::vector<E> u3() const {
    std::vector<E> out;
    for (const U3Part &p : parts) {
      const CellMatrix2 &M = *mats[p.mat];
      const u64 len = p.phi.size(), nb = M.m / len, at = out.size();
      out.resize(at + M.k * nb, E::zero());
      const std::vector<E> sums = e_download(column_sums(M.cell.data(), M.k, nb, p.phi));
      std::copy(sums.begin(), sums.end(), out.begin() + at);
    }
    return out;
  }
  // RNG, after U3's moment: the matrix Theta.
  void draw_theta(ProverRounds &r) {
    u64 columns = 0;
    for (auto *M : mats) columns += M->k;
    auto from = r.close_bits(R_RNG);
    r.require_combinations(tau);
    theta_ = from.trits(columns * tau);
  }
  // ZZ and BZ. False: a combination leaves its window (the proof must be made again).
  bool digits(ArrayData &ZZ, ArrayData &BZ) {
    u64 mz = 0, col0 = 0;
    for (auto *M : mats) mz += M->m;
    mz_ = mz;
    ZZ = ArrayData();
    ZZ.rows = tau, ZZ.columns = 2 * mz, ZZ.f.assign(ZZ.rows * ZZ.columns, F{0});
    lc_ = clog2u(mz);
    n_ = std::max(10u, clog2u(tau) + lc_);
    const size_t nk = size_t(1) << n_;
    auto buffer = [](size_t bytes, mx::Dtype t, int fill) { mx::array x(mx::allocator::malloc(bytes), {(int)(bytes / mx::size_of(t))}, t); memset(x.data<uint8_t>(), fill, bytes); return x; };
    mx::array kl = buffer(nk * 8, mx::uint64, 0), kh = buffer(nk * 8, mx::uint64, 0), pad = buffer(nk, mx::uint8, 1), zz = buffer(ZZ.f.size() * 8, mx::uint64, 0), flag = buffer(64, mx::uint64, 0);
    std::vector<mx::array> done;
    std::vector<Buf> kept;
    u64 off = 0;
    for (auto *M : mats) {
      kept.push_back(combinations(M->cell.data(), M->m, M->k, theta_.data() + col0 * tau, tau));
      done.push_back(digit_split(kept.back().array(), M->m, tau, off, mz, lc_, kl, kh, pad, zz, flag));
      off += M->m, col0 += M->k;
    }
    mx::eval(done);
    if (flag.data<uint64_t>()[0]) return false;
    memcpy((void *)ZZ.f.data(), zz.data<uint64_t>(), ZZ.f.size() * 8);
    calls_.assign(2, LookupCall());
    for (unsigned q = 0; q < 2; q++) {
      LookupCall &L = calls_[q];
      L.n = n_, L.k = q ? 8 : 10, L.t0 = 0;
      L.keys.resize(nk), L.padding.resize(nk);
      memcpy(L.keys.data(), (q ? kh : kl).data<uint64_t>(), nk * 8);
      memcpy(L.padding.data(), pad.data<uint8_t>(), nk);
    }
    sizes_ = lookup_sizes(calls_, lidb);
    if (!lookup_multiplicities(calls_, sizes_, mult_)) return false;
    BZ = ArrayData();
    BZ.rows = 32, BZ.columns = 1 << 11, BZ.f.assign(BZ.rows * BZ.columns, F{0});
    for (u64 t = 0; t < 1024; t++)
      for (u64 c = 0; c < 2; c++)
        for (u64 j = 0; j < 32; j++)
          if ((mult_[c][t] >> j) & 1) {
            const u64 X = (2 * t + c) * 32 + j;
            BZ.f[(X >> 11) * BZ.columns + (X & 2047)] = F{1};
          }
    return true;
  }
  LookupOut run_lookup(Chan &ch, const E *mask, u32 &slot) { return lookup_run(ch, mask, slot, calls_, sizes_, mult_); }
private:
  std::vector<signed char> theta_;
  std::vector<LookupCall> calls_;
  LookupSizes sizes_;
  std::vector<std::vector<u32>> mult_;
  u64 mz_ = 0;
  unsigned lc_ = 0, n_ = 0;
};
}
