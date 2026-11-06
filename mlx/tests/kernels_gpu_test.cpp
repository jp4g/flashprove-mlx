// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The witness and lookup kernels against plain host code on random data: combinations (the window's
// edges too), column sums, g^T M on bytes, product levels, lookup leaves, the row-first transpose.
#include <cstdio>
#include <random>
#include "engine/table.h"
#include "mlx_kernels.h"
using namespace blk;
using namespace fpmlx;
static std::mt19937_64 g(21);
static E rnd() { E x; for (int k = 0; k < 3; k++) x.c[k] = from_u64(g()); return x; }
static int bad = 0, runs = 0;
static void report(const char *name, bool ok) { runs++; if (!ok) bad++, printf("DIFFERS: %s\n", name); }
static bool same(const u64 *a, const std::vector<E> &b) { return !memcmp(a, b.data(), b.size() * 24); }
int main() {
  { // combinations
    const size_t m = 1000, k = 37, tau = 5;
    std::vector<int32_t> cells(m * k);
    std::vector<signed char> th(k * tau);
    for (auto &c : cells) c = (int)(g() % 4001) - 2000;
    for (auto &t : th) t = (signed char)((int)(g() % 3) - 1);
    cells[3 * k] = 1 << 20, th[0] = 1; // row 3 leaves the window at u = 0
    for (size_t c = 1; c < k; c++) cells[3 * k + c] = 0;
    const Buf v = combinations(cells.data(), m, k, th.data(), tau);
    bool ok = true;
    for (size_t u = 0; u < tau; u++)
      for (size_t i = 0; i < m; i++) {
        long long s = 0;
        for (size_t c = 0; c < k; c++) s += (long long)th[c * tau + u] * cells[i * k + c];
        const u64 want = (s < -(1 << 17) || s >= (1 << 17)) ? ~u64(0) : (u64)(s + (1 << 17));
        ok = ok && v.data()[u * m + i] == want;
      }
    report("combinations", ok && v.data()[3] == ~u64(0));
  }
  { // column sums
    const size_t k = 48, nb = 6, len = 64;
    std::vector<int32_t> cells(nb * len * k);
    for (auto &c : cells) c = (g() % 3 == 0) ? 0 : (int)(g() % 2000001) - 1000000;
    std::vector<E> phi(len);
    for (E &x : phi) x = rnd();
    std::vector<E> want(k * nb, E::zero());
    for (size_t c = 0; c < k; c++)
      for (size_t b = 0; b < nb; b++)
        for (size_t r = 0; r < len; r++) want[c * nb + b] = want[c * nb + b] + phi[r] * from_i64(cells[(b * len + r) * k + c]);
    report("column sums", same(column_sums(cells.data(), k, nb, phi).data(), want));
  }
  { // g^T M
    const size_t rows = 300, cols = 257;
    std::vector<int8_t> cells(rows * cols);
    for (auto &c : cells) c = (int8_t)g();
    std::vector<E> gv(rows), want(cols, E::zero());
    for (E &x : gv) x = rnd();
    for (size_t c = 0; c < cols; c++) for (size_t y = 0; y < rows; y++) want[c] = want[c] + gv[y] * from_i64(cells[y * cols + c]);
    report("bytes left", same(bytes_left(cells.data(), rows, cols, gv).data(), want));
  }
  { // product level
    std::vector<E> a(4096), want(2048);
    for (E &x : a) x = rnd();
    for (size_t i = 0; i < 2048; i++) want[i] = a[i] * a[i + 2048];
    report("product level", !memcmp(e_fetch(product_level(e_on_gpu(a))).data(), want.data(), want.size() * 24));
  }
  { // lookup leaves: 3 calls over 8 trees of 2^6 keys, one tree slot empty
    LeafPlan L;
    L.rK = 3, L.dT = 6, L.v = 9, L.NT = 7, L.alpha = rnd(), L.theta = rnd();
    const u64 rowpad[3] = {0, 77, ~u64(0) - 5}; // the last is above p
    L.pad_value.assign(rowpad, rowpad + 3);
    L.pad0 = L.alpha - E::from(from_u64(rowpad[0]));
    std::vector<u64> keys(7 << 6);
    std::vector<char> pad(keys.size());
    for (auto &x : keys) x = g();
    for (auto &x : pad) x = g() % 4 == 0;
    const unsigned call[7] = {0, 0, 0, 1, 1, 2, 2};
    for (u64 y = 0; y < 7; y++) L.tree_call.push_back(call[y]), L.tree_first.push_back(y << 6);
    std::vector<E> want(512);
    for (u64 i = 0; i < 512; i++) {
      const u64 y = i & 7, x = i >> 3;
      if (y >= 7) { want[i] = L.pad0; continue; }
      const u64 at = (y << 6) + x, value = pad[at] ? rowpad[call[y]] : keys[at];
      want[i] = L.alpha - (E::from(from_u64(value)) + E::from(from_u64(call[y])) * L.theta);
    }
    report("lookup leaves", !memcmp(e_fetch(lookup_leaves(L, keys, pad)).data(), want.data(), want.size() * 24));
  }
  { // rows first
    const size_t rows = 64, ell = 24;
    std::vector<E> a(rows * ell), want(rows * ell);
    for (E &x : a) x = rnd();
    for (size_t b = 0; b < rows; b++) for (size_t i = 0; i < ell; i++) want[b * ell + i] = a[b + rows * i];
    report("rows first", same(rows_first(e_upload(a), rows, ell).data(), want));
  }
  { // a table's dot, deg 1 and 3, rows below 2^log_rows
    for (unsigned deg : {1u, 3u}) {
      Table t(deg, 5, 64);
      for (size_t r = 0; r < 5; r++) for (size_t c = 0; c < 64; c++) { E x = rnd(); if (deg == 1) x.c[1] = x.c[2] = F{0}; t.set(r, c, x); }
      std::vector<E> w(t.entries());
      for (E &x : w) x = rnd();
      E want = E::zero();
      for (u64 j = 0; j < t.entries(); j++) want = want + w[j] * t.entry(j);
      report(deg == 1 ? "dot table F" : "dot table E", dot_table(e_upload(w), table_words(t)) == want);
    }
  }
  { // rows gathered into a fold
    Table a(1, 4, 16), b(3, 2, 16);
    for (size_t r = 0; r < 4; r++) for (size_t c = 0; c < 16; c++) { E x = rnd(); x.c[1] = x.c[2] = F{0}; a.set(r, c, x); }
    for (size_t r = 0; r < 2; r++) for (size_t c = 0; c < 16; c++) b.set(r, c, rnd());
    mx::array o = e_rows_buffer(8 * 16);
    std::vector<mx::array> done{put_rows(o, table_words(a), 0), put_rows(o, table_words(b), 4)};
    mx::eval(done);
    std::vector<E> want(8 * 16, E::zero());
    for (size_t r = 0; r < 4; r++) for (size_t c = 0; c < 16; c++) want[r * 16 + c] = a.at(r, c);
    for (size_t r = 0; r < 2; r++) for (size_t c = 0; c < 16; c++) want[(4 + r) * 16 + c] = b.at(r, c);
    report("put rows", same(o.data<uint64_t>() ? reinterpret_cast<const u64 *>(o.data<uint64_t>()) : nullptr, want));
  }
  { // an engine's term under choice bits
    Table t(3, 4, 8);
    for (size_t r = 0; r < 4; r++) for (size_t c = 0; c < 8; c++) t.set(r, c, rnd());
    std::vector<E> base(256), w(t.entries());
    for (E &x : base) x = rnd();
    for (u64 j = 0; j < t.entries(); j++) w[j] = t.entry(j);
    EngineTerm term; term.scale = rnd(), term.choice_mask = 0b10010, term.choice_value = 0b00010;
    std::vector<E> want = base;
    for (u64 j = 0; j < 128; j++) want[j] = want[j] + engine_term_at(term, w, j, 7);
    const Buf o = e_upload(base);
    add_term(o, table_words(t), term.scale, term.choice_mask, term.choice_value);
    report("engine term", !memcmp(o.data(), want.data(), 128 * 24));
  }
  { // rows_left over F
    const size_t rows = 40, len = 300;
    std::vector<u64> m(rows * len);
    for (auto &x : m) x = from_u64(g()).v;
    std::vector<E> w(rows), want(len, E::zero());
    for (E &x : w) x = rnd();
    for (size_t j = 0; j < len; j++) for (size_t x = 0; x < rows; x++) want[j] = want[j] + w[x] * F{m[x * len + j]};
    const mx::array a(reinterpret_cast<const uint64_t *>(m.data()), {(int)m.size()}, mx::uint64);
    report("rows left", !memcmp(rows_left(a, rows, len, w).data(), want.data(), len * 24));
  }
  { // counts: a range table, a table of values with a repeated value, and refusals
    std::vector<u64> keys(1 << 14);
    std::vector<char> pad(keys.size());
    for (size_t i = 0; i < keys.size(); i++) keys[i] = g() % 1024, pad[i] = g() % 5 == 0;
    std::vector<u32> c(1024, 0), want(1024, 0);
    for (size_t i = 0; i < keys.size(); i++) if (!pad[i]) want[keys[i]]++;
    report("counts in a range", lookup_counts(keys, pad, 10, {}, {}, c.data()) && c == want);
    keys[7] = 5000, pad[7] = 0;
    report("a key beyond a range is refused", !lookup_counts(keys, pad, 10, {}, {}, c.data()));
    pad[7] = 1;
    report("a key beyond a range under padding counts for nothing", lookup_counts(keys, pad, 10, {}, {}, c.data()));
    std::vector<u64> values; std::vector<u32> rows;
    for (u32 r = 0; r < 256; r++) values.push_back((u64)(r / 2) * 1000), rows.push_back(r); // each value twice: the lower row counts
    std::vector<u64> k2(4096); std::vector<char> p2(4096, 0);
    std::vector<u32> c2(256, 0), w2(256, 0);
    for (auto &x : k2) { const u64 r = g() % 128; x = r * 1000; w2[2 * r]++; }
    report("counts in a table", lookup_counts(k2, p2, 8, values, rows, c2.data()) && c2 == w2);
    k2[9] = 17;
    report("a key in no row is refused", !lookup_counts(k2, p2, 8, values, rows, c2.data()));
  }
  { // packing of cells, three rows to a word, the last group short
    const size_t m = 100, k = 13;
    std::vector<int32_t> cells(m * k);
    for (auto &c : cells) c = (int)(g() % 1000001) - 500000;
    const Buf w = pack_cells(cells.data(), m, k);
    bool ok = true;
    for (size_t i = 0; i < (m + 2) / 3; i++)
      for (size_t c = 0; c < k; c++) {
        long long v = 0;
        for (int q = 2; q >= 0; q--) { const size_t r = 3 * i + q; v = (long long)((__int128)v * (1 << 20)) + (r < m ? cells[r * k + c] : 0); }
        ok = ok && w.data()[i * k + c] == from_i64(v).v;
      }
    report("pack cells", ok);
  }
  { // a laid block, both index orders, F into E
    for (int low = 0; low < 2; low++) {
      const size_t cols = 16, len = 32;
      std::vector<u64> src(16 * cols);
      for (auto &x : src) x = from_u64(g()).v;
      const mx::array s(reinterpret_cast<const uint64_t *>(src.data()), {(int)src.size()}, mx::uint64);
      mx::array t(mx::allocator::malloc(8 * len * 3 * 8), {(int)(8 * len * 3)}, mx::uint64);
      memset(t.data<uint64_t>(), 0, 8 * len * 3 * 8);
      const u64 b[8] = {4, 8, 2, 8, 2, 3, 4, (u64)low}; // first_i, first_c, first_row, first_place, bits_i, bits_c, place_bits, i_is_low
      mx::eval(lay_block(t, 3, len, s, 1, cols, b));
      std::vector<u64> want(8 * len * 3, 0);
      for (u64 x = 0; x < 32; x++) {
        const u64 i = low ? x & 3 : x >> 3, c = low ? x >> 2 : x & 7, row = 2 + (x >> 4), place = 8 + (x & 15);
        want[(row * len + place) * 3] = src[(4 + i) * cols + 8 + c];
      }
      report("lay block", !memcmp(t.data<uint64_t>(), want.data(), want.size() * 8));
    }
  }
  { // digits of combinations
    const size_t m = 64, tau = 4, mz = 128; const unsigned lc = 7;
    std::vector<u64> v(m * tau);
    for (auto &x : v) x = g() % (1 << 18);
    auto zero = [](size_t bytes, mx::Dtype t) { mx::array a(mx::allocator::malloc(bytes), {(int)(bytes / mx::size_of(t))}, t); memset(a.data<uint8_t>(), 0, bytes); return a; };
    mx::array kl = zero(8 << 9, mx::uint64), kh = zero(8 << 9, mx::uint64), pd = zero(1 << 9, mx::uint8), zz = zero(tau * 2 * mz * 8, mx::uint64), fl = zero(64, mx::uint64);
    pd = mx::array(std::vector<uint8_t>(512, 1).data(), {512}, mx::uint8); mx::eval(pd);
    const mx::array a(reinterpret_cast<const uint64_t *>(v.data()), {(int)v.size()}, mx::uint64);
    mx::eval(digit_split(a, m, tau, 16, mz, lc, kl, kh, pd, zz, fl));
    bool ok = fl.data<uint64_t>()[0] == 0;
    for (size_t u = 0; u < tau; u++) for (size_t i = 0; i < m; i++) {
      const u64 x = v[u * m + i], at = (u << lc) + 16 + i;
      ok = ok && kl.data<uint64_t>()[at] == (x & 1023) && kh.data<uint64_t>()[at] == (x >> 10) && pd.data<uint8_t>()[at] == 0;
      ok = ok && zz.data<uint64_t>()[u * 2 * mz + 16 + i] == (x & 1023) && zz.data<uint64_t>()[u * 2 * mz + mz + 16 + i] == (x >> 10);
    }
    report("digit split", ok);
  }
  { // the table side of a lookup
    const unsigned kT = 6, cK = 1, NC = 2, nT = 5 + kT + cK;
    std::vector<u64> rows((size_t)NC << kT); std::vector<u32> mult((size_t)NC << kT);
    for (auto &x : rows) x = g();
    for (auto &x : mult) x = (u32)g();
    const E alpha = rnd(), theta = rnd();
    const TableSide t = lookup_table_side(kT, cK, NC, rows, mult, alpha, theta);
    const std::vector<E> br = e_fetch(t.bits_rows), lv = e_fetch(t.leaves);
    bool ok = true;
    const size_t n = size_t(1) << nT;
    for (size_t i = 0; i < n; i++) {
      const u64 j = i & 31, c = (i >> 5) & 1, tt = i >> 6;
      const E rv = E::from(from_u64(rows[(c << kT) + tt])) + E::from(from_u64(c)) * theta, rowA = alpha - E::one() - rv;
      const E b = ((mult[(c << kT) + tt] >> j) & 1) ? E::one() : E::zero();
      ok = ok && br[i] == b && br[n + i] == rowA && lv[i] == E::one() + b * rowA;
    }
    report("lookup table side", ok);
  }
  printf("%d of %d kernels equal\n", runs - bad, runs);
  return bad != 0;
}
