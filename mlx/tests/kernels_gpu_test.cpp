// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The witness and lookup kernels against plain host code on random data: combinations (the window's
// edges too), column sums, g^T M on bytes, product levels, lookup leaves, the row-first transpose.
#include <cstdio>
#include <random>
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
  printf("%d of %d kernels equal\n", runs - bad, runs);
  return bad != 0;
}
