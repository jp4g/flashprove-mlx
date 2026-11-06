// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
#include "integer_forward.h"
#include <fstream>
#include <sstream>
namespace fpmlx {
using mx::array;
static array I(long long v) { return array((int64_t)v, mx::int64); }
static array rq(const array &a, int k) { return mx::right_shift(mx::add(a, I(1ll << (k - 1))), I(k)); }
static array clamp(const array &a, long long b) { return mx::minimum(mx::maximum(a, I(-b)), I(b)); }
static const array &at(const Tensors &w, const std::string &n) {
  auto it = w.find(n);
  if (it == w.end()) throw std::runtime_error("no tensor " + n);
  return it->second;
}
ModelShape read_shape(const std::string &path) {
  ModelShape p{};
  std::ifstream f(path);
  if (!f) throw std::runtime_error("no " + path);
  for (std::string line; std::getline(f, line);) {
    std::istringstream s(line);
    std::string k;
    s >> k;
    long long v = 0;
    if (k == "layer") {
      LayerScales l{};
      int i, e;
      s >> i >> l.kq >> l.kk >> l.kv >> l.ko >> l.kg >> l.ku >> l.kd >> l.M1 >> l.K1 >> l.M2 >> l.K2 >> l.sh >> l.rp >> l.kr >> l.kgR >> l.kuR >> e;
      l.experts = e;
      p.layers.push_back(l);
      continue;
    }
    s >> v;
    if (k == "T") p.T = v; else if (k == "L") p.L = v; else if (k == "d") p.d = v; else if (k == "n_h") p.n_h = v; else if (k == "n_g") p.n_g = v;
    else if (k == "dh") p.dh = v; else if (k == "f") p.f = v; else if (k == "fe") p.fe = v; else if (k == "NE") p.NE = v; else if (k == "V") p.V = v;
    else if (k == "cs") p.cs = v; else if (k == "dq") p.dq = v; else if (k == "dkv") p.dkv = v; else if (k == "eps1") p.eps1 = v; else if (k == "eps2") p.eps2 = v;
    else if (k == "epsF") p.epsF = v; else if (k == "MF") p.MF = v; else if (k == "KF") p.KF = v; else if (k == "gcl") p.gcl = v; else if (k == "rcl") p.rcl = v;
  }
  return p;
}
array exact_product(const array &x, const array &w) {
  static auto kernel = mx::fast::metal_kernel("fp_int_product", {"x", "w", "q"}, {"out"},
    R"(uint j = thread_position_in_grid.x, i = thread_position_in_grid.y; int k = q[0], o = q[1];
       long s = 0; const device long* a = x + (ulong)i * k; const device int* b = w + (ulong)j * k;
       for (int c = 0; c < k; c++) s += a[c] * (long)b[c];
       if ((int)j < o) out[(ulong)i * o + j] = s;)");
  const int n = x.shape(0), k = x.shape(1), o = w.shape(0);
  const int q[2] = {k, o};
  const int gx = (o + 63) / 64 * 64;
  return kernel({mx::contiguous(mx::astype(x, mx::int64)), mx::contiguous(mx::astype(w, mx::int32)), array(q, {2}, mx::int32)}, {{n, o}}, {mx::int64}, {gx, n, 1}, {64, 1, 1}, {}, std::nullopt, false, {})[0];
}
// sigma = isqrt(sum x^2 + eps), Rc = floor((2^M + floor(sigma / 2)) / sigma), one row per thread: the
// sum needs 128 bits.
static std::pair<array, array> norm_parts(const array &x, int M, long long eps) {
  static auto kernel = mx::fast::metal_kernel("fp_norm", {"x", "q"}, {"sigma", "rc"},
    R"(uint i = thread_position_in_grid.x; long d = q[0]; ulong eps = (ulong)q[1], M = (ulong)q[2];
       ulong hi = 0, lo = 0;
       for (long c = 0; c < d; c++) { long v = x[(ulong)i * d + c]; ulong u = (ulong)(v < 0 ? -v : v);
         ulong l = u * u, h = mulhi(u, u); lo += l; hi += h + (ulong)(lo < l); }
       ulong l0 = lo; lo += eps; hi += (ulong)(lo < l0);
       ulong r = 0;
       for (int b = 39; b >= 0; b--) { ulong c = r | (1ul << b), cl = c * c, ch = mulhi(c, c);
         if (ch < hi || (ch == hi && cl <= lo)) r = c; }
       sigma[i] = (long)r; rc[i] = (long)(((1ul << M) + (r >> 1)) / r);)");
  const int n = x.shape(0);
  const int64_t q[3] = {x.shape(1), eps, M};
  auto r = kernel({mx::contiguous(x), array(q, {3}, mx::int64)}, {{n}, {n}}, {mx::int64, mx::int64}, {n, 1, 1}, {std::min(n, 64), 1, 1}, {}, std::nullopt, false, {});
  return {r[0], r[1]};
}
static array norm(const array &x, int M, int K, long long eps, const Note &note, const std::string &name, const std::string &suffix) {
  auto [sigma, Rc] = norm_parts(x, M, eps);
  const array w = rq(mx::multiply(x, mx::expand_dims(Rc, 1)), K), nr = clamp(w, 2047);
  note(name + "_sigma" + suffix, sigma);
  note(name + "_Rc" + suffix, Rc);
  note(name + "_w" + suffix, w);
  note(name + suffix, nr);
  return nr;
}
// Rotary step on rows of heads: pairs (2 i, 2 i + 1), by the tables at each row's position.
static array rotary(const array &A, const array &C, const array &S, int heads, int dh) {
  const int n = A.shape(0);
  const array v = mx::reshape(A, {n, heads, dh / 2, 2});
  const array a0 = mx::squeeze(mx::slice(v, {0, 0, 0, 0}, {n, heads, dh / 2, 1}), 3), a1 = mx::squeeze(mx::slice(v, {0, 0, 0, 1}, {n, heads, dh / 2, 2}), 3);
  const array c = mx::expand_dims(C, 1), s = mx::expand_dims(S, 1);
  return mx::reshape(mx::stack({mx::subtract(mx::multiply(c, a0), mx::multiply(s, a1)), mx::add(mx::multiply(s, a0), mx::multiply(c, a1))}, 3), {n, heads * dh});
}
static array first_argmax(const array &a) { // the lowest index at the maximum
  const array m = mx::max(a, 1, true);
  const array idx = mx::arange(0, a.shape(1), 1, mx::int64);
  return mx::min(mx::where(mx::equal(a, m), idx, I(a.shape(1))), 1);
}
static array slice_cols(const array &a, int from, int to) { return mx::slice(a, {0, from}, {a.shape(0), to}); }
array forward(const ModelShape &p, const Tensors &w, const array &tokens, const Note &note) {
  const int n = tokens.shape(0);
  const array cos_t = mx::slice(at(w, "table.cos"), {0, 0}, {n, p.dh / 2}), sin_t = mx::slice(at(w, "table.sin"), {0, 0}, {n, p.dh / 2});
  const array &EX = at(w, "table.softmax"), &Ts = at(w, "table.silu"), &Sig = at(w, "table.sigmoid");
  const int u0 = EX.shape(0) - 1;
  array x = mx::take(at(w, "Emb"), tokens, 0);
  const array pos = mx::arange(0, n, 1, mx::int64);
  const array causal = mx::less_equal(mx::expand_dims(pos, 0), mx::expand_dims(pos, 1)); // [i][j]: j <= i
  for (int l = 0; l < p.L; l++) {
    const LayerScales &k = p.layers[l];
    const std::string L = "." + std::to_string(l);
    note("x" + L, x);
    const array a = norm(x, k.M1, k.K1, p.eps1, note, "a", L);
    const array one = mx::full({n, p.dh / 2}, I(1ll << p.cs)), zero = mx::zeros({n, p.dh / 2}, mx::int64);
    const array C = k.rp ? cos_t : one, S = k.rp ? sin_t : zero;
    const array pq = exact_product(a, at(w, "Wq" + L)), pk = exact_product(a, at(w, "Wk" + L));
    note("sum_preq" + L, pq), note("sum_prek" + L, pk);
    const array Aq = rotary(pq, C, S, p.n_h, p.dh), Ak = rotary(pk, C, S, p.n_g, p.dh);
    const array Av = exact_product(a, at(w, "Wv" + L));
    const array qw = rq(Aq, k.kq), kw = rq(Ak, k.kk), vw = rq(Av, k.kv);
    note("sum_q" + L, Aq), note("sum_k" + L, Ak), note("sum_v" + L, Av);
    const array q = clamp(qw, 2047), key = clamp(kw, 2047), v = clamp(vw, 2047);
    note("qw" + L, qw), note("q" + L, q), note("kw" + L, kw), note("k" + L, key), note("vw" + L, vw), note("v" + L, v);
    std::vector<array> heads;
    const int per = p.n_h / p.n_g;
    for (int h = 0; h < p.n_h; h++) {
      const int g = h / per;
      const array qh = slice_cols(q, h * p.dh, (h + 1) * p.dh), kh = slice_cols(key, g * p.dh, (g + 1) * p.dh), vh = slice_cols(v, g * p.dh, (g + 1) * p.dh);
      array s = exact_product(qh, kh); // n x n
      s = mx::where(causal, s, I(INT64_MIN / 2));
      const array m = mx::max(s, 1, true);
      const array u = mx::right_shift(mx::subtract(m, s), I(k.sh));
      const array e = mx::where(causal, mx::take(EX, mx::minimum(u, I(u0)), 0), I(0));
      const array Z = mx::sum(e, 1, true);
      const array acc = exact_product(e, mx::transpose(vh));
      const array num = mx::add(mx::multiply(acc, I(2)), Z), den = mx::multiply(Z, I(2));
      heads.push_back(mx::floor_divide(num, den));
    }
    const array o = mx::concatenate(heads, 1);
    note("o" + L, o);
    const array Ao = exact_product(o, at(w, "Wo" + L));
    note("sum_o" + L, Ao);
    x = mx::add(x, rq(Ao, k.ko));
    note("x_mid" + L, x);
    const array m = norm(x, k.M2, k.K2, p.eps2, note, "m", L);
    const int shared = k.experts ? p.fe : p.f;
    array Ag = exact_product(m, at(w, "Wg" + L)), Au = exact_product(m, at(w, "Wu" + L));
    array gw = rq(Ag, k.kg), u = rq(Au, k.ku);
    array e_star = mx::zeros({n}, mx::int64);
    std::vector<array> gr, ur;
    array xr = m;
    if (k.experts) {
      const array Ar = exact_product(m, at(w, "Wr" + L));
      note("sum_r" + L, Ar);
      const array lw = rq(Ar, k.kr);
      array lr = lw;
      if (p.rcl) lr = mx::add(mx::multiply(clamp(mx::right_shift(lw, I(3)), 2047), I(8)), mx::bitwise_and(lw, I(7)));
      e_star = first_argmax(lr);
      const array chosen = mx::squeeze(mx::take_along_axis(lr, mx::expand_dims(e_star, 1), 1), 1);
      const array sc = mx::take(Sig, mx::add(chosen, I(16384)), 0);
      xr = rq(mx::multiply(mx::expand_dims(sc, 1), m), 16);
      note("lr" + L, lr), note("e_star" + L, e_star), note("sc" + L, sc), note("xr" + L, xr);
      array gsel = mx::zeros({n, p.fe}, mx::int64), usel = gsel, sg = gsel, su_ = gsel;
      for (int e = 0; e < p.NE; e++) {
        const std::string E = L + "." + std::to_string(e);
        const array mine = mx::expand_dims(mx::equal(e_star, I(e)), 1);
        const array eg = exact_product(xr, at(w, "Wgr" + E)), eu = exact_product(xr, at(w, "Wur" + E));
        sg = mx::where(mine, eg, sg), su_ = mx::where(mine, eu, su_);
        gsel = mx::where(mine, rq(eg, k.kgR), gsel);
        usel = mx::where(mine, rq(eu, k.kuR), usel);
      }
      gw = mx::concatenate({gw, gsel}, 1);
      u = mx::concatenate({u, usel}, 1);
      Ag = mx::concatenate({Ag, sg}, 1), Au = mx::concatenate({Au, su_}, 1);
    }
    array table_key = gw;
    if (p.gcl) {
      const array g0 = mx::subtract(mx::bitwise_and(mx::add(gw, I(128)), I(255)), I(128));
      const array gh = mx::right_shift(mx::subtract(gw, g0), I(8));
      table_key = mx::add(mx::multiply(clamp(gh, 127), I(256)), g0);
    }
    const array s = mx::take(Ts, mx::add(table_key, I(32768)), 0);
    note("gw" + L, gw), note("key" + L, table_key), note("s" + L, s), note("u" + L, u), note("sum_g" + L, Ag), note("sum_u" + L, Au);
    const array su = mx::multiply(s, u);
    array ad = exact_product(slice_cols(su, 0, shared), at(w, "Wd" + L));
    if (k.experts) {
      const array sr = slice_cols(su, shared, shared + p.fe);
      for (int e = 0; e < p.NE; e++) {
        const std::string E = L + "." + std::to_string(e);
        const array mine = mx::expand_dims(mx::equal(e_star, I(e)), 1);
        ad = mx::add(ad, mx::where(mine, exact_product(sr, at(w, "Wdr" + E)), I(0)));
      }
    }
    note("sum_d" + L, ad);
    x = mx::add(x, rq(ad, k.kd));
  }
  note("x_last", x);
  const array aF = norm(x, p.MF, p.KF, p.epsF, note, "aF", "");
  const array lam = exact_product(aF, at(w, "Wh"));
  note("lam", lam);
  note("first_choice", first_argmax(lam));
  return lam;
}
}
