// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The witness of a feed-forward segment (docs/format/11-2-feed-forward-segment.md): cells, a norm's row
// of digits, and the keys of the lookup of step 1. Every key must be a row of its call's table, which
// lookup_multiplicities() tests.
#pragma once
#include <array>
#include <fstream>
#include <sstream>
#include "integer_forward.h"
#include "seg_common.h"
namespace fpmlx {
// policy.txt as "name value" lines, and its call lines.
struct Policy {
  std::map<std::string, long long> v;
  struct CallLine { std::string name, kind; std::vector<std::string> segs; };
  std::vector<CallLine> call, callA, callH;
  long long get(const std::string &k, long long dflt = -1) const { auto it = v.find(k); if (it != v.end()) return it->second; if (dflt == -1) throw std::runtime_error("policy: no line " + k); return dflt; }
  static Policy read(const std::string &text) {
    Policy p;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);) {
      std::istringstream s(line);
      std::vector<std::string> w;
      for (std::string x; s >> x;) w.push_back(x);
      if (w.size() < 2) continue;
      if (w[0] == "call" || w[0] == "callA" || w[0] == "callH") {
        CallLine c{w[1], w[2], std::vector<std::string>(w.begin() + 3, w.end())};
        (w[0] == "call" ? p.call : w[0] == "callA" ? p.callA : p.callH).push_back(c);
      } else if (w[0] == "table") {
        p.v["table_w"] = atoll(w.back().c_str()), p.v["table_b"] = atoll(w[1].c_str());
      } else if (w.size() == 2) {
        p.v[w[0]] = atoll(w[1].c_str());
      }
    }
    return p;
  }
};
using I64 = long long;
// A forward pass's values by name: rows by token.
struct Values {
  std::map<std::string, std::vector<I64>> v;
  std::map<std::string, int> cols;
  std::map<std::string, mx::array> g; // the same values on the GPU, int64
  const std::vector<I64> &at(const std::string &n) const { auto it = v.find(n); if (it == v.end()) throw std::runtime_error("no value " + n); return it->second; }
  I64 get(const std::string &n, int t, int x) const { return at(n)[(size_t)t * cols.at(n) + x]; }
};
inline Values capture_forward(const ModelShape &p, const Tensors &w, const mx::array &tokens) {
  Values out;
  const Note note = [&](const std::string &name, const mx::array &a) {
    const mx::array c = mx::astype(a, mx::int64);
    mx::eval(c);
    out.v[name].assign(c.data<int64_t>(), c.data<int64_t>() + c.size());
    out.cols[name] = c.ndim() == 2 ? c.shape(1) : 1;
    out.g.insert_or_assign(name, c);
  };
  mx::array lam = forward(p, w, tokens, note);
  mx::eval(lam);
  return out;
}
inline unsigned lift(int k) { return k < 16 ? 16 - k : 0; }
inline I64 floordiv_pow2(__int128 a, int k) { return (I64)(a >> k); }
// value as nl signed bytes, the top limb taking what is left.
inline void limbs_of(I64 v, int nl, int32_t *out) {
  for (int l = 0; l < nl; l++) {
    if (l + 1 == nl) { out[l] = (int32_t)v; return; }
    const I64 c = ((v + 128) & 255) - 128;
    out[l] = (int32_t)c;
    v = (v - c) >> 8;
  }
}
inline I64 ofs_of(int h) { return h == 1 ? 128 : h == 2 ? 32896 : 0; }
// A norm's row at one token, with its numbers.
struct NormRow {
  std::vector<int> D;  // the digits
  I64 a = 0, b = 0, sig = 0, Rlo = 0, Rhi = 0, R = 0;
  std::vector<__int128> W; // W_s
};
inline void put_digits(std::vector<int> &D, int at, int n, unsigned __int128 x) { for (int j = 0; j < n; j++) D[at + j] = (int)(x & 255), x >>= 8; }
// x: the stream row (d values); limbs of each value in lim[j * XL + lb].
inline NormRow norm_row(int XL, const std::vector<int32_t> &lim, int d, I64 eps, int M) {
  NormRow r;
  r.D.assign(99, 0);
  const int NS = 2 * XL - 1;
  r.W.assign(NS, 0);
  for (int j = 0; j < d; j++)
    for (int p = 0; p < XL; p++)
      for (int q = 0; q < XL; q++) r.W[p + q] += (__int128)lim[(size_t)j * XL + p] * lim[(size_t)j * XL + q];
  __int128 V = 0;
  for (int s = NS - 1; s >= 0; s--) V = V * 256 + r.W[s];
  const unsigned __int128 Ve = (unsigned __int128)V + (unsigned __int128)eps;
  unsigned __int128 sig = 0;
  for (int b = 39; b >= 0; b--) { const unsigned __int128 c = sig | ((unsigned __int128)1 << b); if (c * c <= Ve) sig = c; }
  r.sig = (I64)sig;
  const I64 R = (I64)((((unsigned __int128)1 << M) + (sig >> 1)) / sig);
  r.R = R;
  // Carries.
  __int128 c = 0;
  std::vector<__int128> cs(NS + 1, 0);
  std::vector<int> vb(NS + 1, 0);
  for (int s = 0; s < NS; s++) {
    const __int128 t = r.W[s] + c;
    const int v = (int)(((t % 256) + 256) % 256);
    vb[s] = v;
    c = (t - v) / 256;
    cs[s + 1] = c;
  }
  r.a = r.sig & ((1ll << 24) - 1), r.b = r.sig >> 24;
  r.Rlo = R & ((1ll << 24) - 1), r.Rhi = R >> 24;
  const I64 l1 = (I64)(Ve - sig * sig), h1 = 2 * r.sig - l1;
  for (int s = 0; s < NS; s++) r.D[s] = vb[s];
  r.D[NS] = (int)cs[NS]; // D_9 = c_9
  for (int s = 1; s < NS; s++) { const unsigned __int128 e = (unsigned __int128)(cs[s] + ((__int128)1 << 47)); put_digits(r.D, 10 + 6 * (s - 1), 6, e); }
  const I64 a = r.a, b = r.b;
  r.D[59] = (int)(a & 1), r.D[58] = (int)((a & 255) >> 1), r.D[60] = (int)((a >> 8) & 255), r.D[61] = (int)((a >> 16) & 255);
  r.D[62] = (int)(b & 255), r.D[63] = (int)(b >> 8);
  put_digits(r.D, 64, 5, (unsigned __int128)l1), put_digits(r.D, 69, 5, (unsigned __int128)h1);
  const I64 ab = a * b, Blo = ab & ((1ll << 23) - 1), Bhi = ab >> 23;
  put_digits(r.D, 74, 3, (unsigned __int128)Blo), put_digits(r.D, 77, 2, (unsigned __int128)Bhi);
  // kap: from the line LE1.
  __int128 Vlo = 0;
  for (int j = 5; j >= 0; j--) Vlo = Vlo * 256 + r.D[j];
  const __int128 rest = (__int128)a * a + ((__int128)Blo << 25) + l1 - Vlo - eps;
  const I64 kap = (I64)(rest / ((__int128)1 << 48));
  r.D[79] = (int)(kap + 4);
  put_digits(r.D, 80, 3, (unsigned __int128)r.Rlo), put_digits(r.D, 83, 3, (unsigned __int128)r.Rhi);
  const I64 q1 = r.Rlo * b + r.Rhi * a;
  put_digits(r.D, 86, 3, (unsigned __int128)q1);
  const I64 rr = (I64)(((__int128)1 << M) + (r.sig >> 1) - (__int128)R * r.sig), rq = r.sig - 1 - rr;
  put_digits(r.D, 89, 5, (unsigned __int128)rr), put_digits(r.D, 94, 5, (unsigned __int128)rq);
  return r;
}
inline u64 row_mul(int j) { return j == 9 ? 4 : j == 58 ? 2 : j == 59 ? 128 : j == 63 ? 2 : j == 76 ? 2 : j == 79 ? 32 : 1; }
// The public tables of the calls.
struct CallTables { std::vector<u64> FC, FL, FS; };
inline CallTables call_tables(const Tensors &w) {
  CallTables c;
  c.FC.resize(65536), c.FL.resize(65536), c.FS.resize(65536);
  mx::array si = mx::astype(w.at("table.silu"), mx::int64), sg = mx::astype(w.at("table.sigmoid"), mx::int64);
  mx::eval(si), mx::eval(sg);
  const int64_t *s = si.data<int64_t>(), *g = sg.data<int64_t>();
  auto cl = [](I64 x, I64 b) { return std::min(std::max(x, -b), b); };
  for (u64 t = 0; t < 65536; t++) {
    c.FC[t] = t + ((u64)(cl((I64)t - 32768, 2047) + 2048) << 28);
    c.FL[t] = t + ((u64)(s[t] + 4096) << 28);
    c.FS[t] = t < 32768 ? t + ((u64)g[t] << 17) : t + ((u64)(cl((I64)t - 49152, 127) + 128) << 28) + (1ull << 40);
  }
  return c;
}
// Where the indices of a call's segments lie.
struct SegPlace { std::vector<std::pair<u64, u64>> pieces; /* (first index, size), with column */ std::vector<u64> col; u64 n = 0; bool routed = false; u64 pos(u64 x) const { for (size_t i = 0; i < pieces.size(); i++) if (x >= pieces[i].first && x < pieces[i].first + pieces[i].second) return col[i] + (x - pieces[i].first); throw std::runtime_error("pos"); } };
struct FFShape {
  int lt, T, L, ll, Lam, LamM, NI, d, H, G, dh, f, fe, NE, la, XL, ls, NR, dq, dkv, nqk;
  int lmaf, NAf, lmad, NAd, lmar, lmb, lmq, lmx, hq, hv, ho, hgD, huD, hdD, hr, nmin, tau;
  bool gcl, rcl;
  long long fr(int lm) const { return 1ll << (lm - lt); }
};
inline FFShape ff_shape(const Policy &p, const ModelShape &m) {
  FFShape s{};
  s.lt = (int)p.get("lt"), s.T = 1 << s.lt, s.L = (int)p.get("L"), s.ll = (int)p.get("ll"), s.Lam = 1 << s.ll;
  s.NE = (int)p.get("NE"), s.LamM = s.NE ? s.Lam / 2 : 0, s.NI = 2 * s.Lam;
  s.d = (int)p.get("d"), s.H = (int)p.get("H"), s.G = (int)p.get("G"), s.dh = (int)p.get("dh"), s.f = (int)p.get("f"), s.fe = (int)p.get("fe");
  s.la = (int)p.get("la"), s.XL = (int)p.get("xl", 4), s.ls = s.la - s.lt, s.NR = s.XL == 5 ? 99 : 74;
  s.dq = s.H * s.dh, s.dkv = s.G * s.dh, s.nqk = s.dq + s.dkv;
  s.lmaf = (int)p.get("lmaf"), s.NAf = (int)p.get("NAf"), s.lmad = (int)p.get("lmad"), s.NAd = (int)p.get("NAd"), s.lmar = (int)p.get("lmar");
  s.lmb = (int)p.get("lmb"), s.lmq = (int)p.get("lmq"), s.lmx = (int)p.get("lmx");
  s.hq = (int)p.get("hq"), s.hv = (int)p.get("hv"), s.ho = (int)p.get("ho"), s.hgD = (int)p.get("hgD"), s.huD = (int)p.get("huD"), s.hdD = (int)p.get("hdD"), s.hr = (int)p.get("hr");
  s.nmin = (int)p.get("nmin"), s.tau = (int)p.get("tau");
  s.gcl = p.v.count("gcl") && p.v.at("gcl"), s.rcl = p.v.count("rcl") && p.v.at("rcl");
  return s;
}
// A tensor of a matrix of features.
struct Tensor2 { int off = 0, nl = 0, cnt = 0, J = 0; int feat(int lb, int ix, int j) const { return off + (lb * cnt + ix) * J + j; } };
// A matrix of features in np parts.
struct FeatMat {
  int np = 1, lm = 0, k = 0, T = 0;
  long long fr = 0;
  std::vector<CellMatrix2> part;
  void init(int np_, int lm_, long long n, int lt) {
    np = np_, lm = lm_, T = 1 << lt, fr = 1ll << (lm - lt);
    k = (int)((n + np * fr - 1) / (np * fr));
    part.assign(np, CellMatrix2(1ull << lm, k));
  }
  int32_t &cell(long long ft, int t) {
    const long long kf = k * fr;
    return part[ft / kf].at((ft % fr) * T + t, (ft / fr) % k);
  }
};
}
