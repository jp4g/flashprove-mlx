// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The rounds of a feed-forward segment (docs/format/11-2-feed-forward-segment.md, steps 1 to 19). The
// verifier's side is verifier/src/vf_layers_check.inc. A hidden value's public part comes from the
// companion, and the claims are the verifier's to form: this file only has to send true values.
#pragma once
#include <unordered_map>
#include "seg_ff_build.h"
#include "seg_frame.h"
#include "weight_units.h"
namespace fpmlx {
inline E fe_of(long long v) { return E::from(from_i64(v)); }
inline E pow2e(unsigned k) { return E::from(from_u64(1ull << k)); }
inline E dot_t(const std::vector<E> &w, const std::function<I64(int)> &f) {
  E s = E::zero();
  for (size_t t = 0; t < w.size(); t++) s = s + w[t] * fe_of(f((int)t));
  return s;
}
inline std::vector<unsigned> order_default(unsigned m) { std::vector<unsigned> o; for (unsigned b = m; b-- > 0;) o.push_back(b); return o; }
inline std::vector<unsigned> order_range(unsigned m) {
  std::vector<unsigned> o{2, 1, 0, 5, 4, 3, 8, 7, 6};
  for (unsigned b = m; b-- > 9;) o.push_back(b);
  return o;
}
// A product level on two vectors. r: the challenges by coordinate.
struct LevelOut { std::vector<E> r, values; };
template <class Ch> LevelOut product_level(Ch &ch, const E *mask, u32 &slot, std::vector<E> L, std::vector<E> R, const std::vector<E> &point, const std::vector<unsigned> &order) {
  const unsigned m = (unsigned)point.size();
  const std::vector<E> w = eq_all(point);
  PscJob j;
  for (size_t i = 0; i < w.size(); i++) j.start_value = j.start_value + w[i] * L[i] * R[i];
  for (unsigned l = 0; l < m; l++) j.vars.push_back({Weight<E>::equality(point[m - 1 - order[l]], 3), order[l]});
  j.maps = {std::move(L), std::move(R)};
  j.summand = {{E::one(), {0, 1}}};
  j.shown = 2, j.degree_y = 2;
  const PscResult r = psc_run(ch, mask, slot, std::move(j));
  (void)ch.line();
  LevelOut o;
  o.r.resize(m);
  for (unsigned l = 0; l < m; l++) o.r[m - 1 - order[l]] = r.challenges[l];
  o.values = r.values;
  return o;
}
// The panels of step 11.
struct Panel {
  bool tri = false, weights = false, virt = false, second = false;
  long virt_no = -1; // 0: Hh1; 1 + lamM: Hh2
  E ec = E::zero();
  u64 ni = 0, ny = 0, off = 0;
  unsigned wy = 0, w = 0;
  long v0 = -1;
  std::vector<E> wt, FA, FB, sl, xx; // FA, FB: ni x ny
};
struct FFProved {
  std::vector<E> rm;
};
struct FFInput {
  const Policy &pol;
  const ModelShape &m;
  const Values &V;
  const WeightInts &W;
  const Units &units;
  const int8_t *matrix;
  u64 columns;
  u64 matrix_cells;
  std::vector<u32> tokens;
};
class FFProver {
public:
  FFProver(const FFInput &in, FFGroup &G, const CallTables &tabs) : in_(in), G_(G), s_(G.s), g_(G.g) { (void)tabs; }
  // The lists of the smallness test and the arrays of moment 0. Call before the frame's first().
  void prepare(bool own_input) {
    sizes_ = lookup_sizes(G_.calls, (unsigned)in_.pol.get("lidb", 22));
    if (!lookup_multiplicities(G_.calls, sizes_, mult_)) throw std::runtime_error("a key is in no row of its table");
    const unsigned kb = sizes_.NC << (sizes_.kT + 5 - s_.lmb);
    b_ = CellMatrix2(1ull << s_.lmb, kb);
    for (unsigned c = 0; c < sizes_.NC; c++)
      for (u64 t = 0; t < (1ull << sizes_.kT); t++)
        for (unsigned j = 0; j < 32; j++)
          if ((mult_[c][t] >> j) & 1) {
            const u64 S = (((u64)c << sizes_.kT) + t) * 32 + j;
            b_.at(S & ((1ull << s_.lmb) - 1), S >> s_.lmb) = 1;
          }
    auto add = [&](const CellMatrix2 *M, bool committed) {
      if (committed) arrays["M" + std::to_string(small.mats.size())] = array_of(*M);
      small.mats.push_back(M);
    };
    for (auto &p : G_.af.part) add(&p, true);
    for (auto &p : G_.ad.part) add(&p, true);
    add(&G_.ar.part[0], true);
    add(&b_, true);
    add(&G_.qa.part[0], true);
    add(&G_.xbO.part[0], true);
    add(&G_.xbI.part[0], own_input);
    small.tau = (unsigned)in_.pol.get("tau"), small.lidb = (unsigned)in_.pol.get("lidb", 22);
  }
  Arrays arrays;
  Smallness small;
  // Steps 1 to 16. cm: this segment's entry of the index-mask message.
  void before_u3(Chan &ch, Companion &io, const E *mask, u32 &slot, const E &cm);
  // After U3's moment: RNG, then the digits. False: the proof must be made again.
  bool after_u3(ProverRounds &r) {
    small.draw_theta(r);
    arrays.clear();
    return small.digits(arrays["ZZ"], arrays["BZ"]);
  }
  void moment_u3(Companion &io) {
    const std::vector<u8> bytes = io.take(fpstream::FUNS);
    RecordReader r(bytes);
    const u64 n = r.u();
    small.parts.resize(n);
    for (U3Part &p : small.parts) {
      p.mat = (unsigned)r.u();
      (void)r.u(), (void)r.u();
      p.phi.resize(r.u());
      for (E &x : p.phi) { u64 w[3]; w[0] = r.u(), w[1] = r.u(), w[2] = r.u(); x = E{{F{w[0]}, F{w[1]}, F{w[2]}}}; }
    }
    ArrayData U;
    U.over_e = true, U.e = small.u3(), U.rows = 1, U.columns = U.e.size();
    arrays.clear();
    arrays["U3"] = std::move(U);
  }
  // Steps 18 and 19.
  void after_digits(Chan &ch, const E *mask, u32 &slot) {
    small.run_lookup(ch, mask, slot);
    (void)ch.batch_open();
  }
private:
  const FFInput &in_;
  FFGroup &G_;
  const FFShape &s_;
  int g_;
  LookupSizes sizes_;
  std::vector<std::vector<u32>> mult_;
  CellMatrix2 b_;
  std::vector<E> rV_, rtV_;
  // Helpers on values.
  std::string L(const char *n, int lam) const { return std::string(n) + "." + std::to_string(g_ * s_.Lam + lam); }
  I64 str(int sg, int t, int j) const {
    const int lam = sg / 2;
    if (sg == 2 * s_.Lam) return (g_ + 1) * s_.Lam == s_.L ? in_.V.get("x_last", t, j) : in_.V.get("x." + std::to_string((g_ + 1) * s_.Lam), t, j);
    return in_.V.get(L(sg % 2 ? "x_mid" : "x", lam), t, j);
  }
  bool moe(int lam) const { return s_.NE && (lam & 1); }
  int call_of(const std::string &seg) const {
    for (size_t c = 0; c < G_.place.size(); c++) if (G_.place[c].count(seg)) return (int)c;
    throw std::runtime_error("no call holds " + seg);
  }
};
inline void FFProver::before_u3(Chan &ch, Companion &io, const E *mask, u32 &slot, const E &cm) {
  const FFShape &s = s_;
  const int T = s.T, Lam = s.Lam, LamM = s.LamM, d = s.d, f = s.f, fe = s.fe, NE = s.NE, lt = s.lt, ll = s.ll, ls = s.ls, NI = s.NI, g = g_, XL = s.XL;
  const Values &V = in_.V;
  const unsigned lPd = clog2u(d), lPf = clog2u(f), lfe = clog2u(std::max(fe, 1)), lNI = ll + 1, NS = 2 * XL - 1;
  const u64 Pd = 1ull << lPd, Pf = 1ull << lPf;
  const bool hasE = g == 0;
  // FP_TAMPER_HIDDEN=n: a negative test, the n-th hidden value sent alone is wrong.
  const char *tamper = getenv("FP_TAMPER_HIDDEN");
  int hidden_no = 0;
  auto hide = [&](const E &v) {
    const bool bad = tamper && atoi(tamper) == hidden_no++;
    if (bad) printf("tampering with the hidden value of scalar %u\n", slot);
    ch.send(v + mask[slot] + (bad ? E::one() : E::zero()));
    slot++;
  };
  // Step 1.
  const LookupOut lk = lookup_run(ch, mask, slot, G_.calls, sizes_, mult_);
  const std::vector<E> zt(lk.zx.end() - lt, lk.zx.end());
  auto zl = [&](int c) { return std::vector<E>(lk.zk[c].begin(), lk.zk[c].begin() + ll); };
  auto zj = [&](int c) { return std::vector<E>(lk.zk[c].begin() + ll, lk.zk[c].begin() + ll + G_.lJ[c]); };
  // Steps 2, 3.
  const std::vector<E> PV = ch.point_zero(lt);
  std::vector<E> ZE, tkh;
  if (hasE) ZE = ch.point_zero(ls + lt), tkh = ch.point_zero(lPd - ls);
  const std::vector<E> ePV = eq_all(PV);
  // Step 4.
  for (int i = 0; i < NI; i++)
    for (unsigned q = 0; q < NS; q++) {
      E w = E::zero();
      for (int t = 0; t < T; t++) w = w + ePV[t] * fe_of((long long)G_.rows[i][t].W[q]);
      hide(w);
    }
  const E gam = ch.batch(NS);
  // Step 5.
  const std::vector<E> rhoI = ch.point(R_JOIN, lNI);
  // Step 6.
  {
    std::vector<E> gp(XL, E::one());
    for (int l = 1; l < XL; l++) gp[l] = gp[l - 1] * gam;
    std::vector<E> Y(1ull << (lNI + lPd + lt), E::zero());
    psc_parallel((size_t)NI * d, [&](size_t a, size_t e) {
      int32_t c[8];
      for (size_t q = a; q < e; q++) {
        const int i = (int)(q / d), j = (int)(q % d);
        for (int t = 0; t < T; t++) {
          limbs_of(str(i, t, j), XL, c);
          E y = E::zero();
          for (int l = 0; l < XL; l++) y = y + gp[l] * fe_of(c[l]);
          Y[((u64)i * Pd + j) * T + t] = y;
        }
      }
    });
    std::vector<E> rho = rhoI;
    const E half = inv(E::from(from_u64(2)));
    rho.insert(rho.end(), lPd, half);
    rho.insert(rho.end(), PV.begin(), PV.end());
    std::vector<E> Y2 = Y;
    rV_ = product_level(ch, mask, slot, std::move(Y), std::move(Y2), rho, order_default((unsigned)rho.size())).r;
  }
  // Step 7.
  auto prod = [&](const NormRow &r, int k) -> I64 {
    switch (k) {
      case 0: return r.a * r.a;
      case 1: return r.a * r.b;
      case 2: return r.b * r.b;
      case 3: return r.Rlo * r.a;
      case 4: return r.Rlo * r.b;
      case 5: return r.Rhi * r.a;
      default: return r.Rhi * r.b;
    }
  };
  if (XL != 5) throw std::runtime_error("fp_prove serves five limbs alone");
  const int NPr = 7, NO = 4;
  std::vector<E> Pv((size_t)NI * NPr);
  for (int i = 0; i < NI; i++)
    for (int k = 0; k < NPr; k++) {
      E w = E::zero();
      for (int t = 0; t < T; t++) w = w + ePV[t] * fe_of(prod(G_.rows[i][t], k));
      Pv[(size_t)i * NPr + k] = w;
      hide(w);
    }
  const E om = ch.batch((u64)NPr * NI);
  // Step 8.
  {
    std::vector<E> op((size_t)NI * NPr + 1, E::one());
    for (size_t i = 1; i < op.size(); i++) op[i] = op[i - 1] * om;
    E cl = E::zero();
    for (size_t i = 0; i < Pv.size(); i++) cl = cl + op[i] * Pv[i];
    hide(cl);
    PscJob j;
    j.maps.resize((size_t)NI * NO, std::vector<E>(T));
    for (int i = 0; i < NI; i++)
      for (int t = 0; t < T; t++) {
        const NormRow &r = G_.rows[i][t];
        j.maps[NO * i][t] = fe_of(r.a), j.maps[NO * i + 1][t] = fe_of(r.b), j.maps[NO * i + 2][t] = fe_of(r.Rlo), j.maps[NO * i + 3][t] = fe_of(r.Rhi);
      }
    for (int i = 0; i < NI; i++) {
      const unsigned a = NO * i, b = a + 1, lo = a + 2, hi = a + 3;
      const E *w = &op[(size_t)NPr * i];
      j.summand.push_back({w[0], {a, a}}), j.summand.push_back({w[1], {a, b}}), j.summand.push_back({w[2], {b, b}});
      j.summand.push_back({w[3], {lo, a}}), j.summand.push_back({w[4], {lo, b}}), j.summand.push_back({w[5], {hi, a}}), j.summand.push_back({w[6], {hi, b}});
    }
    for (int q = 0; q < lt; q++) j.vars.push_back({Weight<E>::equality(PV[q], 3), (unsigned)(lt - 1 - q)});
    j.shown = NO * NI, j.degree_y = 2, j.start_value = cl;
    rtV_ = psc_run(ch, mask, slot, std::move(j)).challenges;
  }
  // The weights of the calls' segments.
  auto ew_of = [&](const std::string &seg) {
    const int c = call_of(seg);
    const std::vector<E> e = eq_all(zj(c));
    const SegPlace &p = G_.place[c].at(seg);
    std::vector<E> w(p.n);
    for (u64 x = 0; x < p.n; x++) w[x] = e[p.pos(x)];
    return w;
  };
  auto el_of = [&](int c) { return eq_all(zl(c)); };
  auto has_seg = [&](int c, const std::string &n) { return G_.place[c].count(n) > 0; };
  auto lift2 = [&](int k) { return pow2e(lift(k)); };
  const auto &ks = in_.m.layers;
  auto layer = [&](int lam) -> const LayerScales & { return ks[g * Lam + lam]; };
  // sum_x ew[x] W[x * cols + y], for y < cols, x < n.
  auto fold_rows = [&](const std::vector<E> &ew, u64 x0, u64 n, const std::vector<int32_t> &W, u64 cols) {
    std::vector<E> out(cols, E::zero());
    psc_parallel(cols, [&](size_t a, size_t e) {
      for (size_t y = a; y < e; y++) {
        E acc = E::zero();
        for (u64 x = 0; x < n; x++) if (const int32_t v = W[x * cols + y]) acc = acc + ew[x0 + x] * fe_of(v);
        out[y] = acc;
      }
    });
    return out;
  };
  const WeightInts &W = in_.W;
  // Step 9: the routed experts' folded weights come first, since the terms use them.
  std::vector<std::vector<std::vector<E>>> FBg(LamM), FBu(LamM), FBd(LamM); // [lamM][e][yv]
  if (NE) {
    const std::vector<E> eg = ew_of("G"), eu = ew_of("U"), ed = ew_of("D");
    for (int lm = 0; lm < LamM; lm++) {
      const int l = g * Lam + 2 * lm + 1;
      for (int e = 0; e < NE; e++) {
        const std::string S = "." + std::to_string(l) + "." + std::to_string(e);
        FBg[lm].push_back(fold_rows(eg, fe, fe, W.at("Wgr" + S), d));
        FBu[lm].push_back(fold_rows(eu, fe, fe, W.at("Wur" + S), d));
        FBd[lm].push_back(fold_rows(ed, 0, d, W.at("Wdr" + S), fe));
      }
    }
  }
  auto hh = [&](int lam, int x, int t) { return V.get(L("s", lam), t, x) * V.get(L("u", lam), t, x); };
  const unsigned nR = 2 * Lam, nEnd = nR + LamM * (1 + NE);
  std::vector<std::vector<E>> shownM(nEnd, std::vector<E>(T, E::zero())), Z(nEnd, std::vector<E>(T, E::zero()));
  for (unsigned i = 0; i < nR; i++) for (int t = 0; t < T; t++) shownM[i][t] = fe_of(G_.rows[i][t].R);
  auto sc_at = [&](int lm) { return nR + lm; };
  auto sel_at = [&](int lm, int e) { return nR + LamM + lm * NE + e; };
  for (int lm = 0; lm < LamM; lm++) {
    const int lam = 2 * lm + 1;
    for (int t = 0; t < T; t++) {
      shownM[sc_at(lm)][t] = fe_of(V.get(L("sc", lam), t, 0));
      const I64 chosen = V.get(L("e_star", lam), t, 0);
      for (int e = 0; e < NE; e++) shownM[sel_at(lm, e)][t] = e == chosen ? E::one() : E::zero();
    }
  }
  struct Pair { std::vector<E> a, b; };
  std::vector<Pair> pairs; // public factor, private factor
  // Rotary tables as the forward pass has them; a layer with rp 0 has Co = 2^12, Si = 0.
  auto rope = [&](int lam, int t, int pa, bool sine) -> I64 {
    if (!layer(lam).rp) return sine ? 0 : 1 << 12;
    return W.at(sine ? "table.sin" : "table.cos")[(size_t)t * (s.dh / 2) + pa];
  };
  auto pr_of = [&](int lam, int hd, int ch2, int t) -> I64 {
    return hd < s.H ? V.get(L("sum_preq", lam), t, hd * s.dh + ch2) : V.get(L("sum_prek", lam), t, (hd - s.H) * s.dh + ch2);
  };
  for (size_t c = 0; c < in_.pol.call.size(); c++) {
    const std::vector<E> el = el_of((int)c);
    const E ec = lk.e[c];
    for (const std::string &sn : in_.pol.call[c].segs) {
      if (sn == "N1" || sn == "N2") {
        const int n = sn == "N2";
        const std::vector<E> ew = ew_of(sn);
        for (int lam = 0; lam < Lam; lam++) {
          const int i = 2 * lam + n;
          for (int t = 0; t < T; t++) {
            E a = E::zero();
            for (int x = 0; x < d; x++) a = a + ew[x] * fe_of(str(i, t, x));
            Z[i][t] = Z[i][t] + ec * el[lam] * a;
          }
        }
      } else if (sn == "QK") {
        const std::vector<E> ew = ew_of(sn);
        for (int lam = 0; lam < Lam; lam++) {
          const E co = ec * el[lam] * lift2(layer(lam).kq);
          for (int pa = 0; pa < s.dh / 2; pa++) {
            Pair pc{std::vector<E>(T), std::vector<E>(T, E::zero())}, ps{std::vector<E>(T), std::vector<E>(T, E::zero())};
            for (int t = 0; t < T; t++) {
              pc.a[t] = fe_of(rope(lam, t, pa, false)), ps.a[t] = fe_of(rope(lam, t, pa, true));
              for (int hd = 0; hd < s.H + s.G; hd++) {
                const E w0 = ew[hd * s.dh + 2 * pa], w1 = ew[hd * s.dh + 2 * pa + 1];
                const E p0 = fe_of(pr_of(lam, hd, 2 * pa, t)), p1 = fe_of(pr_of(lam, hd, 2 * pa + 1, t));
                pc.b[t] = pc.b[t] + co * (w0 * p0 + w1 * p1);
                ps.b[t] = ps.b[t] + co * (w1 * p0 - w0 * p1);
              }
            }
            pairs.push_back(std::move(pc)), pairs.push_back(std::move(ps));
          }
        }
      } else if (!NE) {
        continue;
      } else if (sn == "XR") {
        const std::vector<E> ew = ew_of(sn);
        for (int lm = 0; lm < LamM; lm++) {
          const int lam = 2 * lm + 1;
          for (int t = 0; t < T; t++) {
            E a = E::zero();
            for (int x = 0; x < d; x++) a = a + ew[x] * fe_of(V.get(L("m", lam), t, x));
            Z[sc_at(lm)][t] = Z[sc_at(lm)][t] + ec * el[lam] * a;
          }
        }
      } else if (sn == "DD" || sn == "MR") {
        const std::vector<E> ew = ew_of(sn);
        E wS = E::zero();
        if (sn == "DD") for (const E &x : ew) wS = wS + x;
        else wS = ew[0];
        for (int lm = 0; lm < LamM; lm++) {
          const int lam = 2 * lm + 1;
          for (int e = 0; e < NE; e++)
            for (int t = 0; t < T; t++) Z[sel_at(lm, e)][t] = Z[sel_at(lm, e)][t] + ec * el[lam] * wS * fe_of(V.get(L("lr", lam), t, e));
        }
      } else if (sn == "G" || sn == "U") {
        for (int lm = 0; lm < LamM; lm++) {
          const int lam = 2 * lm + 1;
          const E co = ec * el[lam] * lift2(sn == "U" ? layer(lam).kuR : layer(lam).kgR);
          const auto &FB = sn == "U" ? FBu[lm] : FBg[lm];
          for (int e = 0; e < NE; e++)
            for (int t = 0; t < T; t++) {
              E a = E::zero();
              for (int y = 0; y < d; y++) a = a + FB[e][y] * fe_of(V.get(L("xr", lam), t, y));
              Z[sel_at(lm, e)][t] = Z[sel_at(lm, e)][t] + co * a;
            }
        }
      } else if (sn == "D") {
        for (int lm = 0; lm < LamM; lm++) {
          const int lam = 2 * lm + 1;
          const E co = ec * el[lam] * lift2(layer(lam).kd);
          for (int e = 0; e < NE; e++)
            for (int t = 0; t < T; t++) {
              E a = E::zero();
              for (int y = 0; y < fe; y++) a = a + FBd[lm][e][y] * fe_of(hh(lam, fe + y, t));
              Z[sel_at(lm, e)][t] = Z[sel_at(lm, e)][t] + co * a;
            }
        }
      }
    }
  }
  const std::vector<E> ezt = eq_all(zt);
  PscResult r9;
  {
    PscJob j;
    j.maps = shownM;
    for (unsigned i = 0; i < nEnd; i++) {
      const unsigned at = (unsigned)j.maps.size();
      j.maps.push_back(Z[i]);
      j.summand.push_back({E::one(), {i, at}});
    }
    for (Pair &p : pairs) {
      const unsigned at = (unsigned)j.maps.size();
      j.maps.push_back(p.a), j.maps.push_back(p.b);
      j.summand.push_back({E::one(), {at, at + 1}});
    }
    E cl = E::zero();
    for (int t = 0; t < T; t++) {
      E h = E::zero();
      for (const Monomial &m : j.summand) h = h + j.maps[m.maps[0]][t] * j.maps[m.maps[1]][t];
      cl = cl + ezt[t] * h;
    }
    hide(cl);
    for (int q = 0; q < lt; q++) j.vars.push_back({Weight<E>::equality(zt[q], 3), (unsigned)(lt - 1 - q)});
    j.shown = nEnd, j.degree_y = 1, j.start_value = cl;
    r9 = psc_run(ch, mask, slot, std::move(j));
  }
  const std::vector<E> &rt = r9.challenges;
  const std::vector<E> ert = eq_all(rt);
  // Step 10.
  const E om1 = ch.batch(2);
  // Step 11: the panels, in the order of the list.
  std::vector<Panel> bl;
  u32 nextV = 0;
  auto dotz = [&](const std::vector<E> &w, const std::string &name, int x) { return dot_t(w, [&](int t) { return V.get(name, t, x); }); };
  auto rsv = [&](const std::vector<E> &w, const char *n, int lam, u64 ny) {
    std::vector<E> v(ny);
    psc_parallel(ny, [&](size_t a, size_t e) { for (size_t y = a; y < e; y++) v[y] = dotz(w, L(n, lam), (int)y); });
    return v;
  };
  auto weights_panel = [&](bool moe_set, u64 ny, const E &ec, const std::function<std::vector<E>(int)> &FA, const std::function<std::vector<E>(int)> &FB, const std::function<E(int)> &wt) {
    Panel b;
    b.weights = true, b.ec = ec;
    b.ni = moe_set ? LamM : Lam, b.ny = ny, b.wy = clog2u(ny), b.w = clog2u(b.ni) + b.wy;
    for (u64 i = 0; i < b.ni; i++) {
      const int lam = moe_set ? 2 * (int)i + 1 : (int)i;
      if (FA) { const auto a = FA(lam); b.FA.insert(b.FA.end(), a.begin(), a.end()); }
      const auto f2 = FB(lam);
      b.FB.insert(b.FB.end(), f2.begin(), f2.end());
      b.wt.push_back(wt(lam));
    }
    b.v0 = nextV, nextV += (u32)b.ni;
    bl.push_back(std::move(b));
  };
  auto lname = [&](const char *n, int lam) { return std::string(n) + "." + std::to_string(g * Lam + lam); };
  for (size_t c = 0; c < in_.pol.call.size(); c++) {
    const std::vector<E> el = el_of((int)c);
    const E ec = lk.e[c];
    for (const std::string &sn : in_.pol.call[c].segs) {
      if (sn == "G" || sn == "U") {
        const bool up = sn == "U";
        const std::vector<E> ew = ew_of(sn);
        weights_panel(false, d, ec, [&](int lam) { return rsv(ezt, "m", lam, d); },
                      [&](int lam) { return fold_rows(ew, 0, moe(lam) ? fe : f, W.at(lname(up ? "Wu" : "Wg", lam)), d); },
                      [&](int lam) { return el[lam] * lift2(up ? layer(lam).ku : layer(lam).kg); });
      } else if (sn == "V") {
        const std::vector<E> ew = ew_of(sn);
        weights_panel(false, d, ec, [&](int lam) { return rsv(ezt, "a", lam, d); }, [&](int lam) { return fold_rows(ew, 0, s.dkv, W.at(lname("Wv", lam)), d); },
                      [&](int lam) { return el[lam] * lift2(layer(lam).kv); });
      } else if (sn == "O") {
        const std::vector<E> ew = ew_of(sn);
        weights_panel(false, s.dq, ec, [&](int lam) { return rsv(ezt, "o", lam, s.dq); }, [&](int lam) { return fold_rows(ew, 0, d, W.at(lname("Wo", lam)), s.dq); },
                      [&](int lam) { return el[lam] * lift2(layer(lam).ko); });
      } else if (sn == "D") {
        const std::vector<E> ew = ew_of(sn);
        weights_panel(false, f, ec, [&](int lam) {
                        std::vector<E> v(f);
                        psc_parallel(f, [&](size_t a, size_t e) { for (size_t y = a; y < e; y++) v[y] = dot_t(ezt, [&](int t) { return hh(lam, (int)y, t); }); });
                        return v;
                      },
                      [&](int lam) {
                        const u64 cols = moe(lam) ? fe : f;
                        std::vector<E> v = fold_rows(ew, 0, d, W.at(lname("Wd", lam)), cols);
                        v.resize(f, E::zero());
                        return v;
                      },
                      [&](int lam) { return el[lam] * lift2(layer(lam).kd); });
        bl.back().virt = true, bl.back().virt_no = 0;
      } else if (sn == "LR") {
        const std::vector<E> ew = ew_of(sn);
        weights_panel(true, d, ec, [&](int lam) { return rsv(ezt, "m", lam, d); }, [&](int lam) { return fold_rows(ew, 0, NE, W.at(lname("Wr", lam)), d); },
                      [&](int lam) { return el[lam] * lift2(layer(lam).kr); });
      }
    }
  }
  const std::vector<E> &sh = r9.end_values;
  // The rotary factors at rt, and the first-factor reads of the second pass.
  std::vector<std::vector<E>> FBqk(Lam);
  for (size_t c = 0; c < in_.pol.call.size(); c++) {
    const std::vector<E> el = el_of((int)c);
    const E ec = lk.e[c];
    for (const std::string &sn : in_.pol.call[c].segs) {
      auto pair_panel = [&](bool moe_set, u64 ny, const std::function<E(int, u64)> &FA, const std::function<E(int, u64)> &FB, const std::function<E(int)> &wt) {
        Panel b;
        b.second = true, b.ec = ec;
        b.ni = moe_set ? LamM : Lam, b.ny = ny, b.wy = clog2u(ny), b.w = clog2u(b.ni) + b.wy;
        for (u64 i = 0; i < b.ni; i++) {
          const int lam = moe_set ? 2 * (int)i + 1 : (int)i;
          for (u64 y = 0; y < ny; y++) b.FA.push_back(FA(lam, y)), b.FB.push_back(FB(lam, y));
          b.wt.push_back(wt(lam));
        }
        bl.push_back(std::move(b));
      };
      if (sn == "N1" || sn == "N2") {
        const int n = sn == "N2";
        const std::vector<E> ew = ew_of(sn);
        pair_panel(false, 1, [&](int lam, u64) { return sh[2 * lam + n]; },
                   [&](int lam, u64) {
                     E a = E::zero();
                     for (int x = 0; x < d; x++) a = a + ew[x] * dot_t(ert, [&](int t) { return str(2 * lam + n, t, x); });
                     return a;
                   },
                   [&](int lam) { return el[lam]; });
      } else if (sn == "QK") {
        const std::vector<E> ew = ew_of(sn);
        weights_panel(false, d, ec, [&](int lam) { return rsv(ert, "a", lam, d); },
                      [&](int lam) {
                        std::vector<E> w((size_t)(s.H + s.G) * s.dh, E::zero());
                        for (int pa = 0; pa < s.dh / 2; pa++) {
                          const E Ct = dot_t(ert, [&](int t) { return rope(lam, t, pa, false); }), St = dot_t(ert, [&](int t) { return rope(lam, t, pa, true); });
                          for (int hd = 0; hd < s.H + s.G; hd++) {
                            const E w0 = ew[hd * s.dh + 2 * pa], w1 = ew[hd * s.dh + 2 * pa + 1];
                            w[hd * s.dh + 2 * pa] = Ct * w0 + St * w1;
                            w[hd * s.dh + 2 * pa + 1] = Ct * w1 - St * w0;
                          }
                        }
                        std::vector<E> a = fold_rows(w, 0, s.dq, W.at(lname("Wq", lam)), d), b = fold_rows(w, s.dq, s.dkv, W.at(lname("Wk", lam)), d);
                        for (int y = 0; y < d; y++) a[y] = a[y] + b[y];
                        return a;
                      },
                      [&](int lam) { return el[lam] * lift2(layer(lam).kq); });
        bl.back().second = true;
      } else if (!NE) {
        continue;
      } else if (sn == "XR") {
        const std::vector<E> ew = ew_of(sn);
        pair_panel(true, 1, [&](int lam, u64) { return sh[sc_at(lam / 2)]; },
                   [&](int lam, u64) {
                     E a = E::zero();
                     for (int x = 0; x < d; x++) a = a + ew[x] * dotz(ert, L("m", lam), x);
                     return a;
                   },
                   [&](int lam) { return el[lam]; });
      } else if (sn == "DD" || sn == "MR") {
        const std::vector<E> ew = ew_of(sn);
        E wS = E::zero();
        if (sn == "DD") for (const E &x : ew) wS = wS + x;
        else wS = ew[0];
        pair_panel(true, NE, [&](int lam, u64 e) { return sh[sel_at(lam / 2, (int)e)]; }, [&](int lam, u64 e) { return dotz(ert, L("lr", lam), (int)e); },
                   [&](int lam) { return el[lam] * wS; });
      } else if (sn == "G" || sn == "U" || sn == "D") {
        for (int lm = 0; lm < LamM; lm++) {
          const int lam = 2 * lm + 1;
          Panel b;
          b.tri = true, b.second = true, b.ec = ec;
          b.ni = NE, b.ny = sn == "D" ? fe : d, b.wy = clog2u(b.ny), b.w = clog2u(NE) + b.wy;
          for (int e = 0; e < NE; e++) b.sl.push_back(sh[sel_at(lm, e)]);
          const auto &FB = sn == "G" ? FBg[lm] : sn == "U" ? FBu[lm] : FBd[lm];
          for (int e = 0; e < NE; e++) b.FB.insert(b.FB.end(), FB[e].begin(), FB[e].end());
          if (sn == "D") {
            b.virt = true, b.virt_no = 1 + lm;
            b.xx.resize(fe);
            psc_parallel(fe, [&](size_t a, size_t e) { for (size_t y = a; y < e; y++) b.xx[y] = dot_t(ert, [&](int t) { return hh(lam, fe + (int)y, t); }); });
            b.wt.push_back(el[lam] * lift2(layer(lam).kd));
          } else {
            b.xx = rsv(ert, "xr", lam, d);
            b.wt.push_back(el[lam] * lift2(sn == "U" ? layer(lam).kuR : layer(lam).kgR));
          }
          b.v0 = nextV++;
          bl.push_back(std::move(b));
        }
      }
    }
  }
  std::vector<size_t> ord(bl.size());
  for (size_t i = 0; i < ord.size(); i++) ord[i] = i;
  std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return bl[a].w > bl[b].w; });
  u64 off = 0;
  for (size_t i : ord) bl[i].off = off, off += 1ull << bl[i].w;
  const unsigned lamS = clog2u(off);
  std::vector<E> SA(1ull << lamS, E::zero()), SB(1ull << lamS, E::zero());
  for (const Panel &b : bl) {
    const E om_p = b.second ? om1 : E::one();
    for (u64 i = 0; i < b.ni; i++)
      for (u64 y = 0; y < b.ny; y++) {
        const u64 n = b.off + (i << b.wy) + y;
        if (b.tri) SA[n] = om_p * b.ec * b.wt[0] * b.sl[i] * b.xx[y], SB[n] = b.FB[i * b.ny + y];
        else if (b.virt) SA[n] = b.FA[i * b.ny + y], SB[n] = b.ec * b.wt[i] * b.FB[i * b.ny + y];
        else SA[n] = om_p * b.ec * b.wt[i] * b.FA[i * b.ny + y], SB[n] = b.FB[i * b.ny + y];
      }
  }
  const unsigned NVIRT = 1 + LamM, NCLp = nextV;
  std::vector<E> rs(lamS), Hv(NVIRT), vA(NCLp), aH(NVIRT, E::zero()), bV(NCLp, E::zero());
  // The Hh tables for steps 12 and 14.
  auto eq_low = [&](unsigned lo, unsigned hi) { std::vector<E> z; for (unsigned v = hi; v-- > lo;) z.push_back(rs[v]); return eq_all(z); };
  auto eq_low_at = [&](unsigned lo, unsigned hi, u64 c) { std::vector<E> z; for (unsigned v = hi; v-- > lo;) z.push_back(rs[v]); return eq_at(z, c); };
  PscJob j11;
  for (unsigned l = 0; l < lamS; l++) j11.vars.push_back({Weight<E>::plain(3), lamS - 1 - l});
  for (size_t i = 0; i < SA.size(); i++) j11.start_value = j11.start_value + SA[i] * SB[i];
  j11.maps = {std::move(SA), std::move(SB)};
  j11.summand = {{E::one(), {0, 1}}};
  j11.shown = 2 + NVIRT + NCLp, j11.degree_y = 2;
  j11.ends = [&](const std::vector<E> &chs, const std::vector<E> &bound) {
    for (unsigned l = 0; l < lamS; l++) rs[lamS - 1 - l] = chs[l];
    E A0 = bound[0], B0 = bound[1];
    for (const Panel &b : bl) {
      const E om_p = b.second ? om1 : E::one();
      const E sb = eq_low_at(b.w, lamS, b.off >> b.w);
      const std::vector<E> EY = eq_low(0, b.wy), EI = eq_low(b.wy, b.w);
      const E co = sb * om_p * b.ec;
      if (b.tri) {
        E s1 = E::zero();
        for (int e = 0; e < NE; e++) s1 = s1 + b.sl[e] * EI[e];
        E v = E::zero();
        for (int e = 0; e < NE; e++) for (u64 y = 0; y < b.ny; y++) v = v + EI[e] * EY[y] * b.FB[e * b.ny + y];
        vA[b.v0] = v, bV[b.v0] = sb;
        if (b.virt) {
          E h = E::zero();
          for (u64 y = 0; y < b.ny; y++) h = h + EY[y] * b.xx[y];
          Hv[b.virt_no] = h, aH[b.virt_no] = co * b.wt[0] * s1;
        }
        continue;
      }
      if (b.v0 < 0) continue;
      for (u64 i = 0; i < b.ni; i++) {
        E v = E::zero();
        for (u64 y = 0; y < b.ny; y++) v = v + EY[y] * b.FB[i * b.ny + y];
        vA[b.v0 + i] = v;
        bV[b.v0 + i] = b.virt ? sb * EI[i] * b.ec * b.wt[i] : sb * EI[i];
      }
      if (b.virt) {
        E h = E::zero();
        for (u64 i = 0; i < b.ni; i++) for (u64 y = 0; y < b.ny; y++) h = h + EI[i] * EY[y] * b.FA[i * b.ny + y];
        Hv[0] = h, aH[0] = sb * om_p;
      }
    }
    for (unsigned k = 0; k < NVIRT; k++) A0 = A0 - aH[k] * Hv[k];
    for (unsigned a = 0; a < NCLp; a++) B0 = B0 - bV[a] * vA[a];
    std::vector<E> out{A0};
    out.insert(out.end(), Hv.begin(), Hv.end());
    out.push_back(B0);
    out.insert(out.end(), vA.begin(), vA.end());
    return out;
  };
  j11.at_end = [&](const std::vector<E> &v) {
    E As = v[0], Bs = v[1 + NVIRT];
    for (unsigned k = 0; k < NVIRT; k++) As = As + aH[k] * v[1 + k];
    for (unsigned a = 0; a < NCLp; a++) Bs = Bs + bV[a] * v[2 + NVIRT + a];
    return As * Bs;
  };
  (void)psc_run(ch, mask, slot, std::move(j11));
  // Step 12.
  auto minus_table = [&](u64 size, const std::function<void(std::vector<E> &, std::vector<E> &)> &fill) {
    std::pair<std::vector<E>, std::vector<E>> p{std::vector<E>(size, E::zero()), std::vector<E>(size, E::zero())};
    fill(p.first, p.second);
    return p;
  };
  {
    auto p = minus_table(1ull << (ll + lPf + lt), [&](std::vector<E> &a, std::vector<E> &b) {
      for (int lam = 0; lam < Lam; lam++)
        for (int x = 0; x < f; x++)
          for (int t = 0; t < T; t++) {
            const u64 i = ((u64)lam * Pf + x) * T + t;
            a[i] = fe_of(-V.get(L("s", lam), t, x)), b[i] = fe_of(-V.get(L("u", lam), t, x));
          }
    });
    std::vector<E> rho;
    for (unsigned v = ll + lPf; v-- > 0;) rho.push_back(rs[v]);
    rho.insert(rho.end(), zt.begin(), zt.end());
    (void)product_level(ch, mask, slot, std::move(p.first), std::move(p.second), rho, order_range((unsigned)rho.size()));
  }
  // Steps 13, 14.
  if (NE) {
    std::vector<E> rho;
    if (ll > 1) rho = ch.point(R_JOIN, ll - 1);
    auto p = minus_table(1ull << (ll - 1 + lfe + lt), [&](std::vector<E> &a, std::vector<E> &b) {
      for (int lm = 0; lm < LamM; lm++)
        for (int x = 0; x < fe; x++)
          for (int t = 0; t < T; t++) {
            const u64 i = ((u64)lm * fe + x) * T + t;
            const int lam = 2 * lm + 1;
            a[i] = fe_of(-V.get(L("s", lam), t, fe + x)), b[i] = fe_of(-V.get(L("u", lam), t, fe + x));
          }
    });
    for (unsigned v = lfe; v-- > 0;) rho.push_back(rs[v]);
    rho.insert(rho.end(), rt.begin(), rt.end());
    (void)product_level(ch, mask, slot, std::move(p.first), std::move(p.second), rho, order_range((unsigned)rho.size()));
  }
  // Step 15.
  if (hasE) {
    const std::vector<E> zsE(ZE.begin(), ZE.begin() + ls), ztE(ZE.begin() + ls, ZE.end());
    std::vector<E> pnt = tkh;
    pnt.insert(pnt.end(), zsE.begin(), zsE.end());
    const std::vector<E> phi = eq_all(pnt), et = eq_all(ztE);
    const std::vector<int32_t> &Emb = W.at("Emb");
    E vE = E::zero();
    for (int t = 0; t < T; t++) {
      E a = E::zero();
      for (int k = 0; k < d; k++) a = a + phi[k] * fe_of(Emb[(size_t)in_.tokens[t] * d + k]);
      vE = vE + et[t] * a;
    }
    vA.push_back(vE);
    hide(vE);
  }
  // Step 16: the merge.
  {
    const std::vector<u8> bytes = io.take(fpstream::CLAIMS);
    const std::vector<WeightClaim> claims = read_claims(bytes);
    if (claims.size() != vA.size()) throw std::runtime_error("the merge: another number of claims");
    const E x = ch.batch(claims.size());
    PscJob j;
    const unsigned lk2 = in_.units.lk;
    for (unsigned l = 0; l < lk2; l++) j.vars.push_back({Weight<E>::plain(3), lk2 - 1 - l});
    E xp = E::one();
    for (size_t a = 0; a < claims.size(); a++, xp = xp * x) {
      std::vector<E> Ea, Ba;
      in_.units.claim_maps(claims[a], in_.matrix, in_.matrix_cells, in_.columns, Ea, Ba);
      const unsigned at = (unsigned)j.maps.size();
      j.maps.push_back(std::move(Ea)), j.maps.push_back(std::move(Ba));
      j.summand.push_back({xp, {at, at + 1}});
      j.start_value = j.start_value + xp * vA[a];
    }
    const std::vector<Monomial> sum = j.summand;
    j.ends = [sum](const std::vector<E> &, const std::vector<E> &b) {
      E v = E::zero();
      for (const Monomial &m : sum) v = v + m.coeff * b[m.maps[0]] * b[m.maps[1]];
      return std::vector<E>{v};
    };
    j.at_end = [](const std::vector<E> &v) { return v[0]; };
    j.end_masks = {cm};
    j.shown = 1, j.degree_y = 1;
    (void)psc_run(ch, mask, slot, std::move(j));
  }
}
}
