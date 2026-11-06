// The output head's witness on the reference's model: every key is a row of its call's table, the
// policy's equations hold in integers, and, with moment 0 run on a mock channel, the key forms and the
// claims that read U3 agree with the values that the sub-proof holds.
// usage: head_keys_test <directory of make_reference.py> <policy.txt> <request file> [file for the grown request]
#include <cstdio>
#include <random>
#include "seg_head_proof.h"
using namespace fpmlx;
static std::string slurp(const char *p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
struct Mock {
  std::mt19937_64 g{12345};
  u64 sent = 0;
  E draw() { auto r = [&] { return from_u64(g() >> 2); }; return E::from(r(), r(), r()); }
  void send(const E &) { sent++; }
  void send_hiding_sum(const E &) { sent++; }
  E epsilon() { return draw(); }
  E round() { return draw(); }
  E y() { return draw(); }
  void values() {}
  E shift() { return draw(); }
  E alpha() { return draw(); }
  E line() { return draw(); }
  std::vector<E> point(const char *, size_t n) { std::vector<E> v(n); for (E &x : v) x = draw(); return v; }
  std::vector<E> point_join(size_t n) { return point("", n); }
  std::vector<E> point_zero(size_t n) { return point("", n); }
  E batch(u64) { return draw(); }
  E batch_open() { return draw(); }
  E public_part(const E &x) { return x; }
};
static int bad = 0;
static void check(const char *name, bool ok) { printf("%-44s %s\n", name, ok ? "ok" : "FAILED"); bad += !ok; }
static bool same(const E &a, const E &b) { const E z = a - b; return z == E::zero(); }
int main(int c, char **v) {
  const std::string dir = v[1];
  const ModelShape m = read_shape(dir + "/params.txt");
  auto w = mx::load_safetensors(dir + "/weights.safetensors").first;
  for (auto &[k, a] : w) if (k.rfind("table", 0) == 0) a = mx::astype(a, mx::int64);
  const Policy pol = Policy::read(slurp(v[2]));
  const unsigned T = 1u << pol.get("lt");
  HeadRequest q = HeadRequest::read(slurp(v[3]), T);
  std::vector<int64_t> tk(q.tokens.begin(), q.tokens.end());
  {
    mx::array rt = mx::astype(w.at("tokens"), mx::int64);
    mx::eval(rt);
    unsigned differ = 0;
    for (unsigned t = 0; t < T; t++) differ += rt.data<int64_t>()[t] != tk[t];
    printf("request r=%u e=%u; the reference's tokens differ from the request's at %u of %u places\n", q.r, q.e, differ, T);
  }
  // own_choices.req goes with `-weights made-up-blocks`, not with the reference's generated weights: under
  // these, its reply is no first choice. So the prompt stays, and the reply is grown on this model: each
  // token from r + 1 on is the first choice of the position before it (the model is causal).
  Values V;
  { unsigned hit = 0; V = capture_forward(m, w, mx::array(tk.data(), {(int)T}, mx::int64)); for (unsigned t = q.r; t + 1 < std::min(q.e, T); t++) hit += V.get("first_choice", t, 0) == tk[t + 1]; printf("the request's own reply: first choice at %u places\n", hit); }
  for (unsigned t = q.r; t + 1 < std::min(q.e, T); t++) {
    tk[t + 1] = V.get("first_choice", t, 0);
    V = capture_forward(m, w, mx::array(tk.data(), {(int)T}, mx::int64));
  }
  for (unsigned t = 0; t < T; t++) q.tokens[t] = (u32)tk[t];
  if (c > 4) { std::ofstream o(v[4], std::ios::binary); o.write((const char *)q.tokens.data(), 4ull * T); o.write((const char *)&q.r, 4); o.write((const char *)&q.e, 4); }
  q.y = head_next_token(V, q, T);
  const CallTables tabs = call_tables(w);
  HeadWitness W = head_build(pol, m, V, tabs, q);
  { // The GPU's witness against the host's, word by word.
    const HeadWitness B = head_build(pol, m, V, tabs, q, false);
    size_t diff = 0, n = 0;
    auto cmp = [&](const FeatMat &x, const FeatMat &y) { for (size_t p = 0; p < x.part.size(); p++) { n += x.part[p].cell.size(); for (size_t i = 0; i < x.part[p].cell.size(); i++) diff += x.part[p].cell[i] != y.part[p].cell[i]; } };
    cmp(W.hf, B.hf), cmp(W.hd, B.hd), cmp(W.hr, B.hr), cmp(W.xbL, B.xbL);
    size_t dk = 0, nk = 0;
    for (size_t c = 0; c < W.calls.size(); c++) { nk += W.calls[c].keys.size(); for (size_t i = 0; i < W.calls[c].keys.size(); i++) dk += W.calls[c].keys[i] != B.calls[c].keys[i] || W.calls[c].padding[i] != B.calls[c].padding[i]; }
    bool same = W.price == B.price && W.hidden == B.hidden && W.mult == B.mult && W.str == B.str && W.act == B.act && W.lam == B.lam && W.norm_ok == B.norm_ok && W.margins_ok == B.margins_ok && W.bounds_ok == B.bounds_ok && W.keys_ok == B.keys_ok && W.bH.cell == B.bH.cell;
    for (int t = 0; t < W.s.T; t++) { const NormRow &x = W.rows[t], &y = B.rows[t]; same = same && x.D == y.D && x.a == y.a && x.b == y.b && x.sig == y.sig && x.R == y.R && x.Rlo == y.Rlo && x.Rhi == y.Rhi && x.W == y.W; }
    printf("head: %zu of %zu cells differ, %zu of %zu keys differ, the values are %s\n", diff, n, dk, nk, same ? "equal" : "NOT EQUAL");
    if (diff || dk || !same) return 1;
  }
  const HeadShape2 &s = W.s;
  for (size_t i = 0; i < W.calls.size(); i++) {
    std::vector<LookupCall> one{W.calls[i]};
    LookupSizes sz = lookup_sizes(one, 28);
    std::vector<std::vector<u32>> mult;
    const bool ok = lookup_multiplicities(one, sz, mult);
    u64 live = 0; for (char p : W.calls[i].padding) live += !p;
    printf("call %s: 2^%u keys, %llu live: %s\n", pol.callH[i].name.c_str(), W.calls[i].n, (unsigned long long)live, ok ? "every key is in the table" : "A KEY IS IN NO ROW");
    bad += !ok;
  }
  check("norm: R, sig and remainders", W.norm_ok);
  check("margins in reach", W.margins_ok);
  check("bounds theta and cap", W.bounds_ok);
  { unsigned hit = 0, n = 0; for (unsigned t = 0; t < T; t++) if (W.co[t]) n++, hit += (u32)V.get("first_choice", t, 0) == W.win[t]; printf("the checked token is the model's first choice at %u of %u positions with a cost\n", hit, n); }
  { I64 mn = 1ll << 60, mxp = 0; for (unsigned t = 0; t < T; t++) if (W.co[t]) mn = std::min(mn, W.price[t]), mxp = std::max(mxp, W.price[t]); printf("costs: least %lld, most %lld, y=%u\n", (long long)mn, (long long)mxp, q.y); }
  // The policy's equations, in integers.
  {
    I64 b2 = 0, b3 = 0, RC = 0; bool b4 = true, z = true;
    for (unsigned t = 0; t < T; t++) {
      if (!W.co[t]) continue;
      RC++;
      const auto &h = W.hidden[t];
      b2 += h[0] + s.C * h[1] + h[2], b3 += h[1] + h[4], b4 = b4 && h[0] + h[3] == s.C;
      I64 sum = h[5];
      for (int x = 0; x < s.JV; x++) { I64 e = s.offE(); e += W.hf.cell(4ll * s.d + 3 * s.JV + x, t) + 256ll * W.hf.cell(4ll * s.d + 4 * s.JV + x, t) + 65536ll * W.hf.cell(4ll * s.d + 5 * s.JV + x, t); sum += e; }
      z = z && sum == ((I64)(s.JV - s.V + 1) << s.F);
    }
    check("(B2) in integers", b2 == s.theta * RC);
    check("(B3) in integers", b3 == (RC ? s.cap : 0));
    check("(B4) in integers", b4);
    check("(Z) in integers", z);
  }
  // Moment 0 on the mock.
  mx::array wh = mx::astype(w.at("Wh"), mx::int32);
  mx::eval(wh);
  const std::vector<int32_t> Wh(wh.data<int32_t>(), wh.data<int32_t>() + wh.size());
  HeadProof P(W, Wh);
  Mock ch;
  std::vector<E> mask(1 << 16);
  for (E &x : mask) x = ch.draw();
  u32 slot = 0;
  P.moment0(ch, mask.data(), slot);
  check("every scalar has a number", ch.sent == slot);
  printf("moment 0: %u scalars\n", slot);
  const std::vector<E> U = P.u3();
  check("NHf is 1 here", s.NHf == 1);
  std::vector<u64> at(16, 0);
  { u64 o = 0; for (size_t i = 0; i < P.small.parts.size(); i++) { at[i + 1] = o; const CellMatrix2 &M = *P.small.mats[P.small.parts[i].mat]; o += M.k * (M.m / P.small.parts[i].phi.size()); } printf("U3: %llu entries\n", (unsigned long long)o); check("U3 has its entries", o == U.size()); }
  auto part = [&](int n, u64 i) { return U[at[n] + i]; };
  auto Fi = [](I64 x) { return E::from(from_i64(x)); };
  const int d = s.d, JV = s.JV, JVa = s.JVa, S2 = s.S2, NR = s.NR;
  const std::vector<E> eZT = eq_all(P.zt);
  E ON = E::zero(), ERC = E::zero(), RCe = E::zero();
  for (unsigned t = 0; t < T; t++) { if (W.on[t]) ON = ON + eZT[t]; if (W.co[t]) ERC = ERC + eZT[t], RCe = RCe + E::one(); }
  auto wsum = [&](const std::string &n, int lim) { std::vector<E> e = P.ew(n); E a = E::zero(); for (size_t x = 0; x < e.size() && (int)x < lim; x++) a = a + e[x]; return a; };
  auto SL = [&](const std::string &sg, long long f0, int nl, long long fs) {
    const int cc = W.call_of(sg);
    std::vector<E> z = P.zj(cc);
    z.resize(z.size() - s.ls);
    const std::vector<E> eh = eq_all(z);
    const SegPlace &pl = W.place[cc].at(sg);
    E a = E::zero(), sc = E::one();
    for (int lb = 0; lb < nl; lb++, sc = sc * Fi(256))
      for (u64 i = 0; i < pl.n / S2; i++) a = a + sc * eh[pl.pos(S2 * i) / S2] * part(1, (f0 + lb * fs) / S2 + i);
    return a;
  };
  static const int dg0[6] = {0, 5, 6, 11, 16, 18}, dgn[6] = {5, 1, 5, 5, 2, 3};
  auto HN = [&](int n, int i) { const E ws = n == 11 ? ERC : RCe; E a = E::zero(), sc = E::one(); for (int lb = 0; lb < dgn[i]; lb++, sc = sc * Fi(256)) a = a + sc * (part(n, NR + dg0[i] + lb) + Fi(128) * ws); return a; };
  auto A2 = [&](int n, int k) { return part(n, k) + Fi(256) * part(n, d + k); };
  // Forms of the key.
  const long long sh = s.sh;
  const E iw = inv(Fi(1ll << (sh + 15)));
  E place = E::zero();
  { const std::vector<E> e0 = P.ew("HD0"); for (unsigned t = 0; t < T; t++) if (W.on[t]) { E a = E::zero(); for (u32 x = 0; x < W.win[t]; x++) a = a + e0[x]; place = place + eZT[t] * a; } }
  const E BIGe = Fi(1ll << 37), Pw = Fi(1ll << s.lPd);
  auto KS = [&](const std::string &n) -> E {
    const long long KF = s.KF, offE = s.offE();
    if (n == "NF") return wsum(n, 1 << 30) * (Fi(1ll << (KF - 1)) - Fi(32896ll << 16)) - Fi(1ll << KF) * SL(n, 0, 2, d) - Fi(1ll << 16) * SL(n, 2 * d, 2, d);
    if (n == "NFH") return Fi(1ll << (32 - KF)) * (Fi(32896) * wsum(n, 1 << 30) + SL(n, 2 * d, 2, d));
    if (n == "HD1") return Fi(32896) * wsum(n, s.V) * ON + SL(n, 4 * d, 2, JV);
    if (n == "HD2") return Fi(128) * wsum(n, s.V) * ON + SL(n, 4 * d + 2 * JV, 1, JV);
    if (n == "HD0") { const E wx = wsum(n, s.V); return iw * (Pw * P.ClH + wx * (HN(11, 0) + BIGe * HN(11, 1)) - place - Fi(1ll << sh) * (Fi(32896) * wx * ON + SL(n, 4 * d, 2, JV)) - (Fi(128) * wx * ON + SL(n, 4 * d + 2 * JV, 1, JV))); }
    if (n == "HE0") return Fi(32896) * wsum(n, s.V) * ON + SL(n, 4 * d, 2, JV) + Fi(65536) * (Fi(offE) * wsum(n, 1 << 30) + SL(n, 4 * d + 3 * JV, 3, JV));
    if (n == "HE1") return Fi(32896) * wsum(n, s.V - JVa) * ON + SL(n, 4 * d + JVa, 2, JV) + Fi(65536) * (Fi(offE) * wsum(n, 1 << 30) + SL(n, 4 * d + 3 * JV + JVa, 3, JV));
    if (n == "ROW") { const std::vector<E> e = P.ew(n); E a = E::zero(); for (int j = 0; j < s.NRH; j++) a = a + Fi(row_mul(j)) * e[j] * (part(7, j) + Fi(128)); return a; }
    if (n == "CNF") { const std::vector<E> e = P.ew(n); E a = E::zero(); for (int k = 0; k < d; k++) a = a + e[k] * A2(5, k); return SL(n, 0, 2, d) + Fi(1ll << 28) * a + wsum(n, 1 << 30) * (Fi(1ll << 15) + Fi(1ll << 39)); }
    throw std::runtime_error("no form");
  };
  for (size_t i = 0; i < W.calls.size(); i++) {
    const LookupCall &L = W.calls[i];
    const unsigned h = W.sizes.rK - W.sizes.iota[i];
    std::vector<E> zk(P.lk.zy.begin() + h, P.lk.zy.end());
    zk.insert(zk.end(), P.lk.zx.begin(), P.lk.zx.end());
    const std::vector<E> e = eq_all(zk);
    E direct = E::zero();
    for (u64 x = 0; x < (1ull << L.n); x++) direct = direct + e[x] * E::from(from_u64(L.keys[x]));
    const std::vector<E> ej = eq_all(P.zj((int)i));
    std::vector<char> taken(ej.size(), 0);
    for (auto &[n, pl] : W.place[i]) for (u64 x = 0; x < pl.n; x++) taken[pl.pos(x)] = 1;
    E padW = E::zero();
    for (size_t j = 0; j < ej.size(); j++) if (!taken[j]) padW = padW + ej[j];
    E form = E::from(from_u64(L.row(L.t0))) * padW;
    if (W.place[i].count("NF")) form = form + P.ClN;
    for (auto &[n, pl] : W.place[i]) form = form + KS(n);
    check(("key form of call " + pol.callH[i].name).c_str(), same(direct, form));
  }
  // The claims that tie end values to U3.
  auto Dg = [&](int n, int j) { return part(n, j) + Fi(128); };
  auto dig = [&](int n, int j, int l) { E a = E::zero(), sc = E::one(); for (int i = 0; i < l; i++, sc = sc * Fi(256)) a = a + sc * Dg(n, j + i); return a; };
  check("EV_0 = a(9)", same(P.s6.values[0], Fi(2) * Dg(9, 58) + Dg(9, 59) + Fi(256) * Dg(9, 60) + Fi(65536) * Dg(9, 61)));
  check("EV_1 = b(9)", same(P.s6.values[1], Dg(9, 62) + Fi(256) * Dg(9, 63)));
  check("EV_2 = Rlo(9)", same(P.s6.values[2], dig(9, 80, 3)));
  check("EV_3 = Rhi(9)", same(P.s6.values[3], dig(9, 83, 3)));
  check("EN_0 = R(10)", same(P.s7.values[0], dig(10, 80, 3) + Fi(1ll << 24) * dig(10, 83, 3)));
  {
    std::vector<E> z(P.rW.begin(), P.rW.begin() + (s.lPd - s.ls));
    const std::vector<E> e = eq_all(z);
    E a = E::zero(), g = E::one();
    for (int lb = 0; lb < s.XL; lb++, g = g * P.gam) for (int i = 0; i < d / S2; i++) a = a + e[i] * g * part(3, lb * d / S2 + i);
    check("vL_4 reads part 3", same(P.s4.values[0], a));
    const int cc = W.call_of("NF");
    std::vector<E> zj = P.zj(cc);
    zj.resize(zj.size() - s.ls);
    const std::vector<E> eh = eq_all(zj);
    const SegPlace &pl = W.place[cc].at("NF");
    E b = E::zero(), sc = E::one();
    for (int lb = 0; lb < s.XL; lb++, sc = sc * Fi(256)) for (int i = 0; i < d / S2; i++) b = b + sc * eh[pl.pos(S2 * i) / S2] * part(4, lb * d / S2 + i);
    check("EN_1 reads part 4", same(P.s7.values[1], b));
    std::vector<E> rsP(P.rs.rbegin(), P.rs.rend());
    const std::vector<E> er = eq_all(rsP);
    E c0 = E::zero();
    for (int k = 0; k < d; k++) c0 = c0 + er[k] * A2(6, k);
    check("EH_0 reads part 6", same(P.s8.values[0], c0));
  }
  check("(B2) as a form", same(HN(12, 0) + Fi(s.C) * HN(12, 1) + HN(12, 2), Fi(s.theta) * RCe));
  check("(B3) as a form", same(HN(12, 1) + HN(12, 4), Fi(RCe == E::zero() ? 0 : s.cap)));
  check("(B4) as a form", same(HN(11, 0) + HN(11, 3), Fi(s.C) * ERC));
  {
    E a = HN(11, 5) + Fi(s.offE()) * Fi(JV) * ERC - Fi(JV - s.V + 1) * Fi(1ll << s.F) * ERC, sc = E::one();
    for (int lb = 0; lb < 3; lb++, sc = sc * Fi(256)) for (int i = 0; i < JV / S2; i++) a = a + sc * part(2, (4ll * d + 3 * JV + (long long)lb * JV) / S2 + i);
    check("(Z) as a form", a == E::zero());
  }
  // Cl_H against the logits themselves.
  {
    const std::vector<E> e0 = P.ew("HD0");
    E a = E::zero();
    for (unsigned t = 0; t < T; t++) if (W.on[t]) for (int x = 0; x < s.V; x++) a = a + eZT[t] * e0[x] * Fi(W.lam[(size_t)t * s.V + W.win[t]] - W.lam[(size_t)t * s.V + x]);
    check("Cl_H is the logits' sum", same(Pw * P.ClH, a));
  }
  // The merge.
  {
    std::vector<E> Ec, Bc;
    P.weight_factors(Ec, Bc);
    E a = E::zero();
    for (size_t i = 0; i < Ec.size(); i++) a = a + Ec[i] * Bc[i];
    check("EH_1 is the weight claim", same(P.s8.values[1], a));
    const E cm = ch.draw();
    const u32 j0 = slot;
    P.merge_run(ch, mask.data(), slot, cm);
    check("the merge sends 2 lkW + 3 scalars", slot - j0 == 2u * s.lkW + 3);
    E Em = Ec[0], Bm = Bc[0];
    { std::vector<E> e2 = Ec, b2 = Bc; for (const E &r : P.merge.challenges) { const size_t h = e2.size() / 2; for (size_t i = 0; i < h; i++) e2[i] = e2[i] + r * (e2[i + h] - e2[i]), b2[i] = b2[i] + r * (b2[i + h] - b2[i]); e2.resize(h), b2.resize(h); } Em = e2[0], Bm = b2[0]; }
    check("fv' = E(rm) B(rm) + beta cm", same(P.merge.end_values[0], Em * Bm + P.merge.beta * cm));
  }
  printf("head_keys_test: %s\n", bad ? "FAILED" : "all keys in their tables, all checks hold");
  return bad != 0;
}
// The frame's channel: these must compile on it.
template void fpmlx::HeadProof::moment0<fpmlx::Chan>(fpmlx::Chan &, const fpmlx::E *, fpmlx::u32 &);
template void fpmlx::HeadProof::moment2<fpmlx::Chan>(fpmlx::Chan &, const fpmlx::E *, fpmlx::u32 &, const fpmlx::E &, fpmlx::u64);
