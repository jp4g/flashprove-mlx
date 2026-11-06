// Every key of an attention segment's lookup is a row of its call's table, on the reference's model.
// Then the segment's rounds on a mock channel with random challenges, and the claims of 11.3 that tie
// the hidden values and end values to the entries of U3, and the keys to the hidden values.
// usage: attn_keys_test <directory of make_reference.py> <policy.txt> [full]
// full: runs the lookup of row 3 as well; otherwise its points are drawn at random.
#include <cstdio>
#include <random>
#include "seg_attn_proof.h"
using namespace fpmlx;
static std::mt19937_64 rng(11);
static E rnd() { E x; for (int k = 0; k < 3; k++) x.c[k] = from_u64(rng()); return x; }
static std::vector<E> rnds(size_t n) { std::vector<E> v(n); for (E &x : v) x = rnd(); return v; }
struct Mock {
  std::vector<E> sent; // by slot, when the lookup has not run
  void send(const E &x) { sent.push_back(x); }
  void send_hiding_sum(const E &x) { sent.push_back(x); }
  E public_part(const E &x) { return x; }
  E epsilon() { return rnd(); }
  E round() { return rnd(); }
  E y() { return rnd(); }
  void values() {}
  E shift() { return rnd(); }
  E alpha() { return rnd(); }
  E line() { return rnd(); }
  std::vector<E> point(const char *, size_t n) { return rnds(n); }
  std::vector<E> point_join(size_t n) { return rnds(n); }
  std::vector<E> point_zero(size_t n) { return rnds(n); }
  E batch(u64) { return rnd(); }
};
static int fails = 0;
static void same(const char *what, const E &a, const E &b) { if (!(a == b)) { printf("  CLAIM FAILS: %s\n", what); fails++; } }
static E eqp(const std::vector<E> &x, const std::vector<E> &y) { E p = E::one(); for (size_t v = 0; v < x.size(); v++) p = p * (x[v] * y[v] + (E::one() - x[v]) * (E::one() - y[v])); return p; }
static E ones(const std::vector<E> &z) { E p = E::one(); for (const E &x : z) p = p * x; return p; }
static E LEof(const std::vector<E> &x, const std::vector<E> &y) { E le = E::one(); for (int v = (int)x.size() - 1; v >= 0; v--) le = (E::one() - x[v]) * y[v] + (x[v] * y[v] + (E::one() - x[v]) * (E::one() - y[v])) * le; return le; }
static E idOf(const std::vector<E> &z) { E s = E::zero(); for (size_t v = 0; v < z.size(); v++) s = s + attn_e(1ll << (z.size() - 1 - v)) * z[v]; return s; }
static E mle(const LookupCall &c, const std::vector<E> &z) { const std::vector<E> e = eq_all(z); E s = E::zero(); for (size_t i = 0; i < e.size(); i++) if (c.keys[i]) s = s + e[i] * E::from(from_u64(c.keys[i])); return s; }
// The verifier's replay of a sumcheck: every round, then the claim Left(C; S0; 0) = 0.
static void replay(const char *what, const Mock &ch, const E *mask, const PscResult &r, const std::vector<unsigned> &d, unsigned dY, const E &pub, const E &C, const E &S0) {
  std::vector<Weight<E>> all;
  for (unsigned x : d) all.push_back(Weight<E>::plain(x));
  all.push_back(Weight<E>::at_zero(dY + 1));
  std::vector<E> gam = r.challenges;
  gam.push_back(r.beta);
  E cur = ch.sent[r.first] + r.eps * pub;
  size_t at = r.first + 1;
  for (size_t l = 0; l < all.size(); l++) {
    std::vector<E> q(all[l].d);
    for (unsigned t = 0; t < all[l].d; t++) if (t != all[l].fixed) q[t] = ch.sent[at++];
    all[l].complete(q, cur);
    cur = poly_at(q, gam[l]);
  }
  const std::vector<E> form = hp_form(all, gam);
  E left = E::zero(), PP = E::one();
  for (size_t i = 0; i < form.size(); i++) left = left + form[i] * mask[r.first + i];
  for (size_t t = 0; t < all.size(); t++) PP = PP * poly_at(all[t].psi, gam[t]);
  same(what, left + r.eps * PP * (pub - C) + r.eps * S0 - cur, E::zero());
}
static void proof_checks(AttnGroup &A, FFGroup &F, bool full) {
  const AShape &s = A.s;
  const int lt = s.lt, lgk = s.lgk, ldh = s.ldh, lB = s.lB, lnT = s.lnT;
  const u64 H = s.H, HA = s.HA, G = s.G, dh = s.dh, Lam = s.Lam, NP = s.NP;
  Mock ch;
  std::vector<E> mask = rnds(1 << 20);
  u32 slot = 0;
  Smallness sm;
  AttnRun R;
  if (full) attn_rounds(ch, mask.data(), slot, A, F, sm, R);
  else {
    R.g1.zx = rnds(A.sizes.dT), R.g1.zy = rnds(A.sizes.rK);
    for (auto &z : R.g1.zb) z = rnds(5 + A.sizes.dL);
    attn_own(ch, mask.data(), slot, A, F, sm, R);
  }
  if (slot >= mask.size()) { printf("  mask row too short\n"); fails++; return; }
  const std::vector<E> U = sm.u3();
  printf("  %u slots, %llu sent, U3 of %zu entries, nW %llu, %zu parts\n", slot, (unsigned long long)ch.sent.size(), U.size(), (unsigned long long)R.nW, sm.parts.size());
  u64 nKF = R.kf.size();
  const u64 fillH = (u64)A.ah.k * s.wcH, fillO = (u64)A.ao.k * s.wcO, fillQ = (u64)F.qa.k * s.wcQ;
  const u64 want = 3 * s.kAs + (nKF + 6) * fillH + (fillO * s.T >> s.la) + (2 * nKF + 5) * (fillQ * s.T >> s.la) + 3 * A.bA.k;
  if (want != U.size() || U.size() != R.nW) { printf("  U3's length: want %llu\n", (unsigned long long)want); fails++; }
  auto PL = [&](u64 t, u64 x, u64 lam, u64 a, const std::vector<E> &Z, bool isD) {
    const int nb = isD ? s.nD : s.nM;
    const u64 a0 = (isD ? (5 * NP) << s.nM : 0) + (((x * Lam + lam) * HA + a) << nb);
    const std::vector<E> eh = eq_all(attn_slice(Z, 0, nb - s.lmS));
    E r = E::zero();
    for (u64 i = 0; i < eh.size(); i++) r = r + eh[i] * U[t + (a0 >> s.lmS) + i];
    return r;
  };
  auto LV = [&](u64 t, u64 off, int nl, u64 J, u64 lam, u64 j) { E r = E::zero(); for (int l = 0; l < nl; l++) r = r + attn_e(1ll << (8 * l)) * U[t + off + ((u64)l * Lam + lam) * J + j]; return r; };
  auto XL = [&](u64 t, const Tensor2 &X, u64 lam, u64 hd, const std::vector<E> &zc) {
    const std::vector<E> ec = eq_all(attn_slice(zc, 0, ldh - s.ls));
    E r = E::zero();
    for (u64 c = 0; c < ec.size(); c++)
      for (int l = 0; l < X.nl; l++) r = r + ec[c] * attn_e(1ll << (8 * l)) * U[t + (u64)X.feat(l, (int)lam, (int)(hd * dh)) / s.S2 + c];
    return r;
  };
  auto overG = [&](const std::vector<E> &z, auto f) { const std::vector<E> e = eq_all(z); E r = E::zero(); for (u64 g = 0; g < G; g++) r = r + e[g] * f(g); return r; };
  auto LW = [](const std::vector<AttnPW> &p) { E r = E::zero(); for (auto &w : p) r = r + w.wp; return r; };
  const Tensor2 tQ = F.tf.at("q"), tK = F.tf.at("kx"), tV = F.tf.at("v"), tO = F.tf.at("o");
  const Tensor2 tRh{0, 1, (int)Lam, (int)s.dq}, tRhp{(int)s.oRhp, 1, (int)Lam, (int)s.dq};
  auto zk_of = [&](size_t q) { const unsigned hi = A.sizes.rK - A.sizes.iota[q]; std::vector<E> z(R.g1.zy.begin() + hi, R.g1.zy.end()); z.insert(z.end(), R.g1.zx.begin(), R.g1.zx.end()); return z; };
  const E c128 = attn_e(128);
  std::vector<E> form(A.ac.size(), E::zero());
  for (auto &p : R.kr) {
    for (u64 x = 0; x < 3; x++) { E v = E::zero(); for (auto &w : p.pw) v = v + w.wp * PL(R.tA, x, w.lam, w.a, p.Z, p.seg == AS_RD); same("KR plane against tA", p.hidden[x], v); }
    const E L = LW(p.pw);
    form[p.call] = form[p.call] + p.hidden[0] + c128 * L + attn_e(1ll << 20) * (p.hidden[1] + c128 * L) + attn_e(1ll << 40) * p.hidden[2];
  }
  for (auto &k : R.kf) {
    const bool isD = k.seg == AS_FD;
    for (u64 x = 0; x < 5; x++) { E v = E::zero(); for (auto &w : k.pw) v = v + w.wp * PL(R.tA, x, w.lam, w.a, k.Z, isD); same("KFD plane against tA", k.hidden[x], v); }
    E mv = E::zero();
    for (auto &w : k.pw) mv = mv + w.wp * overG(k.zg, [&](u64 g) { return LV(k.tM, 0, 4, H, w.lam, g * HA + w.a); });
    same("MV against tM", k.hidden[5], mv);
    const std::vector<E> &r = k.sc.challenges, rg = attn_slice(r, 0, lgk), rc = attn_slice(r, r.size() - ldh, ldh);
    for (size_t b = 0; b < k.pw.size(); b++) {
      auto &w = k.pw[b];
      same("K: AE against tQ", k.sc.values[2 * b], overG(rg, [&](u64 g) { return XL(k.tQ, tQ, w.lam, g * HA + w.a, rc); }));
      same("K: BE against tK", k.sc.values[2 * b + 1], overG(rg, [&](u64 g) { return XL(k.tK, tK, w.lam, g, rc); }));
    }
    // The weight's closed form, which the verifier computes.
    E f = eqp(k.zg, rg);
    if (isD) f = f * eqp(k.zI, attn_slice(r, lgk, lnT));
    else { const auto rP = attn_slice(r, lgk, lnT - 1), rs = attn_slice(r, lgk + lnT - 1, lnT); f = f * eqp(k.zP, rP) * (eqp(k.zs, rs) - ones(k.zs) * ones(rs)); }
    same("K: W at r", k.sc.values[2 * k.pw.size()], f);
    const AShape::ShD d = s.shd(A.ac[k.call].sh);
    const E L = LW(k.pw), DP = k.hidden[5] - k.hidden[6] + (isD ? L * attn_e(d.Om) * (E::one() - LEof(k.zj, k.zi)) : L * attn_e(d.C) * ones(k.zs));
    form[k.call] = form[k.call] + DP - (k.hidden[0] + c128 * L) - attn_e(256) * (k.hidden[1] + c128 * L) + attn_e(1ll << 32) * k.hidden[2] + attn_e(1ll << 33) * (k.hidden[3] + c128 * L + attn_e(256) * k.hidden[4]);
  }
  for (size_t q = 0; q < A.ac.size(); q++)
    if (A.ac[q].kind == "KFD") {
      const std::vector<E> zk = zk_of(q);
      E wall = E::zero();
      for (auto &pc : A.ac[q].pcs) wall = wall + eq_at(attn_slice(zk, 0, zk.size() - pc.b), pc.off >> pc.b);
      form[q] = form[q] + (E::one() - wall) * E::from(from_u64(A.calls[q].row(A.calls[q].t0)));
    }
  const std::vector<E> zgW = attn_slice(R.ZW, 1, lgk), ziW = attn_slice(R.ZW, 1 + lgk, lt), zcW = attn_slice(R.ZW, 1 + lgk + lt, ldh);
  const std::vector<E> zgX = attn_slice(R.ZX, 0, lgk), ziX = attn_slice(R.ZX, lgk, lt), zgH = attn_slice(R.ZH, 0, lgk), zcH = attn_slice(R.ZH, lgk + lt, ldh);
  { E zc = E::zero(), rh = E::zero(), rp = E::zero(), r2 = E::zero(), r2p = E::zero();
    for (auto &w : R.pwW) {
      zc = zc + w.wp * overG(zgW, [&](u64 g) { return LV(R.hW, s.oZr, 4, H, w.lam, g * HA + w.a); });
      rh = rh + w.wp * overG(zgW, [&](u64 g) { return XL(R.tO, tRh, w.lam, g * HA + w.a, zcW); });
      rp = rp + w.wp * overG(zgW, [&](u64 g) { return XL(R.tO, tRhp, w.lam, g * HA + w.a, zcW); });
    }
    for (auto &w : R.pwH) {
      r2 = r2 + w.wp * overG(zgH, [&](u64 g) { return XL(R.tO, tRh, w.lam, g * HA + w.a, zcH); });
      r2p = r2p + w.wp * overG(zgH, [&](u64 g) { return XL(R.tO, tRhp, w.lam, g * HA + w.a, zcH); });
    }
    same("ZC against hW", R.ZC, zc), same("RH against tO", R.RH, rh), same("RHP against tO", R.RHP, rp), same("RH2 against tO", R.RH2, r2), same("RHP2 against tO", R.RHP2, r2p);
    for (int k = 0; k < lt; k++) { E v = E::zero(); for (auto &w : R.pwX) v = v + w.wp * overG(zgX, [&](u64 g) { return LV(R.hX, s.oIb, 1, (u64)lt * H, w.lam, (u64)k * H + g * HA + w.a); }); same("IB against hX", R.IB[k], v); }
  }
  // The keys of WL, IX, WH.
  for (size_t q = 0; q < A.ac.size(); q++) {
    if (A.ac[q].holds(AS_WL)) {
      const E L = LW(R.pwW), zsl = R.ZW[0], rem = attn_e(2) * R.ACC + R.ZC - attn_e(2) * R.ZO;
      form[q] = form[q] + (E::one() - zsl) * (rem - attn_e(65536) * (R.RH + c128 * L)) + zsl * (attn_e(2) * R.ZC - L - rem - attn_e(65536) * (R.RHP + c128 * L));
    }
    if (A.ac[q].holds(AS_IX)) {
      E ibv = E::zero();
      for (int k = 0; k < lt; k++) ibv = ibv + attn_e(1ll << k) * R.IB[k];
      form[q] = form[q] + attn_e(1ll << (16 - lt)) * (LW(R.pwX) * idOf(ziX) - ibv);
    }
    if (A.ac[q].holds(AS_WH)) { const E L = LW(R.pwH); form[q] = form[q] + R.RH2 + c128 * L + attn_e(1ll << 20) * (R.RHP2 + c128 * L); }
    same(("the keys of " + A.ac[q].name + " at zk").c_str(), mle(A.calls[q], zk_of(q)), form[q]);
  }
  // SE, VM, VD.
  { const std::vector<E> &r = R.scE.challenges, rg = attn_slice(r, 0, lgk);
    for (u64 b = 0; b < NP; b++) {
      const u64 lam = b / HA, a = b % HA;
      same("SE: A against hE", R.scE.values[2 * b], overG(rg, [&](u64 g) { return LV(R.hE, s.oZr, 4, H, lam, g * HA + a); }));
      same("SE: B against qO", R.scE.values[2 * b + 1], overG(rg, [&](u64 g) { return XL(R.qO, tO, lam, g * HA + a, zcW); }));
    }
  }
  for (int reg = 0; reg < 2; reg++) {
    const PscResult &P = reg ? R.vpd : R.vpm;
    const std::vector<E> rg = attn_slice(P.challenges, 0, lgk), &pt = reg ? R.ptD : R.ptM;
    const u64 tV2 = reg ? R.tVD : R.tVM, qV = reg ? R.qVD : R.qVM;
    for (u64 b = 0; b < NP; b++) {
      const u64 lam = b / HA, a = b % HA;
      same("V: A against the planes", P.values[2 * b], PL(tV2, 3, lam, a, pt, reg) + c128 + attn_e(256) * PL(tV2, 4, lam, a, pt, reg));
      same("V: B against qV", P.values[2 * b + 1], R.gam + overG(rg, [&](u64 g) { return XL(qV, tV, lam, g, zcW); }));
    }
  }
  // M6, SF, SD, HB.
  const std::vector<E> &rF = R.scF.challenges, ri = attn_slice(rF, 0, lt), rg = attn_slice(rF, lt, lgk), rc = attn_slice(rF, lt + lgk, ldh);
  const std::vector<E> &rd = R.scd.challenges, pj = attn_slice(rd, 0, lt), pg = attn_slice(rd, lt, lgk), pi = attn_slice(rd, lt + lgk, lt);
  for (u64 lam = 0; lam < Lam; lam++) same("KC against qKx", R.scd.values[lam], overG(pg, [&](u64 g) { return XL(R.qKx, tK, lam, g, rc); }));
  const std::vector<E> &rH = R.scH.challenges, rg2 = attn_slice(rH, lt, lgk), rk = attn_slice(rH, lt + lgk, 4), ek = eq_all(rk);
  for (u64 b = 0; b < NP; b++) {
    const u64 lam = b / HA, a = b % HA;
    same("M6 against hT", R.M6[b], overG(R.tg, [&](u64 g) { return LV(R.hT, 0, 4, H, lam, g * HA + a); }));
    same("SF: A against qF", R.scF.values[2 * b], overG(rg, [&](u64 g) { return XL(R.qF, tQ, lam, g * HA + a, rc); }));
    E hb = E::zero();
    for (int k = 0; k < lt; k++) {
      same("IBP against hPi", R.scd.values[Lam + b * lt + k], overG(pg, [&](u64 g) { return LV(R.hPi, s.oIb, 1, (u64)lt * H, lam, (u64)k * H + g * HA + a); }));
      hb = hb + ek[k] * overG(rg2, [&](u64 g) { return LV(R.hR2, s.oIb, 1, (u64)lt * H, lam, (u64)k * H + g * HA + a); });
    }
    same("HB: A against hR2", R.scH.values[2 * b], hb);
    same("HB: B", R.scH.values[2 * b + 1], hb - E::one());
  }
  // The summand of SD at the end, on true values, against the bound maps' product: pj's order.
  { E t = E::zero();
    for (u64 b = 0; b < NP; b++) {
      E p = R.om4[b] * R.scd.values[b / HA];
      for (int k = 0; k < lt; k++) { const E y = pj[lt - 1 - k]; same("y_k", R.scd.values[Lam + NP * lt + 1 + k], y); p = p * (R.scd.values[Lam + b * lt + k] * (y + y - E::one()) + (E::one() - y)); }
      t = t + p;
    }
    same("SD: Eq at the end", R.scd.values[Lam + NP * lt], eqp(ri, pi) * eqp(rg, pg));
    // The same summand on the factors that the rounds bound.
    E u = E::zero();
    for (u64 b = 0; b < NP; b++) {
      E p = R.om4[b] * R.scd.values[b / HA];
      for (int k = 0; k < lt; k++) p = p * R.scd.values[Lam + NP * lt + 1 + lt + b * lt + k];
      u = u + p;
    }
    same("SD: at_end against the rounds' summand", t, u);
  }
  if (!full) {
    // SF, SD and HB as the verifier reads them.
    E pub = E::zero(), C = E::zero(), S0 = E::zero();
    const E fF = eqp(R.tg, rg) * eqp(R.ti, ri);
    for (u64 b = 0; b < NP; b++) {
      pub = pub + R.om[b] * ch.sent[R.scF.first - NP + b], C = C + R.om[b] * R.M6[b];
      S0 = S0 + R.om[b] * fF * R.scF.end_values[2 * b] * R.scF.end_values[2 * b + 1];
    }
    replay("SF replayed", ch, mask.data(), R.scF, std::vector<unsigned>(lt + lgk + ldh, 4), 2, pub, C, S0);
    pub = C = S0 = E::zero();
    for (u64 b = 0; b < NP; b++) {
      pub = pub + R.om4[b] * R.scF.end_values[2 * b + 1], C = C + R.om4[b] * R.scF.values[2 * b + 1];
      E t = R.om4[b] * R.scd.end_values[b / HA];
      for (int k = 0; k < lt; k++) { const E y = pj[lt - 1 - k]; t = t * (R.scd.end_values[Lam + b * lt + k] * (y + y - E::one()) + (E::one() - y)); }
      S0 = S0 + t;
    }
    std::vector<unsigned> deg(lt, 3);
    deg.insert(deg.end(), lgk, lt + 3), deg.insert(deg.end(), lt, lt + 2);
    replay("SD replayed", ch, mask.data(), R.scd, deg, lt + 1, pub, C, eqp(ri, pi) * eqp(rg, pg) * S0);
    S0 = E::zero();
    const E fH = eqp(R.tbi, attn_slice(rH, 0, lt)) * eqp(R.tbg, rg2) * eqp(R.tk, rk);
    for (u64 b = 0; b < NP; b++) S0 = S0 + R.om5[b] * fH * R.scH.end_values[2 * b] * R.scH.end_values[2 * b + 1];
    replay("HB replayed", ch, mask.data(), R.scH, std::vector<unsigned>(lt + lgk + 4, 4), 2, E::zero(), E::zero(), S0);
  }
  // The planes of bA.
  { std::vector<E> sumP; E tot = E::zero();
    for (u64 c = 0; c < A.bA.k; c++) tot = tot + U[R.tP + c];
    u64 live = 0; for (auto &L : A.calls) for (char p : L.padding) live += !p;
    same("tP: the multiplicities' sum", tot, attn_e((I64)live));
  }
}
static std::string slurp(const char *p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
int main(int c, char **v) {
  const std::string dir = v[1];
  const ModelShape m = read_shape(dir + "/params.txt");
  auto w = mx::load_safetensors(dir + "/weights.safetensors").first;
  for (auto &[k, a] : w) if (k.rfind("table", 0) == 0) a = mx::astype(a, mx::int64);
  const Policy pol = Policy::read(slurp(v[2]));
  const Values V = capture_forward(m, w, w.at("tokens"));
  const CallTables tabs = call_tables(w);
  const int groups = m.L >> pol.get("ll");
  int bad = 0;
  for (int g = 0; g < groups; g++) {
    FFGroup F = ff_build(pol, m, V, tabs, g);
    AttnGroup A = attn_build(pol, m, F, g);
    { // The GPU's witness against the host's, word by word.
      const AttnGroup B = attn_build(pol, m, F, g, &V);
      size_t diff = 0, n = 0;
      auto cmp = [&](const std::vector<int32_t> &x, const std::vector<int32_t> &y) { n += x.size(); for (size_t i = 0; i < x.size(); i++) diff += x[i] != y[i]; };
      for (size_t i = 0; i < A.as.size(); i++) cmp(A.as[i].cell, B.as[i].cell);
      cmp(A.ah.part[0].cell, B.ah.part[0].cell), cmp(A.ao.part[0].cell, B.ao.part[0].cell), cmp(A.bA.cell, B.bA.cell);
      printf("group %d: %zu of %zu cells differ\n", g, diff, n);
      size_t dk = 0, nk = 0;
      for (size_t c = 0; c < A.calls.size(); c++) {
        nk += A.calls[c].keys.size();
        for (size_t i = 0; i < A.calls[c].keys.size(); i++) dk += A.calls[c].keys[i] != B.calls[c].keys[i] || A.calls[c].padding[i] != B.calls[c].padding[i];
      }
      printf("group %d: %zu of %zu keys differ\n", g, dk, nk);
      const bool same = A.Q == B.Q && A.K == B.K && A.Vv == B.Vv && A.O == B.O && A.Mx == B.Mx && A.Zd == B.Zd && A.ib == B.ib && A.sc == B.sc && A.acc == B.acc && A.rem == B.rem && A.mult == B.mult;
      printf("group %d: the values are %s\n", g, same ? "equal" : "NOT EQUAL");
      if (diff || dk || !same) return 1;
    }
    for (size_t i = 0; i < A.calls.size(); i++) {
      std::vector<LookupCall> one{A.calls[i]};
      LookupSizes sz = lookup_sizes(one, (unsigned)A.s.lidb);
      std::vector<std::vector<u32>> mult;
      const bool ok = lookup_multiplicities(one, sz, mult);
      u64 live = 0; for (char p : A.calls[i].padding) live += !p;
      printf("group %d call %s: 2^%u keys, %llu live (tot %llu): %s\n", g, A.ac[i].name.c_str(), A.calls[i].n, (unsigned long long)live, (unsigned long long)A.ac[i].tot, ok ? "every key is in the table" : "A KEY IS IN NO ROW");
      bad += !ok || live != A.ac[i].tot;
    }
    bad += !A.keys_ok;
    // o as the forward pass has it, against the same division.
    if (A.sizes.dT < std::max<unsigned>(A.s.lmS, A.s.lt + A.s.ldh)) { printf("group %d: dT %u too small\n", g, A.sizes.dT); bad++; }
    printf("group %d: as %llu x %llu, ah %llu x %d, ao %llu x %d, bA %llu x %llu, dT %u rK %u kT %u\n", g, 1ull << A.s.lmS, (unsigned long long)A.s.kAs, (unsigned long long)A.ah.part[0].m, A.ah.k, (unsigned long long)A.ao.part[0].m, A.ao.k, (unsigned long long)A.bA.m, (unsigned long long)A.bA.k, A.sizes.dT, A.sizes.rK, A.sizes.kT);
    if (A.keys_ok) proof_checks(A, F, c > 3 && std::string(v[3]) == "full");
    printf("group %d: rounds on the mock channel done, %d claims failed so far\n", g, fails);
  }
  bad += fails;
  printf("attn_keys_test: %s\n", bad ? "FAILED" : "all keys in their tables, all claims hold");
  return bad != 0;
}
