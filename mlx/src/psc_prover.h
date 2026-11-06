// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The prover's side of the sumcheck with a hiding sum (docs/format/11-1-machinery.md). The verifier's
// side is verifier/src/vf_sumcheck_psc.h. Reference code first: tables on the host; the heavy loops
// move to Metal by profile.
//
// A sumcheck proves that phi_0, phi_1, .. applied to the summand H give the value of a form C. H is a
// polynomial in the values of multilinear maps at the point:
//   H = sum over monomials of coeff * product of map values.
// The first `shown` maps have end values, sent under masks; the others are kept (K).
#pragma once
#include <cstdlib>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include "field/field.h"
#include "iop/hp.h"
#include "mlx_kernels.h"
namespace fpmlx {
using namespace blk;
struct Monomial {
  E coeff;
  std::vector<unsigned> maps; // indices into PscJob::maps; empty: a constant
};
// One variable's round.
struct PscVar {
  Weight<E> w;
  unsigned bit = 0; // the bit of the maps' index that this round binds
};
struct PscJob {
  std::vector<PscVar> vars;           // in the order of the rounds, without Y
  std::vector<std::vector<E>> maps;   // 2^(vars.size()) values each, or one value if there are no vars
  std::vector<Monomial> summand;
  unsigned shown = 0;                 // the first `shown` maps end in values that are sent
  unsigned degree_y = 0;              // d_Y; 0 exactly if shown is 0
  E start_value = E::zero();          // the true value of C
  E start_public = E::zero();         // pub(C)
  // A summand whose end values are no maps of the rounds (step 11, the merge). ends: from the challenges
  // and the bound maps, the true end values, `shown` of them. at_end: the summand at given end values.
  std::function<std::vector<E>(const std::vector<E> &challenges, const std::vector<E> &bound)> ends;
  std::function<E(const std::vector<E> &values)> at_end;
  // Masks of the end values, if they are not the mask row's entries (the merge's index-mask entry).
  std::vector<E> end_masks;
  // In place of `maps`: maps already on the GPU, device_maps of equal length, one after another.
  std::shared_ptr<mx::array> device;
  unsigned device_maps = 0;
};
struct PscResult {
  std::vector<E> challenges; // gamma_l in the order of the rounds, without Y
  E beta = E::zero(), eps = E::zero();
  std::vector<E> end_values; // as sent: val_i + beta mask(jE + i)
  std::vector<E> values;     // val_i at the point
  u32 first = 0, first_end = 0;
};
template <class Body> inline void psc_parallel(size_t n, Body body) {
  const size_t T = n < 4096 ? 1 : std::max(1u, std::thread::hardware_concurrency());
  if (T == 1) { body(0, n); return; }
  std::vector<std::thread> t;
  for (size_t i = 0; i < T; i++) t.emplace_back([=] { body(n * i / T, n * (i + 1) / T); });
  for (auto &x : t) x.join();
}
inline E power_of(const E &x, unsigned n) { E r = E::one(); for (unsigned i = 0; i < n; i++) r = r * x; return r; }
// The polynomial of degree < n through (0, v0), (1, v1), .., (n - 1, v_(n-1)).
inline std::vector<E> interpolate(const std::vector<E> &v) {
  const size_t n = v.size();
  std::vector<E> out(n, E::zero());
  for (size_t i = 0; i < n; i++) {
    std::vector<E> basis(1, E::one());
    E den = E::one();
    for (size_t j = 0; j < n; j++) {
      if (j == i) continue;
      const E xj = E::from(from_u64(j));
      std::vector<E> next(basis.size() + 1, E::zero());
      for (size_t t = 0; t < basis.size(); t++) next[t + 1] = next[t + 1] + basis[t], next[t] = next[t] - xj * basis[t];
      basis = std::move(next);
      den = den * (E::from(from_u64(i)) - xj);
    }
    const E s = v[i] * inv(den);
    for (size_t t = 0; t < basis.size(); t++) out[t] = out[t] + s * basis[t];
  }
  return out;
}
// From this many values of a map, the rounds run on the GPU. Measured, three proofs each, segments' sum:
// 7.7 s at 512, 7.6 s at 2048, 7.7 s at 8192, 8.9 s at 32768, 9.4 s at 131072. FP_PSC_GPU_MIN sets it: 0
// keeps every round on the CPU.
inline size_t &psc_gpu_min() {
  static size_t v = [] { const char *e = getenv("FP_PSC_GPU_MIN"); return e ? (size_t)strtoull(e, nullptr, 10) : size_t(1) << 13; }();
  return v;
}
// Ch: send(E), send_hiding_sum(E), epsilon(), round() -> E, y() -> E (not 0), values().
template <class Ch> PscResult psc_run(Ch &ch, const E *mask, u32 &slot, PscJob job) {
  job.start_public = ch.public_part(job.start_public);
  const size_t n = job.vars.size();
  const bool has_y = job.shown > 0;
  std::vector<Weight<E>> all;
  for (const PscVar &v : job.vars) all.push_back(v.w);
  if (has_y) all.push_back(Weight<E>::at_zero(job.degree_y + 1));
  const size_t NV = all.size();
  PscResult out;
  out.first = slot;
  std::vector<unsigned> sk(NV + 1, 0);
  for (size_t l = 0; l < NV; l++) sk[l + 1] = sk[l] + all[l].d - 1;
  // u(l, deg) = mask(first + pos(l, deg)).
  auto u_at = [&](size_t l, unsigned deg) {
    if (l + 1 < NV && deg == 0) return E::zero();
    return mask[out.first + hp_place(all, (unsigned)l, deg)];
  };
  auto u_poly = [&](size_t l) { std::vector<E> p(all[l].d); for (unsigned t = 0; t < all[l].d; t++) p[t] = u_at(l, t); return p; };
  // c_m = phi_m(u_m).
  std::vector<E> c(NV);
  for (size_t m = 0; m < NV; m++) c[m] = all[m].apply(u_poly(m));
  // A term of U does not hold the variables before its own: phi of 1 is 2 for a plain variable.
  auto later_terms = [&](size_t l) { // the terms m > l, summed over the variables after l
    E s = E::zero(), f = E::one();
    for (size_t m = l + 1; m < NV; m++) {
      s = s + f * c[m];
      f = f * all[m].of_one();
    }
    return s;
  };
  E hs = c[0];
  { E f = all[0].of_one(); for (size_t m = 1; m < NV; m++) { hs = hs + f * c[m]; f = f * all[m].of_one(); } }
  ch.send_hiding_sum(hs);
  out.eps = ch.epsilon();
  slot += 1;
  const E eps = out.eps, D = job.start_public - job.start_value;
  E head = E::zero(), psi_done = E::one(); // head: sum over m < l of u_m(g_m) prod_{m < i < l} psi_i(g_i)
  // Maps' index: the bits that are still free, compacted in order as they are bound.
  std::vector<unsigned> bits;
  for (const PscVar &v : job.vars) bits.push_back(v.bit);
  size_t size = job.device ? job.device->size() / 3 / job.device_maps : job.maps.empty() ? (size_t(1) << n) : job.maps[0].size();
  std::vector<std::vector<E>> &M = job.maps;
  std::unique_ptr<SumRounds> gpu;
  const bool on_card = psc_gpu_min() && size >= psc_gpu_min() && size >= 512;
  if (job.device && !on_card) { // too short to be worth a launch
    const std::vector<E> all_values = e_fetch(*job.device);
    M.resize(job.device_maps);
    for (unsigned k = 0; k < job.device_maps; k++) M[k].assign(all_values.begin() + k * size, all_values.begin() + (k + 1) * size);
    job.device.reset();
  }
  if (on_card && (!M.empty() || job.device) && size >= 512 && size == (size_t(1) << n)) {
    SumShape shape;
    for (const Monomial &m : job.summand) shape.coeff.push_back(m.coeff), shape.maps.push_back(m.maps);
    std::vector<SumWeight> later;
    std::vector<unsigned> at = bits;
    for (size_t l = 0; l + 1 < n; l++) {
      const unsigned b = at[l];
      for (size_t t = l + 1; t < n; t++) if (at[t] > b) at[t]--;
      later.push_back({all[l + 1].moment[0] - all[l + 1].moment[1], all[l + 1].moment[1], at[l + 1]});
    }
    gpu = job.device ? std::make_unique<SumRounds>(*job.device, job.device_maps, std::move(shape), later) : std::make_unique<SumRounds>(M, std::move(shape), later);
    job.device.reset();
  }
  for (size_t l = 0; l < n; l++) {
    if (gpu && (size < psc_gpu_min() || size < 512)) { gpu->fetch(M); gpu.reset(); }
    const unsigned b = bits[l], d = all[l].d, half = (unsigned)0;
    (void)half;
    const size_t step = size_t(1) << b, pairs = size / 2;
    // Weights of the other free bits: the product over the later rounds of (lo, hi).
    std::vector<E> W(gpu ? 0 : pairs, E::one());
    for (size_t t = l + 1; t < n && !gpu; t++) {
      const unsigned bt = bits[t] > b ? bits[t] - 1 : bits[t];
      const E hi = all[t].moment[1], lo = all[t].moment[0] - all[t].moment[1];
      psc_parallel(pairs, [&, bt, hi, lo](size_t a, size_t e) { for (size_t q = a; q < e; q++) W[q] = W[q] * (((q >> bt) & 1) ? hi : lo); });
    }
    // g at the nodes 0 .. d - 1, the summand alone.
    std::vector<E> values(d, E::zero());
    if (gpu) values = gpu->nodes(l, d, step);
    for (unsigned node = 0; node < d && !gpu; node++) {
      const E x = E::from(from_u64(node));
      const unsigned T = pairs < 4096 ? 1 : std::max(1u, std::thread::hardware_concurrency());
      std::vector<E> part(T, E::zero());
      std::vector<std::thread> th;
      auto one = [&](unsigned ti) {
        E s = E::zero();
        std::vector<E> val(M.size());
        for (size_t q = pairs * ti / T; q < pairs * (ti + 1) / T; q++) {
          const size_t low = q & (step - 1), i0 = ((q - low) << 1) | low, i1 = i0 | step;
          for (size_t k = 0; k < M.size(); k++) val[k] = M[k][i0] + x * (M[k][i1] - M[k][i0]);
          E h = E::zero();
          for (const Monomial &m : job.summand) {
            E t = m.coeff;
            for (unsigned k : m.maps) t = t * val[k];
            h = h + t;
          }
          s = s + W[q] * h;
        }
        part[ti] = s;
      };
      if (T == 1) one(0);
      else {
        for (unsigned ti = 0; ti < T; ti++) th.emplace_back(one, ti);
        for (auto &t : th) t.join();
      }
      for (const E &p : part) values[node] = values[node] + p;
    }
    std::vector<E> g = interpolate(values);
    g.resize(d, E::zero());
    for (E &x : g) x = eps * x;
    // eps D prod_{i < l} psi_i(g_i) psi_l(X), and U.
    const std::vector<E> &psi = all[l].psi;
    for (size_t t = 0; t < psi.size(); t++) g[t] = g[t] + (eps * D * psi_done + head) * psi[t];
    const std::vector<E> ul = u_poly(l);
    for (size_t t = 0; t < ul.size(); t++) g[t] = g[t] + ul[t];
    g[0] = g[0] + later_terms(l);
    for (unsigned t = 0; t < d; t++)
      if (t != all[l].fixed) ch.send(g[t]);
    const E r = ch.round();
    slot += d - 1;
    out.challenges.push_back(r);
    // Bind bit b.
    if (gpu) gpu->bind(r, step);
    else for (auto &m : M) {
      std::vector<E> next(pairs);
      psc_parallel(pairs, [&](size_t a, size_t e) {
        for (size_t q = a; q < e; q++) {
          const size_t low = q & (step - 1), i0 = ((q - low) << 1) | low, i1 = i0 | step;
          next[q] = m[i0] + r * (m[i1] - m[i0]);
        }
      });
      m = std::move(next);
    }
    size = pairs;
    for (size_t t = l + 1; t < n; t++)
      if (bits[t] > b) bits[t]--;
    const E pl = poly_at(all[l].psi, r);
    head = head * pl + poly_at(ul, r);
    psi_done = psi_done * pl;
  }
  if (gpu) { gpu->fetch(M); gpu.reset(); }
  for (const auto &m : M) out.values.push_back(m.empty() ? E::zero() : m[0]);
  if (job.ends) out.values = job.ends(out.challenges, out.values);
  if (has_y) {
    const size_t l = NV - 1;
    const unsigned d = all[l].d;
    const u32 jE = out.first + 1 + sk[NV];
    std::vector<E> values(d, E::zero());
    for (unsigned node = 0; node < d; node++) {
      const E y = E::from(from_u64(node));
      std::vector<E> val = out.values;
      for (unsigned i = 0; i < job.shown; i++) val[i] = val[i] + y * (job.end_masks.empty() ? mask[jE + i] : job.end_masks[i]);
      E h = E::zero();
      if (job.at_end) h = job.at_end(val);
      else
        for (const Monomial &m : job.summand) {
          E t = m.coeff;
          for (unsigned k : m.maps) t = t * val[k];
          h = h + t;
        }
      values[node] = h;
    }
    std::vector<E> g = interpolate(values);
    g.resize(d, E::zero());
    for (E &x : g) x = eps * x;
    g[0] = g[0] + eps * D * psi_done + head;
    const std::vector<E> ul = u_poly(l);
    for (size_t t = 0; t < ul.size(); t++) g[t] = g[t] + ul[t];
    for (unsigned t = 0; t < d; t++)
      if (t != all[l].fixed) ch.send(g[t]);
    out.beta = ch.y();
    slot += d - 1;
    out.first_end = slot;
    for (unsigned i = 0; i < job.shown; i++) {
      const E v = out.values[i] + out.beta * (job.end_masks.empty() ? mask[jE + i] : job.end_masks[i]);
      out.end_values.push_back(v);
      ch.send(v);
    }
    ch.values();
    slot += job.shown;
  }
  return out;
}
}
