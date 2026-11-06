// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
#include "mlx_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <pthread.h>
#include <sstream>
#ifndef FPMLX_KERNEL_DIR
#error "FPMLX_KERNEL_DIR: the directory of mlx/kernels"
#endif
namespace fpmlx {
static bool child_of_fork = false;
static const bool watching = [] { pthread_atfork(nullptr, nullptr, [] { child_of_fork = true; }); return true; }();
bool in_forked_child() { return child_of_fork; }
Buf::Buf(const mx::array &a) {
  mx::eval(a);
  a_ = std::shared_ptr<mx::array>(new mx::array(a), [](mx::array *x) { if (!in_forked_child()) delete x; });
  p_ = reinterpret_cast<const blk::u64 *>(a_->data<uint64_t>());
  n_ = a_->size();
}
Buf::Buf(std::vector<blk::u64> host) : h_(std::make_shared<std::vector<blk::u64>>(std::move(host))) { p_ = h_->data(), n_ = h_->size(); }
static blk::E e_at(const blk::u64 *d, size_t i) { blk::E x; for (int k = 0; k < 3; k++) x.c[k] = blk::F{d[3 * i + k]}; return x; }
static void e_put(blk::u64 *d, size_t i, const blk::E &x) { for (int k = 0; k < 3; k++) d[3 * i + k] = x.c[k].v; }
std::string read_kernel_header(const char *name) {
  std::ifstream f(std::string(FPMLX_KERNEL_DIR) + "/" + name);
  if (!f) throw std::runtime_error(std::string("no kernel file ") + name);
  std::stringstream s;
  s << f.rdbuf();
  std::string t = s.str(), out;
  std::istringstream lines(t);
  for (std::string l; std::getline(lines, l);)
    if (l.rfind("#pragma once", 0)) out += l + "\n";
  return out;
}
static const char *HEADERS[] = {"field.h.metal", "blake2b.h.metal", "sumcheck.h.metal", "chacha.h.metal", "witness_ids.h", "witness.h.metal"};
static std::string all_headers() { std::string h; for (const char *f : HEADERS) h += read_kernel_header(f); return h; }
// One kernel by name, made once. n threads, in groups of up to 256.
static std::vector<mx::array> run(const std::string &name, const std::string &body, const std::vector<mx::array> &in, const std::vector<std::string> &names,
                                  const std::vector<mx::Shape> &shapes, size_t threads, size_t rows = 1, const std::vector<std::string> &outs = {"out"}) {
  static std::map<std::string, mx::fast::CustomKernelFunction> kernels;
  auto it = kernels.find(name);
  if (it == kernels.end()) it = kernels.emplace(name, mx::fast::metal_kernel(name, names, outs, body, all_headers())).first;
  BLK_REQUIRE(threads < (1ull << 31), "mlx: too many threads for one kernel");
  return it->second(in, shapes, std::vector<mx::Dtype>(shapes.size(), mx::uint64), {(int)threads, (int)rows, 1}, {(int)std::min<size_t>(threads, 256), 1, 1}, {}, std::nullopt, false, {});
}
std::vector<mx::array> run_kernel(const std::string &name, const std::string &body, const std::vector<mx::array> &in, const std::vector<std::string> &names,
                                  const std::vector<mx::Shape> &shapes, size_t threads, size_t rows) {
  return run(name, body, in, names, shapes, threads, rows);
}
static mx::array words(const std::vector<blk::u64> &v) { return mx::array(reinterpret_cast<const uint64_t *>(v.data()), {(int)v.size()}, mx::uint64); }
static mx::array words(const blk::u64 *v, size_t n) { return mx::array(reinterpret_cast<const uint64_t *>(v), {(int)n}, mx::uint64); }
static mx::array run1(const std::string &name, const std::string &body, const std::vector<mx::array> &in, const std::vector<std::string> &names, mx::Shape shape, size_t threads) {
  static std::map<std::string, mx::fast::CustomKernelFunction> kernels;
  auto it = kernels.find(name);
  if (it == kernels.end())
    it = kernels.emplace(name, mx::fast::metal_kernel(name, names, {"out"}, body, all_headers())).first;
  return it->second(in, {shape}, {mx::uint64}, {(int)threads, 1, 1}, {(int)std::min<size_t>(threads, 256), 1, 1}, {}, std::nullopt, false, {})[0];
}
mx::array f_binary(const char *op, const mx::array &a, const mx::array &b) {
  const std::string o(op);
  return run1("fp_f_" + o, "uint i = thread_position_in_grid.x; out[i] = gl_" + o + "(a[i], b[i]);", {a, b}, {"a", "b"}, a.shape(), a.size());
}
mx::array e_mul(const mx::array &a, const mx::array &b) {
  return run1("fp_e_mul",
              "uint i = thread_position_in_grid.x; GlE r = e_mul(GlE{a[3*i], a[3*i+1], a[3*i+2]}, GlE{b[3*i], b[3*i+1], b[3*i+2]});"
              "out[3*i] = r.c0; out[3*i+1] = r.c1; out[3*i+2] = r.c2;",
              {a, b}, {"a", "b"}, a.shape(), a.size() / 3);
}

Buf e_upload(const std::vector<blk::E> &v) {
  static_assert(sizeof(blk::E) == 24, "E is three words");
  const blk::u64 *w = reinterpret_cast<const blk::u64 *>(v.data());
  if (in_forked_child()) return Buf(std::vector<blk::u64>(w, w + 3 * v.size()));
  return Buf(words(w, 3 * v.size()));
}
std::vector<blk::E> e_download(const Buf &a) {
  std::vector<blk::E> out(a.size() / 3);
  if (!out.empty()) memcpy((void *)out.data(), a.data(), a.size() * 8);
  return out;
}
static Buf encode_array(const mx::array &coefficients, size_t rows, unsigned ld, unsigned log_n, const blk::F *tw);
Buf encode_rows(const blk::u64 *coefficients, size_t rows, unsigned ld, unsigned log_n, const blk::F *tw) {
  const size_t block = size_t(1) << ld, n = size_t(1) << log_n;
  if (in_forked_child()) {
    std::vector<blk::u64> w(rows * n);
    for (size_t r = 0; r < rows; r++) {
      blk::u64 *a = &w[r * n];
      for (size_t x = 0; x < n; x++) a[x] = coefficients[r * block + (x & (block - 1))];
      for (unsigned L = ld; L-- > 0;) {
        const size_t st = size_t(1) << L;
        for (size_t h = 0; h < (n >> (L + 1)); h++)
          for (size_t j = 0; j < st; j++) {
            blk::u64 &p = a[2 * st * h + j], &q = a[2 * st * h + st + j];
            const blk::u64 u = p, v = blk::GL64::mul(q, tw[(n >> (L + 1)) + h].v);
            p = blk::GL64::add(u, v), q = blk::GL64::sub(u, v);
          }
      }
    }
    return Buf(std::move(w));
  }
  return encode_array(words(coefficients, rows * block), rows, ld, log_n, tw);
}
static Buf encode_array(const mx::array &coefficients, size_t rows, unsigned ld, unsigned log_n, const blk::F *tw) {
  const size_t block = size_t(1) << ld, n = size_t(1) << log_n;
  const mx::array table = words(reinterpret_cast<const blk::u64 *>(tw), n);
  mx::array a = run("fp_spread", "uint x = thread_position_in_grid.x, row = thread_position_in_grid.y; ulong n = p[0], block = p[1]; out[row * n + x] = c[row * block + (x & (block - 1))];",
                    {coefficients, words(std::vector<blk::u64>{n, block})}, {"c", "p"}, {{(int)rows, (int)n}}, n, rows)[0];
  for (unsigned L = ld; L-- > 0;) {
    const mx::array p = words(std::vector<blk::u64>{n, L});
    a = run("fp_layer",
            "uint k = thread_position_in_grid.x, row = thread_position_in_grid.y; ulong n = p[0]; uint L = p[1], st = 1u << L;"
            "uint h = k >> L, j = k & (st - 1);"
            "ulong base = (ulong)row * n + 2ul * st * h + j; ulong t = tw[(n >> (L + 1)) + h];"
            "ulong u = a[base], v = gl_mul(a[base + st], t);"
            "out[base] = gl_add(u, v); out[base + st] = gl_sub(u, v);",
            {a, table, p}, {"a", "tw", "p"}, {{(int)rows, (int)n}}, n / 2, rows)[0];
  }
  return Buf(a);
}
// Each thread adds a stride of products; the partial sums end on the host. Sums in F are exact, so
// the order does not change them.
static blk::E sum_partials(const mx::array &p, size_t count, unsigned at, unsigned stride) {
  mx::eval(p);
  const uint64_t *d = p.data<uint64_t>();
  blk::E s = blk::E::zero();
  for (size_t i = 0; i < count; i++) {
    blk::E x;
    for (int k = 0; k < 3; k++) x.c[k] = blk::F{d[i * stride + at + k]};
    s += x;
  }
  return s;
}
blk::E dot(const Buf &A, const Buf &B) {
  const size_t n = A.size() / 3, T = std::min<size_t>(n, 1 << 16);
  if (in_forked_child()) {
    blk::E s = blk::E::zero();
    for (size_t j = 0; j < n; j++) s += e_at(A.data(), j) * e_at(B.data(), j);
    return s;
  }
  const mx::array &a = A.array(), &b = B.array();
  const mx::array p = run("fp_dot",
                          "uint i = thread_position_in_grid.x; ulong T = q[0], n = q[1]; GlE s = {0, 0, 0};"
                          "for (ulong j = i; j < n; j += T) s = e_add(s, e_mul(GlE{a[3*j], a[3*j+1], a[3*j+2]}, GlE{b[3*j], b[3*j+1], b[3*j+2]}));"
                          "out[3*i] = s.c0; out[3*i+1] = s.c1; out[3*i+2] = s.c2;",
                          {a, b, words(std::vector<blk::u64>{T, n})}, {"a", "b", "q"}, {{(int)(3 * T)}}, T)[0];
  return sum_partials(p, T, 0, 3);
}
void fold_round(const Buf &C, const Buf &M, size_t rows, size_t len, blk::E q[3]) {
  const size_t n = rows / 2 * len, T = std::min<size_t>(n, 1 << 16);
  if (in_forked_child()) {
    q[0] = q[1] = q[2] = blk::E::zero();
    for (size_t b = 0; b < rows; b += 2)
      for (size_t i = 0; i < len; i++) {
        const blk::E c0 = e_at(C.data(), b * len + i), dc = e_at(C.data(), (b + 1) * len + i) - c0;
        const blk::E m0 = e_at(M.data(), b * len + i), dm = e_at(M.data(), (b + 1) * len + i) - m0;
        q[0] += c0 * m0, q[1] += c0 * dm + dc * m0, q[2] += dc * dm;
      }
    return;
  }
  const mx::array &c = C.array(), &m = M.array();
  const mx::array p = run("fp_fold_round",
                          "uint i = thread_position_in_grid.x; ulong T = w[0], n = w[1], len = w[2]; GlE s0 = {0,0,0}, s1 = {0,0,0}, s2 = {0,0,0};"
                          "for (ulong e = i; e < n; e += T) { ulong b = e / len, k = e % len, lo = 3 * (2 * b * len + k), hi = lo + 3 * len;"
                          "GlE c0 = {c[lo], c[lo+1], c[lo+2]}, c1 = {c[hi], c[hi+1], c[hi+2]}, m0 = {m[lo], m[lo+1], m[lo+2]}, m1 = {m[hi], m[hi+1], m[hi+2]};"
                          "GlE dc = e_sub(c1, c0), dm = e_sub(m1, m0);"
                          "s0 = e_add(s0, e_mul(c0, m0)); s1 = e_add(s1, e_add(e_mul(c0, dm), e_mul(dc, m0))); s2 = e_add(s2, e_mul(dc, dm)); }"
                          "out[9*i] = s0.c0; out[9*i+1] = s0.c1; out[9*i+2] = s0.c2; out[9*i+3] = s1.c0; out[9*i+4] = s1.c1; out[9*i+5] = s1.c2; out[9*i+6] = s2.c0; out[9*i+7] = s2.c1; out[9*i+8] = s2.c2;",
                          {c, m, words(std::vector<blk::u64>{T, n, len})}, {"c", "m", "w"}, {{(int)(9 * T)}}, T)[0];
  for (int k = 0; k < 3; k++) q[k] = sum_partials(p, T, 3 * k, 9);
}
Buf halve(const Buf &T, size_t rows, size_t len, const blk::E &g) {
  const size_t n = rows / 2 * len;
  if (!n) return Buf(std::vector<blk::u64>());
  if (in_forked_child()) {
    std::vector<blk::u64> w(3 * n);
    for (size_t b = 0; b < rows / 2; b++)
      for (size_t i = 0; i < len; i++) {
        const blk::E lo = e_at(T.data(), 2 * b * len + i), hi = e_at(T.data(), (2 * b + 1) * len + i);
        e_put(w.data(), b * len + i, lo + g * (hi - lo));
      }
    return Buf(std::move(w));
  }
  const mx::array &t = T.array();
  const mx::array out = run("fp_halve",
                            "uint e = thread_position_in_grid.x; ulong len = w[0]; ulong b = e / len, k = e % len, lo = 3 * (2 * b * len + k), hi = lo + 3 * len;"
                            "GlE a = {t[lo], t[lo+1], t[lo+2]}, h = {t[hi], t[hi+1], t[hi+2]}, y = {w[1], w[2], w[3]};"
                            "GlE r = e_add(a, e_mul(y, e_sub(h, a))); out[3*e] = r.c0; out[3*e+1] = r.c1; out[3*e+2] = r.c2;",
                            {t, words(std::vector<blk::u64>{len, g.c[0].v, g.c[1].v, g.c[2].v})}, {"t", "w"}, {{(int)(3 * n)}}, n)[0];
  return Buf(out);
}
// Proof of work on the GPU, as chain_work_passes(): one thread for each nonce of a batch; the batches
// rise, so the least passing nonce of the first batch with one is the smallest of all.
blk::u64 search_nonce(const blk::ChainPrefix &prefix, unsigned bits) {
  if (in_forked_child() || bits < 12) {
    for (blk::u64 nonce = 0;; nonce++)
      if (blk::chain_work_passes(prefix, nonce, bits)) return nonce;
  }
  static auto kernel = mx::fast::metal_kernel("fp_nonce", {"pre", "q"}, {"out"},
    R"(uint i = thread_position_in_grid.x; ulong nonce = q[0] + i; uint bits = (uint)q[1];
       uint fill = (uint)pre[24], steps = (uint)(pre[24] >> 32), at = fill / 8; // words of 64 bits
       ulong h[8], m[16]; for (int k = 0; k < 8; k++) h[k] = pre[k];
       for (int k = 0; k < 16; k++) m[k] = pre[8 + k];
       uint bytes = steps * 128 + fill + 8;
       if (at == 16) { b2_compress(h, m, 128ul * (steps + 1), 0, false); for (int k = 0; k < 16; k++) m[k] = 0; m[0] = nonce; }
       else m[at] = nonce;
       b2_compress(h, m, bytes, 0, true);
       if ((h[0] << (64 - bits)) == 0) atomic_fetch_min_explicit((device atomic_uint*)out, i, memory_order_relaxed);)",
    all_headers(), true, true);
  static_assert(sizeof(blk::ChainPrefix) == 200, "16 + 32 words of the state, fill and steps");
  std::vector<blk::u64> pre(25);
  memcpy(pre.data(), &prefix, sizeof prefix);
  const mx::array P = words(pre);
  const int batch = 1 << 24;
  for (blk::u64 base = 0;; base += batch) {
    const mx::array r = kernel({P, words(std::vector<blk::u64>{base, bits})}, {{1}}, {mx::uint32}, {batch, 1, 1}, {256, 1, 1}, {}, std::optional<float>(4294967040.0f), false, {})[0];
    mx::eval(r);
    const uint32_t got = r.data<uint32_t>()[0];
    if (got < (uint32_t)batch) {
      BLK_REQUIRE(blk::chain_work_passes(prefix, base + got, bits), "search: the GPU's nonce does not pass");
      return base + got;
    }
  }
}
OracleTreeMlx::OracleTreeMlx(const blk::OracleTreeShape &s, blk::Coins &coins) : s_(s) {
  BLK_REQUIRE(s_.valid(), "oracle tree: not a shape");
  salt_.resize(((size_t)1 << s_.log_n) * s_.salt_bytes);
  coins.bytes(salt_.data(), salt_.size());
}
blk::Digest OracleTreeMlx::digest(unsigned d, blk::u64 place) const {
  const blk::u64 *w = level_[d].data() + 4 * place;
  blk::Digest r;
  for (int i = 0; i < 4; i++) r.w[2 * i] = (blk::u32)w[i], r.w[2 * i + 1] = (blk::u32)(w[i] >> 32);
  return r;
}
static std::vector<blk::u64> start_words(const blk::Tag &tag) {
  std::vector<blk::u64> v(4);
  for (int i = 0; i < 4; i++) v[i] = tag.start.w[2 * i] | (blk::u64)tag.start.w[2 * i + 1] << 32;
  return v;
}
blk::Digest OracleTreeMlx::build(const std::vector<Buf> &main, const std::vector<Buf> &mask) {
  const blk::LeafHash lh(s_);
  if (in_forked_child()) {
    const size_t n = (size_t)1 << s_.log_n, nz = (size_t)1 << (s_.has_main() ? s_.log_n_zk : s_.log_n);
    std::vector<blk::Digest> leaves(n);
    std::vector<blk::F> col;
    for (size_t x = 0; x < n; x++) {
      col.clear();
      for (const Buf &b : main) for (size_t r = 0; r < b.size() / n; r++) col.push_back(blk::F{b.data()[r * n + x]});
      for (const Buf &b : mask) for (size_t r = 0; r < b.size() / nz; r++) col.push_back(blk::F{b.data()[r * nz + (x & (nz - 1))]});
      leaves[x] = lh(x, col.data(), col.size(), salt_.data() + x * s_.salt_bytes);
    }
    const blk::TreeShape ts = s_.tree();
    const blk::Tag tag = ts.vertex_tag();
    level_.assign(s_.log_n + 1, Buf());
    auto pack = [](const std::vector<blk::Digest> &v) { std::vector<blk::u64> w(4 * v.size()); for (size_t i = 0; i < v.size(); i++) for (int k = 0; k < 4; k++) w[4 * i + k] = v[i].w[2 * k] | (blk::u64)v[i].w[2 * k + 1] << 32; return Buf(std::move(w)); };
    level_[s_.log_n] = pack(leaves);
    for (unsigned d = s_.log_n; d > 0; d--) {
      std::vector<blk::Digest> up((size_t)1 << (d - 1));
      for (size_t i = 0; i < up.size(); i++) up[i] = blk::vertex(tag, leaves[2 * i], leaves[2 * i + 1]);
      leaves = std::move(up);
      level_[d - 1] = pack(leaves);
    }
    return digest(0, 0);
  }
  const size_t n = (size_t)1 << s_.log_n, symbols = s_.width();
  const size_t salt_words = s_.salt_bytes / 8, salt_at = blk::leaf_salt_at(symbols, s_.salt_bytes) / 8, total = blk::leaf_bytes(symbols, s_.salt_bytes);
  // All main rows, then all mask rows, as one array of words, each at its own length.
  std::vector<mx::array> flat;
  for (const Buf &a : main) flat.push_back(a.array());
  for (const Buf &a : mask) flat.push_back(a.array());
  const mx::array words_all = flat.empty() ? mx::zeros({1}, mx::uint64) : flat.size() == 1 ? flat[0] : mx::concatenate(flat);
  const size_t main_w = s_.main_width();
  const size_t nz = s_.has_mask() ? (size_t)1 << (s_.has_main() ? s_.log_n_zk : s_.log_n) : 1;
  std::vector<blk::u64> par = {n, symbols, main_w, nz, salt_words, salt_at, total, lh.tag.word, (blk::u64)s_.log_n};
  const std::vector<blk::u64> st = start_words(lh.tag);
  par.insert(par.end(), st.begin(), st.end());
  const mx::array salts = words(reinterpret_cast<const blk::u64 *>(salt_.data()), salt_.size() / 8);
  mx::array leaves = run("fp_leaves",
    "uint x = thread_position_in_grid.x; ulong n = p[0], symbols = p[1], main_w = p[2], nz = p[3], salt_w = p[4], salt_at = p[5], total = p[6];"
    "ulong st[4] = {p[9], p[10], p[11], p[12]}; ulong h[8]; b2_begin(h, st); ulong m[16]; ulong words_n = total / 8, at = 0; uint step = 0;"
    "ulong xs = x & (nz - 1);"
    "while (at < words_n) { for (int k = 0; k < 16; k++) { ulong i = at + k, v = 0;"
    " if (i == 0) v = p[7]; else if (i == 1) v = n + x;"
    " else if (i - 2 < symbols) { ulong y = i - 2; v = y < main_w ? w[y * n + x] : w[main_w * n + (y - main_w) * nz + xs]; }"
    " else if (i >= salt_at && i < salt_at + salt_w) v = salt[(ulong)x * salt_w + (i - salt_at)];"
    " m[k] = v; }"
    " at += 16; bool last = at >= words_n; b2_compress(h, m, last ? total : 128ul * (step + 1), 0, last); step++; }"
    "for (int k = 0; k < 4; k++) out[4 * (ulong)x + k] = h[k];",
    {words_all, salts, words(par)}, {"w", "salt", "p"}, {{(int)(4 * n)}}, n)[0];
  return levels(leaves);
}
blk::Digest OracleTreeMlx::levels(const mx::array &leaves) {
  const blk::Tag vt = s_.tree().vertex_tag();
  std::vector<blk::u64> vp = {vt.word};
  const std::vector<blk::u64> vs = start_words(vt);
  vp.insert(vp.end(), vs.begin(), vs.end());
  const mx::array vpa = words(vp);
  level_.assign(s_.log_n + 1, Buf());
  mx::array level = leaves;
  level_[s_.log_n] = Buf(level);
  for (unsigned d = s_.log_n; d > 0; d--) {
    const size_t half_n = (size_t)1 << (d - 1);
    level = run("fp_vertex",
      "uint i = thread_position_in_grid.x; ulong st[4] = {p[1], p[2], p[3], p[4]}; ulong h[8]; b2_begin(h, st); ulong m[16]; m[0] = p[0]; m[1] = 0;"
      "for (int k = 0; k < 8; k++) m[2 + k] = c[8 * (ulong)i + k]; for (int k = 10; k < 16; k++) m[k] = 0;"
      "b2_compress(h, m, 80, 0, true); for (int k = 0; k < 4; k++) out[4 * (ulong)i + k] = h[k];",
      {level, vpa}, {"c", "p"}, {{(int)(4 * half_n)}}, half_n)[0];
    level_[d - 1] = Buf(level);
  }
  return digest(0, 0);
}
blk::Digest OracleTreeMlx::build_chunks(const std::vector<Buf> &chunks, size_t rows_each) {
  const blk::LeafHash lh(s_);
  const size_t n = (size_t)1 << s_.log_n, symbols = s_.width(), K = chunks.size();
  const size_t salt_words = s_.salt_bytes / 8, salt_at = blk::leaf_salt_at(symbols, s_.salt_bytes) / 8, total = blk::leaf_bytes(symbols, s_.salt_bytes);
  std::vector<blk::u64> par = {n, symbols, rows_each, salt_words, salt_at, total, lh.tag.word};
  const std::vector<blk::u64> st = start_words(lh.tag);
  par.insert(par.end(), st.begin(), st.end());
  std::vector<mx::array> in;
  std::vector<std::string> names;
  std::string list;
  for (size_t k = 0; k < K; k++) { in.push_back(chunks[k].array()); names.push_back("w" + std::to_string(k)); list += (k ? ", w" : "w") + std::to_string(k); }
  in.push_back(words(reinterpret_cast<const blk::u64 *>(salt_.data()), salt_.size() / 8)), names.push_back("salt");
  in.push_back(words(par)), names.push_back("p");
  const std::string body =
    "uint x = thread_position_in_grid.x; ulong n = p[0], symbols = p[1], each = p[2], salt_w = p[3], salt_at = p[4], total = p[5];"
    "const device ulong* ws[] = {" + list + "};"
    "ulong st[4] = {p[7], p[8], p[9], p[10]}; ulong h[8]; b2_begin(h, st); ulong m[16]; ulong words_n = total / 8, at = 0; uint step = 0;"
    "while (at < words_n) { for (int k = 0; k < 16; k++) { ulong i = at + k, v = 0;"
    " if (i == 0) v = p[6]; else if (i == 1) v = n + x;"
    " else if (i - 2 < symbols) { ulong y = i - 2; v = ws[y / each][(y % each) * n + x]; }"
    " else if (i >= salt_at && i < salt_at + salt_w) v = salt[(ulong)x * salt_w + (i - salt_at)];"
    " m[k] = v; }"
    " at += 16; bool last = at >= words_n; b2_compress(h, m, last ? total : 128ul * (step + 1), 0, last); step++; }"
    "for (int k = 0; k < 4; k++) out[4 * (ulong)x + k] = h[k];";
  return levels(run("fp_leaves_chunks_" + std::to_string(K), body, in, names, {{(int)(4 * n)}}, n)[0]);
}
// The rounds of a sumcheck.
static mx::array e_array(const std::vector<blk::E> &v) { return words(reinterpret_cast<const blk::u64 *>(v.data()), 3 * v.size()); }
static std::vector<blk::u64> e_words(const blk::E &x) { return {x.c[0].v, x.c[1].v, x.c[2].v}; }
static mx::array e_scalars(const std::vector<blk::E> &v) { return e_array(v.empty() ? std::vector<blk::E>(1, blk::E::zero()) : v); }
// One thread takes pairs q = tid, tid + T, ..; a group of 256 sums its threads, node by node.
static std::string sum_source(const SumShape &s, const std::vector<unsigned> &used, unsigned d) {
  std::ostringstream o;
  o << "uint tid = thread_position_in_grid.x, lid = thread_position_in_threadgroup.x, grp = threadgroup_position_in_grid.x;\n"
       "ulong pairs = p[0], step = p[1], T = p[2], size = p[3];\n"
       "threadgroup ulong sh[768];\n";
  for (size_t i = 0; i < s.coeff.size(); i++) o << "GlE c" << i << " = e_ld(cf, " << i << ");\n";
  for (unsigned n = 0; n < d; n++) o << "GlE acc" << n << " = {0, 0, 0};\n";
  o << "for (ulong q = tid; q < pairs; q += T) {\n"
       "  ulong low = q & (step - 1), i0 = ((q - low) << 1) | low, i1 = i0 | step;\n"
       "  GlE w = e_ld(wt, q);\n";
  for (unsigned k : used) o << "  GlE v" << k << " = e_ld(m, " << k << " * size + i0), d" << k << " = e_sub(e_ld(m, " << k << " * size + i1), v" << k << ");\n";
  for (unsigned n = 0; n < d; n++) {
    o << "  {\n    GlE h = {0, 0, 0};\n";
    for (size_t i = 0; i < s.coeff.size(); i++) {
      o << "    { GlE t = c" << i << ";";
      for (unsigned k : s.maps[i]) o << " t = e_mul(t, v" << k << ");";
      o << " h = e_add(h, t); }\n";
    }
    o << "    acc" << n << " = e_add(acc" << n << ", e_mul(w, h));\n  }\n";
    if (n + 1 < d) for (unsigned k : used) o << "  v" << k << " = e_add(v" << k << ", d" << k << ");\n";
  }
  o << "}\n";
  for (unsigned n = 0; n < d; n++)
    o << "{ sh[3 * lid] = acc" << n << ".c0; sh[3 * lid + 1] = acc" << n << ".c1; sh[3 * lid + 2] = acc" << n << ".c2;\n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  for (uint s = 128; s > 0; s >>= 1) {\n"
         "    if (lid < s) for (uint c = 0; c < 3; c++) sh[3 * lid + c] = gl_add(sh[3 * lid + c], sh[3 * (lid + s) + c]);\n"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         "  }\n"
         "  if (lid == 0) for (uint c = 0; c < 3; c++) out[((ulong)grp * " << d << " + " << n << ") * 3 + c] = sh[c];\n"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);\n}\n";
  return o.str();
}
// A summand too long to unroll: its monomials are read from pg: first the offsets of each monomial's
// factors, then the factors. A thread keeps its sums in acc[], up to 16 nodes.
static const char *SUM_ANY = R"(
uint tid = thread_position_in_grid.x, lid = thread_position_in_threadgroup.x, grp = threadgroup_position_in_grid.x;
ulong pairs = p[0], step = p[1], T = p[2], size = p[3]; uint d = p[4], nm = p[5];
threadgroup ulong sh[768];
GlE acc[16];
for (uint n = 0; n < d; n++) acc[n] = GlE{0, 0, 0};
for (ulong q = tid; q < pairs; q += T) {
  ulong low = q & (step - 1), i0 = ((q - low) << 1) | low, i1 = i0 | step;
  GlE w = e_ld(wt, q);
  for (uint n = 0; n < d; n++) {
    GlE h = {0, 0, 0};
    for (uint i = 0; i < nm; i++) {
      GlE t = e_ld(cf, i);
      for (ulong f = pg[i]; f < pg[i + 1]; f++) {
        ulong at = pg[f] * size;
        GlE a = e_ld(m, at + i0);
        t = e_mul(t, e_add(a, e_scale(e_sub(e_ld(m, at + i1), a), (ulong)n)));
      }
      h = e_add(h, t);
    }
    acc[n] = e_add(acc[n], e_mul(w, h));
  }
}
for (uint n = 0; n < d; n++) {
  sh[3 * lid] = acc[n].c0; sh[3 * lid + 1] = acc[n].c1; sh[3 * lid + 2] = acc[n].c2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = 128; s > 0; s >>= 1) {
    if (lid < s) for (uint c = 0; c < 3; c++) sh[3 * lid + c] = gl_add(sh[3 * lid + c], sh[3 * (lid + s) + c]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) for (uint c = 0; c < 3; c++) out[((ulong)grp * d + n) * 3 + c] = sh[c];
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
)";
static size_t sum_factors(const SumShape &s) { size_t n = s.coeff.size(); for (const auto &m : s.maps) n += m.size(); return n; }
static std::string sum_name(const SumShape &s, unsigned d) {
  std::string n = "fp_sum" + std::to_string(d);
  for (const auto &m : s.maps) { n += "_"; for (unsigned k : m) n += "x" + std::to_string(k); }
  return n;
}
static mx::array sum_program(const SumShape &s) {
  std::vector<blk::u64> p(s.maps.size() + 1);
  p[0] = p.size();
  for (size_t i = 0; i < s.maps.size(); i++) {
    for (unsigned k : s.maps[i]) p.push_back(k);
    p[i + 1] = p.size();
  }
  return words(p);
}
static mx::array packed(const std::vector<std::vector<blk::E>> &maps) {
  const size_t each = maps[0].size() * 3;
  BLK_REQUIRE(each * maps.size() < (size_t(1) << 31), "sumcheck: too many values for one array");
  mx::array a(mx::allocator::malloc(each * maps.size() * 8), {(int)(each * maps.size())}, mx::uint64);
  for (size_t k = 0; k < maps.size(); k++) memcpy(a.data<uint64_t>() + k * each, maps[k].data(), each * 8);
  return a;
}
SumRounds::SumRounds(std::vector<std::vector<blk::E>> &maps, SumShape shape, const std::vector<SumWeight> &later)
    : shape_(std::move(shape)), m_(packed(maps)), coeff_(e_scalars(shape_.coeff)), program_(sum_program(shape_)), count_(maps.size()), size_(maps[0].size()) {
  for (auto &m : maps) std::vector<blk::E>().swap(m);
  prepare(later);
}
SumRounds::SumRounds(const mx::array &maps, unsigned count, SumShape shape, const std::vector<SumWeight> &later)
    : shape_(std::move(shape)), m_(maps), coeff_(e_scalars(shape_.coeff)), program_(sum_program(shape_)), count_(count), size_(maps.size() / 3 / count) {
  prepare(later);
}
void SumRounds::prepare(const std::vector<SumWeight> &later) {
  std::vector<char> seen(count_, 0);
  for (const auto &m : shape_.maps) for (unsigned k : m) seen[k] = 1;
  for (unsigned k = 0; k < count_; k++) if (seen[k]) used_.push_back(k);
  // From the last round back: each table is the next one spread over one more index bit.
  std::vector<mx::array> w(1, e_array({blk::E::one()}));
  for (size_t l = later.size(); l-- > 0;) {
    const SumWeight &x = later[l];
    const size_t n = size_t(1) << (later.size() - l);
    std::vector<blk::u64> p = {x.at};
    for (const blk::E &e : {x.lo, x.hi}) for (blk::u64 c : e_words(e)) p.push_back(c);
    w.push_back(run("fp_sum_weights",
                    "uint q = thread_position_in_grid.x, at = p[0]; uint low = q & ((1u << at) - 1), below = ((q >> (at + 1)) << at) | low;"
                    "GlE f = ((q >> at) & 1) ? GlE{p[4], p[5], p[6]} : GlE{p[1], p[2], p[3]};"
                    "e_st(out, q, e_mul(f, e_ld(a, below)));",
                    {w.back(), words(p)}, {"a", "p"}, {{(int)(3 * n)}}, n)[0]);
  }
  std::reverse(w.begin(), w.end());
  w_ = std::move(w);
}
std::vector<blk::E> SumRounds::nodes(size_t l, unsigned d, size_t step) {
  const size_t pairs = size_ / 2, T = std::max<size_t>(256, std::min<size_t>(pairs, size_t(1) << 15)), G = T / 256; // the sums of a group take 256 threads
  const bool any = (sum_factors(shape_) + used_.size()) * d > 96;
  BLK_REQUIRE(d <= 16, "sumcheck on the GPU: a round of too high a degree");
  const mx::array p = words(std::vector<blk::u64>{pairs, step, T, size_, d, shape_.coeff.size()});
  const mx::array out = any ? run("fp_sum_any", SUM_ANY, {m_, w_[l], coeff_, program_, p}, {"m", "wt", "cf", "pg", "p"}, {{(int)(G * d * 3)}}, T)[0]
                            : run(sum_name(shape_, d), sum_source(shape_, used_, d), {m_, w_[l], coeff_, p}, {"m", "wt", "cf", "p"}, {{(int)(G * d * 3)}}, T)[0];
  mx::eval(out);
  const blk::u64 *o = reinterpret_cast<const blk::u64 *>(out.data<uint64_t>());
  std::vector<blk::E> v(d, blk::E::zero());
  for (size_t g = 0; g < G; g++) for (unsigned n = 0; n < d; n++) v[n] = v[n] + e_at(o, g * d + n);
  return v;
}
void SumRounds::bind(const blk::E &r, size_t step) {
  const size_t pairs = size_ / 2;
  std::vector<blk::u64> p = {step, size_, pairs};
  for (blk::u64 c : e_words(r)) p.push_back(c);
  m_ = run("fp_sum_bind",
           "ulong i = thread_position_in_grid.x, row = thread_position_in_grid.y, step = p[0], size = p[1], pairs = p[2];"
           "ulong low = i & (step - 1), i0 = row * size + (((i - low) << 1) | low), i1 = i0 + step;"
           "GlE a = e_ld(m, i0); e_st(out, row * pairs + i, e_add(a, e_mul(GlE{p[3], p[4], p[5]}, e_sub(e_ld(m, i1), a))));",
           {m_, words(p)}, {"m", "p"}, {{(int)(3 * pairs * count_)}}, pairs, count_)[0];
  size_ = pairs;
}
void SumRounds::fetch(std::vector<std::vector<blk::E>> &maps) const {
  mx::eval(m_);
  maps.resize(count_);
  for (size_t k = 0; k < count_; k++) {
    maps[k].resize(size_);
    memcpy((void *)maps[k].data(), m_.data<uint64_t>() + k * size_ * 3, size_ * 24);
  }
}
// Writing out a Form's Tensors. A group of 256 threads owns 256 entries of the vector, which differ in
// the 8 `lane` bits; the other bits number the group. Rows are low index bits and many Tensors fix them,
// so the lane bits are those that the Tensors, by their size, fix least. Each group has the list of the
// Tensors that can reach it; no two threads write one entry, and no entry is written twice.
static const char *TENSORS = R"(
uint lane = thread_position_in_threadgroup.x, group = threadgroup_position_in_grid.x;
ulong lanes = p[0], chunks = p[1], j = 0;
{ uint a = 0, b = 0;
  for (uint t = 0; t < 63; t++) {
    if ((lanes >> t) & 1) j |= (ulong)((lane >> a++) & 1) << t;
    else if ((chunks >> t) & 1) j |= (ulong)((group >> b++) & 1) << t;
  }
}
GlE acc = {0, 0, 0};
for (ulong e = cs[group]; e < cs[group + 1]; e++) {
  const ulong t = ct[e], at = 7 * t;
  if ((j & meta[at]) != meta[at + 1]) continue;
  GlE r = e_ld(scal, t);
  const ulong p0 = meta[at + 2], pn = meta[at + 3];
  for (ulong i = p0; i < p0 + pn; i++) r = e_mul(r, e_ld(pr, 2 * i + ((j >> pb[i]) & 1)));
  const ulong d0 = meta[at + 4], dn = meta[at + 5];
  if (dn != 0xFFFFFFFFul) {
    ulong d = 0;
    for (ulong i = 0; i < dn; i++) d |= ((j >> dv[d0 + i]) & 1) << i;
    r = e_mul(r, e_ld(vals, meta[at + 6] + d));
  }
  acc = e_add(acc, r);
}
e_st(out, j, acc);
)";
static blk::u64 bits_taken(blk::u64 x, blk::u64 mask) { // the bits of x at mask, packed low
  blk::u64 r = 0; unsigned n = 0;
  for (; mask; mask &= mask - 1, n++) r |= ((x >> __builtin_ctzll(mask)) & 1) << n;
  return r;
}
Buf tensors_written(const std::vector<blk::Tensor> &ts, unsigned vars) {
  BLK_REQUIRE(vars >= 8 && vars < 31, "tensors on the GPU: a vector of this length");
  const size_t T = ts.size();
  // Lane bits.
  std::vector<double> weight(vars, 0.0);
  for (const blk::Tensor &t : ts) {
    const double size = std::ldexp(1.0, (int)(vars - __builtin_popcountll(t.pick_mask())));
    for (unsigned b = 0; b < vars; b++) if (t.picks(b)) weight[b] += size;
  }
  std::vector<unsigned> order(vars);
  for (unsigned b = 0; b < vars; b++) order[b] = b;
  std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return weight[a] < weight[b]; });
  blk::u64 lanes = 0;
  for (unsigned i = 0; i < 8; i++) lanes |= blk::u64(1) << order[i];
  const blk::u64 all = (blk::u64(1) << vars) - 1, chunks = all & ~lanes;
  const size_t C = size_t(1) << (vars - 8);
  // The Tensors as data.
  std::vector<blk::u64> meta(7 * std::max<size_t>(T, 1), 0), pb, dv, vals;
  std::vector<blk::E> scal, pr;
  std::map<const std::vector<blk::E> *, blk::u64> placed;
  for (size_t i = 0; i < T; i++) {
    const blk::Tensor &t = ts[i];
    BLK_REQUIRE(t.vars() == vars && !t.is_held(), "tensors on the GPU: another length, or a held factor");
    blk::u64 *m = &meta[7 * i];
    m[0] = t.pick_mask(), m[1] = t.pick_value(), m[2] = pb.size();
    const blk::E *real = t.real_pairs();
    unsigned r = 0;
    for (unsigned b = 0; b < vars; b++) {
      if (t.picks(b)) continue;
      const blk::E &lo = real[2 * r], &hi = real[2 * r + 1];
      r++;
      if (lo == blk::E::one() && hi == blk::E::one()) continue;
      pb.push_back(b), pr.push_back(lo), pr.push_back(hi);
    }
    m[3] = pb.size() - m[2];
    scal.push_back(t.scalar);
    m[5] = 0xFFFFFFFFul;
    if (t.has_dense()) {
      m[4] = dv.size(), m[5] = t.dense_vars.size();
      for (unsigned d : t.dense_vars) dv.push_back(d);
      auto it = placed.find(t.values.get());
      if (it == placed.end()) {
        it = placed.emplace(t.values.get(), vals.size() / 3).first;
        const blk::u64 *w = reinterpret_cast<const blk::u64 *>(t.values->data());
        vals.insert(vals.end(), w, w + 3 * t.values->size());
      }
      m[6] = it->second + t.first;
    }
  }
  // Which Tensors reach which group.
  std::vector<blk::u64> cs(C + 1, 0), ct;
  auto each_chunk = [&](const blk::Tensor &t, auto &&body) {
    const blk::u64 fixed = bits_taken(t.pick_mask(), chunks), value = bits_taken(t.pick_value(), chunks), free = (C - 1) & ~fixed;
    blk::u64 sub = 0;
    do { body(sub | value); sub = (sub - free) & free; } while (sub);
  };
  for (const blk::Tensor &t : ts) each_chunk(t, [&](blk::u64 c) { cs[c + 1]++; });
  for (size_t c = 0; c < C; c++) cs[c + 1] += cs[c];
  ct.resize(std::max<blk::u64>(cs[C], 1));
  { std::vector<blk::u64> at(cs.begin(), cs.end() - 1);
    for (size_t i = 0; i < T; i++) each_chunk(ts[i], [&](blk::u64 c) { ct[at[c]++] = i; }); }
  auto some = [](std::vector<blk::u64> &v) -> std::vector<blk::u64> & { if (v.empty()) v.push_back(0); return v; };
  return Buf(run("fp_tensors", TENSORS,
                 {words(meta), e_scalars(scal), e_scalars(pr), words(some(pb)), words(some(dv)), words(some(vals)), words(cs), words(ct), words(std::vector<blk::u64>{lanes, chunks})},
                 {"meta", "scal", "pr", "pb", "dv", "vals", "cs", "ct", "p"}, {{(int)(3 * (size_t(1) << vars))}}, size_t(1) << vars)[0]);
}
Buf rows_first(const Buf &dense, size_t rows, size_t ell) {
  return Buf(run("fp_rows_first",
                 "ulong b = thread_position_in_grid.x, i = thread_position_in_grid.y, rows = p[0], ell = p[1]; e_st(out, b * ell + i, e_ld(a, b + rows * i));",
                 {dense.array(), words(std::vector<blk::u64>{rows, ell})}, {"a", "p"}, {{(int)(3 * rows * ell)}}, rows, ell)[0]);
}
Buf chacha_words(const blk::u32 key[8], blk::u32 stream, size_t n) {
  std::vector<blk::u64> p = {stream, n};
  for (int i = 0; i < 8; i++) p.push_back(key[i]);
  const size_t calls = (n + 3) / 4;
  return Buf(run("fp_chacha",
                 "ulong call = thread_position_in_grid.x, n = p[1]; uint key[8]; for (int i = 0; i < 8; i++) key[i] = (uint)p[2 + i];"
                 "ulong lane[4]; chacha_lanes(lane, key, (uint)call, (uint)p[0], 0, (uint)(call >> 32));"
                 "for (ulong i = 0; i < 4 && 4 * call + i < n; i++) out[4 * call + i] = lane[i];",
                 {words(p)}, {"p"}, {{(int)n}}, calls)[0]);
}
Buf combinations(const int32_t *cells, size_t m, size_t k, const signed char *theta, size_t tau) {
  const mx::array c(cells, {(int)(m * k)}, mx::int32), t(reinterpret_cast<const int8_t *>(theta), {(int)(k * tau)}, mx::int8);
  return Buf(run("fp_combinations",
                 "ulong i = thread_position_in_grid.x, u = thread_position_in_grid.y, m = p[0], k = p[1], tau = p[2];"
                 "long s = 0; for (ulong x = 0; x < k; x++) s += (long)th[x * tau + u] * (long)ce[i * k + x];"
                 "out[u * m + i] = (s < -(1l << 17) || s >= (1l << 17)) ? ~0ul : (ulong)(s + (1l << 17));",
                 {c, t, words(std::vector<blk::u64>{m, k, tau})}, {"ce", "th", "p"}, {{(int)(m * tau)}}, m, tau)[0]);
}
mx::array product_level(const mx::array &below) {
  const size_t h = below.size() / 6;
  return run("fp_product_level", "ulong i = thread_position_in_grid.x, h = p[0]; e_st(out, i, e_mul(e_ld(a, i), e_ld(a, i + h)));",
             {below, words(std::vector<blk::u64>{h})}, {"a", "p"}, {{(int)(3 * h)}}, h)[0];
}
std::vector<blk::E> e_fetch(const mx::array &a) {
  mx::eval(a);
  std::vector<blk::E> v(a.size() / 3);
  memcpy((void *)v.data(), a.data<uint64_t>(), a.size() * 8);
  return v;
}
mx::array e_on_gpu(const std::vector<blk::E> &v) { return e_array(v); }
mx::array lookup_leaves(const LeafPlan &L, const std::vector<blk::u64> &keys, const std::vector<char> &padding) {
  std::vector<blk::u64> p = {L.rK, L.dT, L.NT};
  for (const blk::E &e : {L.alpha, L.theta, L.pad0}) for (blk::u64 c : e_words(e)) p.push_back(c);
  const mx::array pad(reinterpret_cast<const uint8_t *>(padding.data()), {(int)padding.size()}, mx::uint8);
  const size_t n = size_t(1) << L.v;
  return run("fp_lookup_leaves",
             "ulong i = thread_position_in_grid.x, rK = p[0], dT = p[1], NT = p[2], y = i & ((1ul << rK) - 1), x = i >> rK;"
             "if (y >= NT) { e_st(out, i, GlE{p[9], p[10], p[11]}); return; }"
             "ulong c = tc[y], at = tf[y] + x, value = pd[at] ? pv[c] : ks[at];"
             "GlE keyed = e_scale(GlE{p[6], p[7], p[8]}, gl_canonical(c));"
             "keyed.c0 = gl_add(keyed.c0, gl_canonical(value));"
             "e_st(out, i, e_sub(GlE{p[3], p[4], p[5]}, keyed));",
             {words(L.tree_call), words(L.tree_first), words(L.pad_value), words(keys), pad, words(p)}, {"tc", "tf", "pv", "ks", "pd", "p"}, {{(int)(3 * n)}}, n)[0];
}
Buf column_sums(const int32_t *cells, size_t k, size_t nb, const std::vector<blk::E> &phi) {
  const size_t len = phi.size();
  const mx::array c(cells, {(int)(nb * len * k)}, mx::int32);
  return Buf(run("fp_column_sums",
                 "ulong x = thread_position_in_grid.x, b = thread_position_in_grid.y, k = p[0], nb = p[1], len = p[2]; GlE s = {0, 0, 0};"
                 "for (ulong r = 0; r < len; r++) { long v = ce[(b * len + r) * k + x]; if (v) s = e_add(s, e_scale(e_ld(ph, r), v < 0 ? GL_P - (ulong)(-v) : (ulong)v)); }"
                 "e_st(out, x * nb + b, s);",
                 {c, e_array(phi), words(std::vector<blk::u64>{k, nb, len})}, {"ce", "ph", "p"}, {{(int)(3 * k * nb)}}, k, nb)[0]);
}
Buf bytes_left(const int8_t *cells, size_t rows, size_t columns, const std::vector<blk::E> &g) {
  const mx::array c(cells, {(int)(rows * columns)}, mx::int8);
  return Buf(run("fp_bytes_left",
                 "ulong x = thread_position_in_grid.x, columns = p[0], rows = p[1]; GlE s = {0, 0, 0};"
                 "for (ulong y = 0; y < rows; y++) { long v = ce[y * columns + x]; if (v) s = e_add(s, e_scale(e_ld(g, y), v < 0 ? GL_P - (ulong)(-v) : (ulong)v)); }"
                 "e_st(out, x, s);",
                 {c, e_array(g), words(std::vector<blk::u64>{columns, rows})}, {"ce", "g", "p"}, {{(int)(3 * columns)}}, columns)[0]);
}
TableWords table_words(const blk::Table &t) {
  TableWords w;
  w.deg = t.deg, w.log_rows = t.log_rows, w.rows = t.rows, w.len = t.len;
  const blk::u64 *p = reinterpret_cast<const blk::u64 *>(t.a.data());
  w.a = t.a.empty() ? Buf(words(std::vector<blk::u64>(8, 0))) : Buf(words(p, t.a.size()));
  return w;
}
blk::E dot_table(const Buf &W, const TableWords &t) {
  const size_t n = t.len << t.log_rows, T = std::min<size_t>(n, 1 << 16);
  const mx::array p = run("fp_dot_table", R"(
    ulong i = thread_position_in_grid.x, T = q[0], n = q[1], deg = q[2], lr = q[3], rows = q[4], len = q[5]; GlE s = {0, 0, 0};
    for (ulong j = i; j < n; j += T) {
      ulong row = j & ((1ul << lr) - 1), place = j >> lr;
      if (row >= rows) continue;
      ulong at = (row * len + place) * deg;
      GlE x = deg == 1 ? GlE{a[at], 0, 0} : GlE{a[at], a[at + 1], a[at + 2]};
      s = e_add(s, e_mul(e_ld(w, j), x));
    }
    e_st(out, i, s);)", {W.array(), t.a.array(), words(std::vector<blk::u64>{T, n, t.deg, t.log_rows, t.rows, t.len})}, {"w", "a", "q"}, {{(int)(3 * T)}}, T)[0];
  return sum_partials(p, T, 0, 3);
}
mx::array e_rows_buffer(size_t entries) {
  mx::array a = run("fp_zero", "out[thread_position_in_grid.x] = 0;", {words(std::vector<blk::u64>(8, 0))}, {"unused_"}, {{(int)(3 * entries)}}, 3 * entries)[0];
  mx::eval(a);
  return a;
}
mx::array put_rows(const mx::array &o, const TableWords &t, size_t first) {
  return run("fp_put_rows", R"(
    ulong i = thread_position_in_grid.x, b = thread_position_in_grid.y, deg = q[0], len = q[1], first = q[2], at = (b * len + i) * deg;
    GlE x = deg == 1 ? GlE{a[at], 0, 0} : GlE{a[at], a[at + 1], a[at + 2]};
    e_st((device ulong *)o, (first + b) * len + i, x);)", {o, t.a.array(), words(std::vector<blk::u64>{t.deg, t.len, first})}, {"o", "a", "q"}, {{1}}, t.len, t.rows)[0];
}
Buf encode_table(const TableWords &m, const TableWords &tl, size_t message_at, size_t tail_at, size_t ell, size_t r, unsigned ld, unsigned log_n, const blk::F *tw, size_t first, size_t count) {
  const size_t block = size_t(1) << ld, rows = (count ? count : m.rows) * m.deg;
  const mx::array c = run("fp_coefficients", R"(
    ulong x = thread_position_in_grid.x, row = thread_position_in_grid.y, deg = q[0], ell = q[1], r = q[2], ma = q[3], ta = q[4], block = q[5], b = q[6] + row / deg, d = row % deg, v = 0;
    if (x >= ma && x < ma + ell) v = a[(b * ell + x - ma) * deg + d];
    else if (x >= ta && x < ta + r) v = tl[(b * r + x - ta) * deg + d];
    out[row * block + x] = v;)", {m.a.array(), tl.a.array(), words(std::vector<blk::u64>{m.deg, ell, r, message_at, tail_at, block, first, 0})}, {"a", "tl", "q"}, {{(int)(rows * block)}}, block, rows)[0];
  return encode_array(c, rows, ld, log_n, tw);
}
TableSide lookup_table_side(unsigned kT, unsigned cK, unsigned NC, const std::vector<blk::u64> &rows, const std::vector<blk::u32> &mult, const blk::E &alpha, const blk::E &theta) {
  const unsigned nT = 5 + kT + cK;
  const size_t n = size_t(1) << nT;
  std::vector<blk::u64> p = {cK, NC, kT, n};
  for (const blk::E &e : {alpha, theta}) for (blk::u64 c : e_words(e)) p.push_back(c);
  std::vector<blk::u64> m64(mult.begin(), mult.end());
  const mx::array bits_rows = run("fp_lookup_table", R"(
    ulong i = thread_position_in_grid.x, cK = p[0], NC = p[1], kT = p[2], n = p[3], j = i & 31, c = (i >> 5) & ((1ul << cK) - 1), t = i >> (5 + cK), cc = min(c, NC - 1);
    GlE theta = {p[7], p[8], p[9]}, keyed = e_scale(theta, cc);
    keyed.c0 = gl_add(keyed.c0, gl_canonical(rw[(cc << kT) + t]));
    e_st(out, i, GlE{(c < NC && ((mu[(c << kT) + t] >> j) & 1)) ? 1ul : 0ul, 0, 0});
    e_st(out, n + i, e_sub(e_sub(GlE{p[4], p[5], p[6]}, GlE{1, 0, 0}), keyed));)", {words(rows), words(m64), words(p)}, {"rw", "mu", "p"}, {{(int)(6 * n)}}, n)[0];
  const mx::array leaves = run("fp_lookup_table_leaves", "ulong i = thread_position_in_grid.x, n = p[0]; e_st(out, i, e_add(GlE{1, 0, 0}, e_mul(e_ld(a, i), e_ld(a, n + i))));",
                               {bits_rows, words(std::vector<blk::u64>{n})}, {"a", "p"}, {{(int)(3 * n)}}, n)[0];
  return {bits_rows, leaves};
}
void add_term(const Buf &o, const TableWords &t, const blk::E &scale, blk::u64 mask, blk::u64 value) {
  const size_t n = t.len << t.log_rows;
  const mx::array done = run("fp_add_term", R"(
    ulong i = thread_position_in_grid.x, deg = q[0], lr = q[1], rows = q[2], len = q[3], mask = q[4], value = q[5];
    ulong row = i & ((1ul << lr) - 1), place = i >> lr;
    if (row >= rows) return;
    ulong at = (row * len + place) * deg, j = value, rest = i;
    for (uint b = 0; rest != 0; b++) if (!((mask >> b) & 1)) { j |= (rest & 1) << b; rest >>= 1; }
    GlE x = deg == 1 ? GlE{a[at], 0, 0} : GlE{a[at], a[at + 1], a[at + 2]};
    device ulong *O = (device ulong *)o;
    e_st(O, j, e_add(e_ld(O, j), e_mul(GlE{q[6], q[7], q[8]}, x)));)",
    {o.array(), t.a.array(), words(std::vector<blk::u64>{t.deg, t.log_rows, t.rows, t.len, mask, value, scale.c[0].v, scale.c[1].v, scale.c[2].v})}, {"o", "a", "q"}, {{1}}, n)[0];
  mx::eval(done);
}
mx::array index_message(const int8_t *cells, size_t rows, size_t columns, size_t table_rows, size_t ell, size_t per, size_t plain, const std::vector<blk::u64> &pads, size_t pads_each) {
  const mx::array c(cells, {(int)(rows * columns)}, mx::int8);
  std::vector<blk::u64> pd = pads;
  pd.resize(std::max<size_t>(pd.size(), 8), 0);
  return run("fp_index_message", R"(
    ulong j = thread_position_in_grid.x, r = thread_position_in_grid.y, rows = q[0], columns = q[1], ell = q[2], per = q[3], plain = q[4], each = q[5], y = j / per, v = 0;
    if (y < rows) { long x = ce[y * columns + j % per + per * r]; v = x < 0 ? GL_P - (ulong)(-x) : (ulong)x; }
    else if (j >= plain && j < plain + each) v = pd[r * each + j - plain];
    out[r * ell + j] = v;)", {c, words(pd), words(std::vector<blk::u64>{rows, columns, ell, per, plain, pads_each, 0, 0})}, {"ce", "pd", "q"}, {{(int)(table_rows * ell)}}, ell, table_rows)[0];
}
std::vector<blk::E> rows_left(const mx::array &m, size_t rows, size_t len, const std::vector<blk::E> &w) {
  const mx::array o = run("fp_rows_left", R"(
    ulong j = thread_position_in_grid.x, rows = q[0], len = q[1]; GlE s = {0, 0, 0};
    for (ulong x = 0; x < rows; x++) s = e_add(s, e_scale(e_ld(w, x), m[x * len + j]));
    e_st(out, j, s);)", {m, e_array(w), words(std::vector<blk::u64>{rows, len, 0, 0, 0, 0, 0, 0})}, {"m", "w", "q"}, {{(int)(3 * len)}}, len)[0];
  return e_fetch(o);
}
bool lookup_counts(const std::vector<blk::u64> &keys, const std::vector<char> &padding, unsigned k, const std::vector<blk::u64> &values, const std::vector<blk::u32> &rows, blk::u32 *count) {
  std::vector<blk::u64> v = values;
  std::vector<uint32_t> r(rows.begin(), rows.end());
  const size_t tn = v.size();
  v.resize(std::max<size_t>(v.size(), 8), 0), r.resize(std::max<size_t>(r.size(), 16), 0);
  const size_t cn = size_t(1) << k;
  mx::array cnt(mx::allocator::malloc(4 * std::max<size_t>(cn, 16)), {(int)std::max<size_t>(cn, 16)}, mx::uint32);
  memset(cnt.data<uint32_t>(), 0, 4 * std::max<size_t>(cn, 16));
  const mx::array pad(reinterpret_cast<const uint8_t *>(padding.data()), {(int)padding.size()}, mx::uint8);
  mx::array flag(mx::allocator::malloc(64), {16}, mx::uint32);
  memset(flag.data<uint32_t>(), 0, 64);
  const mx::array done = run("fp_lookup_counts", R"(
    ulong i = thread_position_in_grid.x, tn = q[0], k = q[1], key = ks[i];
    if (pd[i]) return;
    uint row;
    if (tn == 0) { if (key >> k) { atomic_store_explicit((device atomic_uint *)fl, 1u, memory_order_relaxed); return; } row = (uint)key; }
    else {
      ulong lo = 0, hi = tn;
      while (lo < hi) { ulong mid = (lo + hi) / 2; if (tv[mid] < key) lo = mid + 1; else hi = mid; }
      if (lo == tn || tv[lo] != key) { atomic_store_explicit((device atomic_uint *)fl, 1u, memory_order_relaxed); return; }
      row = tr[lo];
    }
    atomic_fetch_add_explicit((device atomic_uint *)cnt + row, 1u, memory_order_relaxed);)",
    {words(keys), pad, words(v), mx::array(r.data(), {(int)r.size()}, mx::uint32), cnt, flag, words(std::vector<blk::u64>{tn, k, 0, 0, 0, 0, 0, 0})}, {"ks", "pd", "tv", "tr", "cnt", "fl", "q"}, {{1}}, keys.size())[0];
  mx::eval(done);
  memcpy(count, cnt.data<uint32_t>(), 4 * cn);
  return flag.data<uint32_t>()[0] == 0;
}
static const mx::array &matrix_on_gpu(const int8_t *matrix, size_t n);
mx::array wf_run(const mx::array &keys, const mx::array &pad, const int8_t *matrix, size_t cells, size_t columns, size_t first, size_t first_row, size_t places, size_t stride, blk::F offset, const std::vector<blk::F> &weights) {
  std::vector<blk::u64> p = {columns, first, first_row, stride, offset.v, weights.size(), 0, 0};
  for (const blk::F &w : weights) p.push_back(w.v);
  return run("fp_wf_run", R"(
    ulong c = thread_position_in_grid.x, i = thread_position_in_grid.y, columns = q[0], first = q[1], row0 = q[2], stride = q[3], n = q[5], key = q[4];
    for (ulong d = 0; d < n; d++) {
      long v = ce[(row0 + i + d * stride) * columns + c];
      key = gl_add(key, gl_mul(q[8 + d], v < 0 ? GL_P - (ulong)(-v) : (ulong)v));
    }
    ulong at = first + c + i * columns;
    ((device ulong *)ks)[at] = key, ((device uchar *)pd)[at] = 0;)", {keys, pad, matrix_on_gpu(matrix, cells), words(p)}, {"ks", "pd", "ce", "q"}, {{1}}, columns, places)[0];
}
mx::array lkp_leaves(const mx::array &keys, size_t first, unsigned v, unsigned g, const std::vector<blk::u64> &blocks, const std::vector<blk::E> &rows, const blk::E &alpha, const blk::E &theta) {
  std::vector<blk::u64> p = {first, g, rows.size(), 0};
  for (const blk::E &e : {alpha, theta}) for (blk::u64 c : e_words(e)) p.push_back(c);
  std::vector<blk::u64> b = blocks;
  b.resize(std::max<size_t>(b.size(), 8), 0);
  const size_t n = size_t(1) << v;
  return run("fp_lkp_leaves", R"(
    ulong i = thread_position_in_grid.x, first = q[0], g = q[1], nb = q[2], y = i & ((1ul << g) - 1);
    GlE alpha = {q[4], q[5], q[6]}, theta = {q[7], q[8], q[9]};
    for (ulong b = 0; b < nb; b++) {
      ulong lt = bl[3 * b + 1];
      if ((y >> lt) != (bl[3 * b] >> lt)) continue;
      ulong f = bl[3 * b + 2];
      GlE x;
      if (f == 0) x = e_ld(rw, b);
      else { x = e_scale(theta, f - 1); x.c0 = gl_add(x.c0, ks[first + i]); }
      e_st(out, i, e_sub(alpha, x));
      return;
    })", {keys, words(b), e_scalars(rows), words(p)}, {"ks", "bl", "rw", "q"}, {{(int)(3 * n)}}, n)[0];
}
mx::array lkp_pad(const mx::array &leaves, size_t at, unsigned step, size_t length, const blk::E &value) {
  return run("fp_lkp_pad", "ulong i = thread_position_in_grid.x; e_st((device ulong *)a, q[0] + (i << q[1]), GlE{q[2], q[3], q[4]});",
             {leaves, words(std::vector<blk::u64>{at, step, value.c[0].v, value.c[1].v, value.c[2].v, 0, 0, 0})}, {"a", "q"}, {{1}}, length)[0];
}
TableSide lkp_table_side(const mx::array &bits, const mx::array &table, size_t w, const blk::E &alpha) {
  const size_t n = bits.size() / 3;
  const mx::array R = run("fp_lkp_rows", "ulong i = thread_position_in_grid.x, w = q[0]; e_st(out, i, e_sub(e_sub(GlE{q[1], q[2], q[3]}, GlE{1, 0, 0}), e_ld(t, i / w)));",
                          {table, words(std::vector<blk::u64>{w, alpha.c[0].v, alpha.c[1].v, alpha.c[2].v, 0, 0, 0, 0})}, {"t", "q"}, {{(int)(3 * n)}}, n)[0];
  const mx::array L = run("fp_lkp_table_leaves", "ulong i = thread_position_in_grid.x; e_st(out, i, e_add(GlE{1, 0, 0}, e_mul(e_ld(b, i), e_ld(r, i))));", {bits, R}, {"b", "r"}, {{(int)(3 * n)}}, n)[0];
  return {R, L};
}
static const mx::array &matrix_on_gpu(const int8_t *matrix, size_t n) {
  static const int8_t *at = nullptr;
  static size_t size = 0;
  static std::unique_ptr<mx::array> kept;
  if (at != matrix || size != n) kept = std::make_unique<mx::array>(matrix, mx::Shape{(int)n}, mx::int8), at = matrix, size = n;
  return *kept;
}
Buf matrix_left(const int8_t *matrix, size_t cells, size_t first, size_t rows, size_t columns, const std::vector<blk::E> &g) {
  return Buf(run("fp_matrix_left",
                 "ulong x = thread_position_in_grid.x, columns = p[0], rows = p[1], first = p[2]; GlE s = {0, 0, 0};"
                 "for (ulong y = 0; y < rows; y++) { long v = ce[(first + y) * columns + x]; if (v) s = e_add(s, e_scale(e_ld(g, y), v < 0 ? GL_P - (ulong)(-v) : (ulong)v)); }"
                 "e_st(out, x, s);",
                 {matrix_on_gpu(matrix, cells), e_array(g), words(std::vector<blk::u64>{columns, rows, first})}, {"ce", "g", "p"}, {{(int)(3 * columns)}}, columns)[0]);
}
}
