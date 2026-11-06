#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Writes the two headers that make a companion of the verifier: copies of blocks/iop/message.h and
blocks/iop/req_rows.h with small insertions. The sources in the repository stay as they are, and
the stock cp_verify is built without these. usage: make_shadow.py <repository> <out directory>"""
import os, sys
root, out = sys.argv[1], sys.argv[2]
os.makedirs(out + "/iop", exist_ok=True)
s = open(root + "/blocks/iop/message.h").read()
old = """  Reader(const u8 *p, size_t n) : p_(p), end_(p + n) {}
"""
new = """  Reader(const u8 *p, size_t n) : p_(p), end_(p + n) {
    if (n == fpstream::MARK_BYTES && fpstream::companion_map()) { // the proof comes through the stream
      stream_ = fpstream::companion_map();
      p_ = stream_ + fpstream::PROOF_AT, end_ = p_;
    }
  }
"""
assert old in s; s = s.replace(old, new)
old = """  bool at_end() const { return ok_ && p_ == end_; }
"""
new = """  bool at_end() const {
    if (stream_) {
      fpstream::Head *h = fpstream::head(stream_);
      while (!h->closed.load() && (u64)(p_ - stream_ - fpstream::PROOF_AT) == h->written.load()) usleep(200);
      return ok_ && h->closed.load() && (u64)(p_ - stream_ - fpstream::PROOF_AT) == h->written.load();
    }
    return ok_ && p_ == end_;
  }
"""
assert old in s; s = s.replace(old, new)
old = """  const u8 *take(size_t n) {
    if (!ok_ || (size_t)(end_ - p_) < n) {"""
new = """  const u8 *take(size_t n) {
    if (stream_) {
      fpstream::Head *h = fpstream::head(stream_);
      const u64 want = (u64)(p_ - stream_ - fpstream::PROOF_AT) + n;
      while (h->written.load() < want && !h->closed.load()) usleep(200);
      end_ = stream_ + fpstream::PROOF_AT + h->written.load();
    }
    if (!ok_ || (size_t)(end_ - p_) < n) {"""
assert old in s; s = s.replace(old, new)
old = """  const u8 *p_, *end_;
  bool ok_ = true;
};"""
new = """  const u8 *p_, *end_;
  bool ok_ = true;
  u8 *stream_ = nullptr;
};"""
assert old in s; s = s.replace(old, new)
s = s.replace("namespace blk {", '#include <unistd.h>\n#include "stream.h"\nnamespace blk {', 1)
open(out + "/iop/message.h", "w").write(s)

s = open(root + "/blocks/iop/req_rows.h").read()
old = "  Row row(const ReqEquation &eq) const {\n"
new = """  Row row(const ReqEquation &eq) const {
    fp_export_equation(eq);
"""
assert old in s; s = s.replace(old, new)
old = "// Builds a segment's linear claim."
new = '''// The companion's export of a segment's equation to fp_prove.
inline void fp_put(u8 *&w, const void *p, size_t n) { memcpy(w, p, n), w += n; }
inline void fp_put_u64(u8 *&w, u64 x) { fp_put(w, &x, 8); }
inline void fp_put_es(u8 *&w, const std::vector<E> &v) { fp_put_u64(w, v.size()); if (!v.empty()) fp_put(w, v.data(), v.size() * sizeof(E)); }
inline void fp_put_iw(u8 *&w, const IndexWeights &x) {
  fp_put_es(w, x.lo), fp_put_es(w, x.hi);
  fp_put_u64(w, x.values ? 1 : 0);
  if (x.values) fp_put_es(w, *x.values);
}
inline void fp_export_equation(const ReqEquation &eq) {
  u8 *m = fpstream::companion_map();
  if (!m) return;
  u8 *start = fpstream::log_begin(m), *w = start + 16;
  fp_put_u64(w, eq.terms.size());
  for (const ReqTerm &t : eq.terms) {
    fp_put_u64(w, t.born), fp_put_u64(w, t.array.size()), fp_put(w, t.array.data(), t.array.size());
    fp_put(w, &t.scale, sizeof(E));
    fp_put_iw(w, t.on_rows), fp_put_iw(w, t.on_columns);
  }
  fp_put_es(w, eq.on_masks);
  fp_put(w, &eq.target, sizeof(E));
  fpstream::log_end(m, start, w, fpstream::EQUATION);
}
// Builds a segment's linear claim.'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace('#pragma once\n', '#pragma once\n#include <cstring>\n#include "stream.h"\n', 1)
open(out + "/iop/req_rows.h", "w").write(s)

# The verifier's sources, copied, with hooks that export what it computes.
import shutil
v = out + "/vsrc"
if os.path.exists(v): shutil.rmtree(v)
shutil.copytree(root + "/verifier/src", v)
def patch(name, old, new):
    p = v + "/" + name
    t = open(p).read()
    assert old in t, (name, old)
    open(p, "w").write(t.replace(old, new, 1))
hooks = """#include "stream.h"
static inline void fp_log_bytes(uint64_t type, const void *p, size_t n) {
  uint8_t *m = fpstream::companion_map();
  if (!m) return;
  uint8_t *s = fpstream::log_begin(m);
  memcpy(s + 16, p, n);
  fpstream::log_end(m, s, s + 16 + n, type);
}
"""
patch("vf_sumcheck_psc.h", '#include "iop/hp.h"\n', '#include "iop/hp.h"\n' + hooks)
patch("vf_sumcheck_psc.h", "  Psc p;\n  p.target = target;\n", """  Psc p;
  p.target = target;
  { const blk::u64 w[3] = {eC(target, 0), eC(target, 1), eC(target, 2)}; fp_log_bytes(fpstream::PUBLIC, w, sizeof w); }
""")
# U3's parts: matrix, run length, weights.
funs = """{ std::vector<blk::u64> o; o.push_back(fun.size());
    for (auto &x_ : fun) { o.push_back((blk::u64)x_.M); o.push_back(x_.L); o.push_back(x_.at); o.push_back(x_.phi.size());
      for (auto &e_ : x_.phi) for (int k_ = 0; k_ < 3; k_++) o.push_back(eC(e_, k_)); }
    fp_log_bytes(fpstream::FUNS, o.data(), o.size() * 8); }
"""
patch("vf_layers_check.inc", "  rootInH(F, tr, {(int)iU3});\n  tl.closeFuns(fun);\n", funs + "  rootInH(F, tr, {(int)iU3});\n  tl.closeFuns(fun);\n")
patch("vf_output_head_check.inc", "  tl.closeFuns(fun);\n", funs + "  tl.closeFuns(fun);\n")
patch("vf_attention_check.h", "    seam.tl.closeFuns(fun);\n", funs + "    seam.tl.closeFuns(fun);\n")
# The pending row: rows' weights from `first`, the point, beta, the target.
patch("vf_weights_pending_claim.h", "  p.g = blk::member_weights_on_rows(S, first, std::move(values));\n", """  const std::vector<blk::E> fp_values = *values;
  p.g = blk::member_weights_on_rows(S, first, std::move(values));
""")
patch("vf_weights_pending_claim.h", "  p.beta = toBlkE(o.beta), p.target = toBlkE(o.fvp);\n", """  p.beta = toBlkE(o.beta), p.target = toBlkE(o.fvp);
  { std::vector<blk::u64> b; b.push_back(first); b.push_back(fp_values.size());
    for (auto &e_ : fp_values) for (int k_ = 0; k_ < 3; k_++) b.push_back(e_.c[k_].v);
    b.push_back(p.psi.size());
    for (auto &e_ : p.psi) for (int k_ = 0; k_ < 3; k_++) b.push_back(e_.c[k_].v);
    for (int k_ = 0; k_ < 3; k_++) b.push_back(p.beta.c[k_].v);
    for (int k_ = 0; k_ < 3; k_++) b.push_back(p.target.c[k_].v);
    fp_log_bytes(fpstream::PENDING, b.data(), b.size() * 8); }
""")

patch("vf_weights_layout.h", "  vector<Ex> rho;\n  WsForkOut o;\n", """  { std::vector<blk::u64> b; b.push_back(cl.size());
    auto ve = [&](const vector<Ex> &v) { b.push_back(v.size()); for (auto &e_ : v) for (int k_ = 0; k_ < 3; k_++) b.push_back(eC(e_, k_)); };
    for (auto &c : cl) { b.push_back(c.ui); b.push_back(c.f.size()); for (auto &f_ : c.f) ve(f_.HV), ve(f_.pt), ve(f_.AV); }
    fp_log_bytes(fpstream::CLAIMS, b.data(), b.size() * 8); }
  vector<Ex> rho;
  WsForkOut o;
""")

patch("vf_transcript_calls.h", "static void trBatchRows(Tr &tr, u64 rows) { chainBatchRows(tr.one, rows); }", """static void trBatchRows(Tr &tr, u64 rows) { { uint64_t r_ = rows; fp_log_bytes(fpstream::ROWS, &r_, 8); } chainBatchRows(tr.one, rows); }""")
patch("vf_transcript_calls.h", '#pragma once\n', '#pragma once\n#include <cstring>\n' + hooks)
patch("vf_sumcheck_psc.h", '#include "iop/hp.h"\n' + hooks, '#include "iop/hp.h"\n')

# The first query's words, before the proof is read.
patch("cp_verify.cpp", "    const blk::IfTested verdict = blk::check_proof_if_tested(", """    { std::vector<uint8_t> b; auto put_ = [&](const void *p_, size_t n_) { const uint64_t m_ = n_; b.insert(b.end(), (const uint8_t *)&m_, (const uint8_t *)&m_ + 8); b.insert(b.end(), (const uint8_t *)p_, (const uint8_t *)p_ + n_); };
      const std::vector<blk::u8> a_(requestH.begin(), requestH.end()), c_(fileWordsH.begin(), fileWordsH.end());
      put_(a_.data(), a_.size()), put_(c_.data(), c_.size());
      fp_log_bytes(fpstream::STATEMENT, b.data(), b.size()); }
    const blk::IfTested verdict = blk::check_proof_if_tested(""")
