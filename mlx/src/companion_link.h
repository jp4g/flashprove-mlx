// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// fp_prove's end of the shared stream: proof bytes go out as they are written, and what the verifier's
// walk computes comes back as typed records (stream.h, mlx/DESIGN_PROVER.md).
#pragma once
#include <deque>
#include <map>
#include <stdexcept>
#include <sys/wait.h>
#include "iop/message.h"
#include "iop/req_rows.h"
#include "stream.h"
namespace fpmlx {
using namespace blk;
class Companion {
public:
  uint8_t *m = nullptr;
  const Writer *file = nullptr;
  pid_t pid = 0;
  void flush() {
    const u64 n = file->size();
    if (n > sent_) memcpy(m + fpstream::PROOF_AT + sent_, file->data() + sent_, n - sent_), sent_ = n, fpstream::head(m)->written.store(n);
  }
  void close_stream() { flush(), fpstream::head(m)->closed.store(1); }
  std::vector<u8> take(u64 type) {
    for (;;) {
      flush();
      auto &q = kept_[type];
      if (!q.empty()) { std::vector<u8> v = std::move(q.front()); q.pop_front(); return v; }
      if (at_ < fpstream::head(m)->eq_bytes.load()) { read_one(); continue; }
      int status = 0;
      if (pid && !dead_ && waitpid(pid, &status, WNOHANG) == pid) dead_ = true;
      if (dead_ && at_ >= fpstream::head(m)->eq_bytes.load())
        throw std::runtime_error("the companion ended while the prover waited for a record of type " + std::to_string(type));
      usleep(100);
    }
  }
  E take_e(u64 type) { const std::vector<u8> v = take(type); E x; memcpy(&x, v.data(), sizeof x); return x; }
private:
  void read_one() {
    const uint8_t *p = m + fpstream::EQ_AT + at_;
    u64 type, n;
    memcpy(&type, p, 8), memcpy(&n, p + 8, 8);
    kept_[type].emplace_back(p + 16, p + 16 + n);
    at_ += 16 + n;
  }
  u64 sent_ = 0, at_ = 0;
  bool dead_ = false;
  std::map<u64, std::deque<std::vector<u8>>> kept_;
};
// Reads a log record's words.
struct RecordReader {
  const u8 *p;
  explicit RecordReader(const std::vector<u8> &v) : p(v.data()) {}
  u64 u() { u64 x; memcpy(&x, p, 8); p += 8; return x; }
  E e() { E x; memcpy(&x, p, sizeof x); p += sizeof x; return x; }
  std::vector<E> es() { std::vector<E> v(u()); for (E &x : v) x = e(); return v; }
};
inline IndexWeights read_iw(RecordReader &r) {
  IndexWeights w;
  w.lo = r.es(), w.hi = r.es();
  if (r.u()) w.values = std::make_shared<const std::vector<E>>(r.es());
  return w;
}
inline ReqEquation read_equation(const std::vector<u8> &bytes) {
  RecordReader r(bytes);
  ReqEquation q;
  q.terms.resize(r.u());
  for (ReqTerm &t : q.terms) {
    t.born = r.u();
    const u64 n = r.u();
    t.array.assign((const char *)r.p, n), r.p += n;
    t.scale = r.e(), t.on_rows = read_iw(r), t.on_columns = read_iw(r);
  }
  q.on_masks = r.es();
  q.target = r.e();
  return q;
}
}
