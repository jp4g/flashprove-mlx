// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The memory that fp_prove shares with its companion, a build of the verifier that walks a proof while
// it is being written (mlx/DESIGN_PROVER.md). One file, mapped by both.
//   header: written (bytes of proof available), closed (no more will come), eq_bytes, eq_count
//   proof bytes at PROOF_AT; equations, serialized, at EQ_AT.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
namespace fpstream {
constexpr uint64_t PROOF_AT = 4096, EQ_AT = 1ull << 30, SIZE = 1ull << 33;
// A proof file of this length is the companion's sign that the proof comes through the stream.
constexpr uint64_t MARK_BYTES = 1048583;
struct Head { std::atomic<uint64_t> written, closed, eq_bytes, eq_count; };
inline uint8_t *map(const char *path, bool create) {
  const int fd = open(path, create ? O_RDWR | O_CREAT | O_TRUNC : O_RDWR, 0600);
  if (fd < 0) return nullptr;
  if (create && ftruncate(fd, (off_t)SIZE)) return nullptr;
  void *p = mmap(nullptr, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  return p == MAP_FAILED ? nullptr : (uint8_t *)p;
}
inline Head *head(uint8_t *m) { return reinterpret_cast<Head *>(m); }
// The log: records [type, bytes, payload], appended by the companion, read in order by fp_prove.
enum Type : uint64_t { EQUATION = 1, PUBLIC = 2, FUNS = 3, PENDING = 4, CLAIMS = 5, ROWS = 6, STATEMENT = 7 };
inline uint8_t *companion_map();
inline uint8_t *log_begin(uint8_t *m) { return m + EQ_AT + head(m)->eq_bytes.load(); }
inline void log_end(uint8_t *m, uint8_t *start, uint8_t *end, uint64_t type) {
  // start points at 16 free bytes before the payload.
  const uint64_t n = (uint64_t)(end - start) - 16;
  memcpy(start, &type, 8), memcpy(start + 8, &n, 8);
  head(m)->eq_bytes.fetch_add((uint64_t)(end - start));
  head(m)->eq_count.fetch_add(1);
}
inline uint8_t *companion_map() {
  static uint8_t *m = [] { const char *e = getenv("FP_STREAM"); return e ? map(e, false) : nullptr; }();
  return m;
}
}
