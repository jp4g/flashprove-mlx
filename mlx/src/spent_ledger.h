// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// How many columns of a table proofs have opened, kept on disk. A table's random tails hide only so many
// opened columns (hiding_positions()), and the table is rebuilt by each run, so a count in memory would
// start again and let proofs pass the limit, which no verifier would see. The count is raised and flushed
// before any column is opened: a crash loses columns, never hides a spent one. A file that is missing or
// unreadable refuses the batch: starting again from 0 is the fault that this file is there to prevent.
#pragma once
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <sys/file.h>
#include <unistd.h>
namespace fpmlx {
class SpentLedger {
public:
  explicit SpentLedger(std::string path) : path_(std::move(path)) {}
  // fp_commit: a new ledger at 0. One that exists stays as it is: the seed's table is the same table.
  bool create() const {
    const int fd = ::open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return errno == EEXIST;
    const bool ok = put(fd, 0);
    close(fd);
    return ok;
  }
  // Adds `columns` if the sum stays within `limit`. False: it does not, or the count cannot be read or kept.
  bool spend(unsigned long long columns, unsigned long long limit, unsigned long long *now = nullptr) const {
    const int fd = ::open(path_.c_str(), O_RDWR);
    if (fd < 0) return false;
    bool ok = false;
    if (!flock(fd, LOCK_EX)) { // two provers on one table must not read the same count
      unsigned long long spent = 0;
      if (get(fd, spent) && spent <= limit && columns <= limit - spent && put(fd, spent + columns)) {
        ok = true;
        if (now) *now = spent + columns;
      }
    }
    close(fd); // releases the lock
    return ok;
  }
  bool read(unsigned long long &spent) const {
    const int fd = ::open(path_.c_str(), O_RDONLY);
    if (fd < 0) return false;
    const bool ok = get(fd, spent);
    close(fd);
    return ok;
  }
private:
  static constexpr size_t WIDTH = 21; // twenty digits and a newline: one write of one length
  static bool get(int fd, unsigned long long &v) {
    char b[WIDTH + 1] = {};
    if (pread(fd, b, sizeof b, 0) != (ssize_t)WIDTH || b[WIDTH - 1] != '\n') return false;
    for (size_t i = 0; i + 1 < WIDTH; i++) if (b[i] < '0' || b[i] > '9') return false;
    v = strtoull(b, nullptr, 10);
    return true;
  }
  static bool put(int fd, unsigned long long v) {
    char b[WIDTH + 1];
    snprintf(b, sizeof b, "%020llu\n", v);
    if (pwrite(fd, b, WIDTH, 0) != (ssize_t)WIDTH) return false;
    return fcntl(fd, F_FULLFSYNC) == 0 || fsync(fd) == 0;
  }
  std::string path_;
};
}
