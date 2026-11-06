// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The count of a table's opened columns: it survives its objects, stops at the limit, is not reset by a
// second create(), refuses when missing or damaged, and loses no column to concurrent provers.
#include <cstring>
#include <sys/wait.h>
#include <vector>
#include "spent_ledger.h"
using namespace fpmlx;
static int bad = 0;
static void check(bool ok, const char *what) { printf("%s: %s\n", ok ? "ok" : "FAILED", what); bad += !ok; }
int main() {
  char dir[] = "/tmp/ledger_testXXXXXX";
  if (!mkdtemp(dir)) return 2;
  const std::string p = std::string(dir) + "/seed.spent";
  unsigned long long v = 99;
  check(!SpentLedger(p).spend(1, 10), "a missing file refuses");
  check(SpentLedger(p).create() && SpentLedger(p).read(v) && v == 0, "a new ledger is at 0");
  check(SpentLedger(p).spend(4, 10), "a batch within the limit");
  check(SpentLedger(p).read(v) && v == 4, "the count survives its object");
  check(SpentLedger(p).create() && SpentLedger(p).read(v) && v == 4, "a second create() resets nothing");
  check(SpentLedger(p).spend(6, 10), "a batch up to the limit");
  check(!SpentLedger(p).spend(1, 10), "a batch past the limit is refused");
  check(SpentLedger(p).read(v) && v == 10, "a refused batch counts nothing");
  check(!SpentLedger(p).spend(~0ull, 10), "a huge batch does not wrap");
  { FILE *f = fopen(p.c_str(), "r+"); fputs("garbage", f); fclose(f); }
  check(!SpentLedger(p).spend(1, 1000), "a damaged file refuses");
  { FILE *f = fopen(p.c_str(), "w"); fclose(f); }
  check(!SpentLedger(p).spend(1, 1000), "an empty file refuses");
  const std::string q = std::string(dir) + "/many.spent";
  SpentLedger(q).create();
  std::vector<pid_t> kids;
  for (int k = 0; k < 8; k++) {
    const pid_t c = fork();
    if (!c) { int n = 0; for (int i = 0; i < 50; i++) n += SpentLedger(q).spend(1, 300); _exit(n); }
    kids.push_back(c);
  }
  int won = 0;
  for (pid_t c : kids) { int s = 0; waitpid(c, &s, 0); won += WEXITSTATUS(s); }
  check(won == 300 && SpentLedger(q).read(v) && v == 300, "eight provers at once: exactly the limit is granted");
  remove(p.c_str()), remove(q.c_str()), rmdir(dir);
  printf("%s\n", bad ? "ledger_test: FAILED" : "ledger_test: all hold");
  return bad != 0;
}
