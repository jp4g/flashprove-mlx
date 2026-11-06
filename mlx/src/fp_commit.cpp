// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Commits to a model's weights on MLX: the native counterpart of cp_commit. It writes ivk/key.bin and
// ivk/leaf/0.bin, which the unmodified cp_verify checks, and prints the commitment ID.
// usage: fp_commit <config directory> <matrix.i8> <out directory> <seed file>
// The seed file is secret: it re-derives the tables' random entries, so that fp_prove can rebuild the
// leaf table. It is made if it does not exist, readable by its owner alone, and it must not leave them.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include "cp_configured.h"
#include "coins/run_coins.h"
#include "lkp_mlx.h"
#include "engine/sumcheck_cpu.h"
#include "engine_mlx.h"
#include "frame/check_entry.h"
#include "frame/check_key.h"
#include "index_mlx.h"
#include "seed_coins.h"
using namespace blk;
static std::string slurp(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "fp_commit: cannot read %s\n", p.c_str()); exit(2); }
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void spit(const std::string &p, const std::vector<u8> &b) {
  std::ofstream f(p, std::ios::binary);
  f.write((const char *)b.data(), b.size());
}
int main(int argc, char **argv) {
  if (argc != 5) { fprintf(stderr, "usage: fp_commit <config directory> <matrix.i8> <out directory> <seed file>\n"); return 2; }
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string cfg = argv[1], out = argv[3];
  const std::string pol = slurp(cfg + "/policy.txt"), lay = slurp(cfg + "/layout.txt"), kl = slurp(cfg + "/KL.bin");
  std::vector<uint32_t> KL(kl.size() / 4);
  memcpy(KL.data(), kl.data(), kl.size());
  const std::string raw = slurp(argv[2]);
  return cp_with_configured(pol, lay, KL, 0, true, [&](const Configured &c) -> int {
    const double t0 = now();
    const KeyFixed f = key_fixed(c);
    const LinShape &S = f.lin;
    const IndexShape &I = S.fold.index;
    const u64 columns = u64(1) << S.k_col();
    printf("table: 2^%u rows, %zu message entries, tail %zu, 2^%u positions; matrix %llu x %llu\n", I.v, I.code.ell, I.code.r, I.code.log_n, (unsigned long long)S.matrix_rows, (unsigned long long)columns);
    if (raw.size() != S.matrix_rows * columns) { fprintf(stderr, "fp_commit: the matrix has %zu cells, not %llu\n", raw.size(), (unsigned long long)(S.matrix_rows * columns)); return 2; }
    const int8_t *cells = (const int8_t *)raw.data();
    fpmlx::CellMatrix M{S.matrix_rows, columns, cells};
    const Digest inputs = CoinInputs().add("the public file", pol).finish();
    const CoinKey seed = fpmlx::load_or_make_seed(argv[4]);
    auto coins = fpmlx::seeded_coins(seed, fpmlx::COINS_ANCHOR), key_coins = fpmlx::seeded_coins(seed, fpmlx::COINS_KEY);
    auto leaf_coins = fpmlx::seeded_coins(seed, fpmlx::COINS_LEAF), link_coins = fpmlx::seeded_coins(seed, fpmlx::COINS_LINK);
    fpmlx::mx::set_cache_limit((size_t)4e9); // MLX keeps freed buffers for reuse, by default as much again as is in use
    EngineMlx engine;
    const NonceSearch search = [&](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); };
    if (getenv("FP_STOP_AFTER_TABLE")) { fpmlx::IndexMlx t(S, M, *coins, c.configuration()); printf("table only: %.1f s\n", now() - t0); return 0; }
    fpmlx::IndexMlx first(S, M, *coins, c.configuration());
    printf("anchor table: %.1f s\n", now() - t0); fflush(stdout);
    fpmlx::WfMlx ranges(f.ranges, wf_matrix_of(S), [&](u64 row, u64 column) { return from_i64(cells[row * columns + column]); });
    fpmlx::MemberMlx products(S, M);
    Writer key;
    const double t1 = now();
    if (!make_key(c, first, ranges, products, engine, *key_coins, search, key)) { fprintf(stderr, "fp_commit: make_key failed: a cell is out of its range\n"); return 3; }
    printf("key file: %zu bytes, %.1f s\n", key.bytes().size(), now() - t1); fflush(stdout);
    const double t2 = now();
    fpmlx::IndexMlx further(S, M, *leaf_coins, c.configuration());
    const fpmlx::SpentLedger ledger(std::string(argv[4]) + ".spent");
    if (!ledger.create()) { fprintf(stderr, "fp_commit: cannot make %s.spent\n", argv[4]); return 2; }
    further.keep_count_in(ledger);
    Writer entry;
    if (!make_entry(c, first, further, engine, *link_coins, search, entry)) { fprintf(stderr, "fp_commit: make_entry failed\n"); return 3; }
    printf("leaf file: %zu bytes, %.1f s\n", entry.bytes().size(), now() - t2);
    mkdir(out.c_str(), 0700);
    mkdir((out + "/leaf").c_str(), 0700);
    spit(out + "/key.bin", key.bytes());
    spit(out + "/leaf/0.bin", entry.bytes());
    const IndexRecord rec = first.record();
    const Digest id = published_id(c.configuration(), first.name(), record_seeds(I, first.name()), rec.y);
    u8 b[DIGEST_BYTES];
    digest_to_bytes(id, b);
    printf("commitment ID: ");
    for (u8 x : b) printf("%02x", x);
    printf("\n");
    return 0;
  });
}
