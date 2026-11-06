// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Proves a request on MLX: the native counterpart of cp_prove. Its companion, a build of the verifier
// (mlx/DESIGN_PROVER.md), supplies each segment's equation; the final proof is for the stock cp_verify.
// usage: fp_prove <config directory> <ivk directory> <reference directory> <seed file> <request file> <y> <commitment id> <out proof> <companion>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <csignal>
#include <spawn.h>
#include <sstream>
#include <sys/stat.h>
#include "cp_configured.h"
#include "engine/sumcheck_cpu.h"
#include "engine_mlx.h"
#include "frame/check.h"
#include "frame/check_entry.h"
#include "frame/check_key.h"
#include "index_mlx.h"
#include "seed_coins.h"
#include "seg_ff_proof.h"
#include "seg_attn_proof.h"
#include "seg_head_proof.h"
using namespace blk;
using namespace fpmlx;
extern char **environ;
static std::string slurp(const std::string &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "fp_prove: cannot read %s\n", p.c_str()); exit(2); }
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
static volatile pid_t companion_group = 0;
static void stop_companion() { if (companion_group > 0) kill(-companion_group, SIGKILL); }
static void on_fatal_signal(int s) {
  stop_companion();
  signal(s, SIG_DFL);
  raise(s);
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char **argv) {
  if (argc != 10) { fprintf(stderr, "usage: fp_prove <config> <ivk> <reference> <seed> <request> <y> <id> <out proof> <companion>\n"); return 2; }
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::string cfg = argv[1], ivk = argv[2], ref = argv[3], seedf = argv[4], reqf = argv[5], id = argv[7], outp = argv[8], comp = argv[9];
  const std::string pol_t = slurp(cfg + "/policy.txt"), lay = slurp(cfg + "/layout.txt"), kl = slurp(cfg + "/KL.bin");
  std::vector<uint32_t> KL(kl.size() / 4);
  memcpy(KL.data(), kl.data(), kl.size());
  const std::string raw = slurp(ref + "/matrix.i8"), req = slurp(reqf);
  const ModelShape shape = read_shape(ref + "/params.txt");
  const Tensors W = [&] { auto l = mx::load_safetensors(ref + "/weights.safetensors"); Tensors t; for (auto &kv : l.first) t.emplace(kv.first, kv.second); return t; }();
  const Policy pol = Policy::read(pol_t);
  std::vector<u32> tokens(shape.T);
  memcpy(tokens.data(), req.data(), 4 * shape.T);
  std::vector<int32_t> tk(tokens.begin(), tokens.end());
  const mx::array tok_a(tk.data(), {(int)tk.size()}, mx::int32);
  const double t1 = now();
  const Values V = capture_forward(shape, W, tok_a);
  const CallTables tabs = call_tables(W);
  printf("forward: %.1f s\n", now() - t1);
  HeadRequest q0 = HeadRequest::read(req, shape.T);
  const u32 y = strcmp(argv[6], "auto") ? (u32)atoi(argv[6]) : head_next_token(V, q0, shape.T);
  printf("next token: %u\n", y);
  // The stream, the marker file and the companion.
  const std::string dir = outp + ".work";
  mkdir(dir.c_str(), 0700);
  const std::string spath = dir + "/stream.bin", mark = dir + "/mark.proof", yf = dir + "/y";
  unlink(spath.c_str()); // a new file: whatever an earlier run left behind can no longer reach this stream
  uint8_t *m = fpstream::map(spath.c_str(), true);
  if (!m) { fprintf(stderr, "fp_prove: no stream\n"); return 2; }
  { std::ofstream f(yf, std::ios::binary); f.write((const char *)&y, 4); }
  {
    Writer h;
    write_head(h, TAG_PROOF, 0);
    std::vector<u8> b = h.bytes();
    b.resize(fpstream::MARK_BYTES, 0);
    std::ofstream f(mark, std::ios::binary);
    f.write((const char *)b.data(), b.size());
  }
  setenv("FP_STREAM", spath.c_str(), 1);
  const std::string log = dir + "/companion.log";
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  posix_spawn_file_actions_adddup2(&fa, 1, 2);
  const char *cargs[] = {comp.c_str(), cfg.c_str(), ivk.c_str(), reqf.c_str(), yf.c_str(), mark.c_str(), "-id", id.c_str(), nullptr};
  pid_t pid = 0;
  // The companion and its children form a process group, which ends with this process, however it ends.
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&at, 0);
  if (posix_spawn(&pid, comp.c_str(), &fa, &at, (char *const *)cargs, environ)) { fprintf(stderr, "fp_prove: no companion\n"); return 2; }
  companion_group = pid;
  atexit(stop_companion);
  for (int s : {SIGABRT, SIGINT, SIGTERM, SIGSEGV, SIGBUS, SIGILL, SIGHUP}) signal(s, on_fatal_signal);
  return cp_with_configured(pol_t, lay, KL, 0, true, [&](const Configured &c) -> int {
    const double t0 = now();
    Companion io;
    io.m = m, io.pid = pid;
    const KeyFixed f = key_fixed(c);
    const LinShape &S = f.lin;
    const u64 columns = u64(1) << S.k_col();
    const int8_t *cells = (const int8_t *)raw.data();
    CellMatrix M{S.matrix_rows, columns, cells};
    const CoinKey seed = load_or_make_seed(seedf.c_str());
    auto leaf_coins = seeded_coins(seed, COINS_LEAF);
    // A request's masks must be fresh: two proofs under one mask would show differences of hidden values.
    auto coins = run_coins_without_a_seed<MetalCoins>(CoinInputs().add("a request", req).finish(), CoinContext{});
    // MLX keeps freed buffers for reuse, by default as much again as is in use.
    mx::set_cache_limit((size_t)(atof(getenv("FP_MLX_CACHE_GB") ? getenv("FP_MLX_CACHE_GB") : "4") * 1e9));
    EngineMlx engine;
    const NonceSearch search = [&](const ChainPrefix &p, unsigned bits) { return engine.search_nonce(p, bits); };
    IndexMlx table(S, M, *leaf_coins, c.configuration());
    table.keep_count_in(SpentLedger(seedf + ".spent"));
    printf("leaf table: %.1f s\n", now() - t0);
    const WeightInts wi(W);
    const Units units = Units::of(shape, S.k_col());
    // The statement, from the walk.
    const std::vector<u8> st = [&] {
      Writer none; io.file = &none; return io.take(fpstream::STATEMENT);
    }();
    RecordReader sr(st);
    const u64 na = sr.u();
    const std::vector<u8> request(sr.p, sr.p + na);
    sr.p += na;
    const u64 nb = sr.u();
    const std::vector<u8> file_words(sr.p, sr.p + nb);
    std::unique_ptr<LinBegun> begun_first;
    if (!lin_begin(table, S, begun_first)) { fprintf(stderr, "fp_prove: the table allows no further batch, or its count %s.spent is missing or unreadable\n", seedf.c_str()); return 3; }
    Writer file;
    io.file = &file;
    write_head(file, TAG_PROOF, 0);
    const IndexName name{table.commitment(), table.record()};
    Binder b(LABEL_FIRST_CHAIN, c.proof().chains.at(0).plan);
    RequestPublic x = bind_request_parameters(b, c);
    bind_request_statement(b, x, name, 0, request, file_words);
    u8 salt[DIGEST_BYTES];
    coins->bytes(salt, sizeof salt);
    Digest random_bytes;
    digest_from_bytes(salt, random_bytes);
    ProverRounds ch(b, random_bytes, file, search);
    ProverContext P(x.lin.codes, ch, engine, *coins);
    ReqProver frame(P, x.shape);
    ReqRows<ProverContext> rows(P, frame, x.layout);
    const ReqShape &sh = ch.get(x.shape);
    const ReqLayout &ly = ch.get(x.layout);
    Chan chan{ch};
    chan.next_public = [&] { const std::vector<u8> v = io.take(fpstream::PUBLIC); u64 w[3]; memcpy(w, v.data(), 24); return E{{F{w[0]}, F{w[1]}, F{w[2]}}}; };
    chan.next_rows = [&] { const std::vector<u8> v = io.take(fpstream::ROWS); u64 n; memcpy(&n, v.data(), 8); return n; };
    FFInput in{pol, shape, V, wi, units, cells, columns, tokens};
    HeldArrays held;
    const bool checking = !getenv("FP_NO_EQUATION_TEST");
    std::vector<MaskSlot> pend_slots;
    const int Lam = 1 << pol.get("ll"), NG = shape.L / Lam;
    auto hold = [&](size_t link, const Arrays &a) { if (checking) for (auto &kv : a) held[{link, kv.first}] = kv.second; };
    auto take_index_mask = [&] {
      pend_slots.push_back(P.st.masks(frame.at_start().at(0)).take());
      return P.mask_value(pend_slots.back());
    };
    auto named = [](const std::vector<ArrayData> &v) { Arrays a; for (size_t i = 0; i < v.size(); i++) a["M" + std::to_string(i)] = v[i]; return a; };
    for (int g = 0; g < NG; g++) {
      // The feed-forward segment.
      double ts = now();
      const size_t lg = 2 * g, la = lg + 1;
      FFGroup G = ff_build(pol, shape, V, tabs, g);
      FFProver pr(in, G, tabs);
      pr.prepare(g == 0);
      SegMask mask = make_mask(P, sh, lg);
      std::vector<OracleDraft> at_start;
      if (g == 0) at_start.push_back(lin_draft(P, sh.at_start[0].shape, false, sh.at_start[0].leaf_class, {}));
      hold(lg, pr.arrays);
      (void)frame.first(place_tables(P, ly, sh, lg, 0, pr.arrays), std::move(at_start), std::move(mask.table));
      const E cm = take_index_mask();
      u32 slot = 0;
      pr.before_u3(chan, io, mask.v.data(), slot, cm);
      pr.moment_u3(io);
      hold(lg, pr.arrays);
      (void)frame.later(place_tables(P, ly, sh, lg, 1, pr.arrays));
      if (!pr.after_u3(ch)) { fprintf(stderr, "fp_prove: a combination leaves its window\n"); return 3; }
      hold(lg, pr.arrays);
      (void)frame.later(place_tables(P, ly, sh, lg, 2, pr.arrays));
      pr.after_digits(chan, mask.v.data(), slot);
      close_segment(P, frame, rows, io, checking ? &held : nullptr, &mask.v);
      printf("G%d: %.1f s, %u scalars, %zu bytes so far\n", g, now() - ts, slot, file.size());
      // The attention segment.
      ts = now();
      AttnGroup A = attn_build(pol, shape, G, g);
      if (!A.keys_ok) { fprintf(stderr, "fp_prove: a key of attention is in no row\n"); return 3; }
      SegMask am = make_mask(P, sh, la);
      Arrays arrays = named(attn_arrays(A));
      hold(la, arrays);
      (void)frame.first(place_tables(P, ly, sh, la, 0, arrays), {}, std::move(am.table));
      Smallness sm;
      AttnRun R;
      u32 as = 0;
      attn_rounds(chan, am.v.data(), as, A, G, sm, R);
      arrays.clear();
      { ArrayData U; U.over_e = true, U.e = sm.u3(), U.rows = 1, U.columns = U.e.size(); arrays["U3"] = std::move(U); }
      hold(la, arrays);
      (void)frame.later(place_tables(P, ly, sh, la, 1, arrays));
      sm.draw_theta(ch);
      arrays.clear();
      if (!sm.digits(arrays["ZZ"], arrays["BZ"])) { fprintf(stderr, "fp_prove: a combination leaves its window\n"); return 3; }
      hold(la, arrays);
      (void)frame.later(place_tables(P, ly, sh, la, 2, arrays));
      sm.run_lookup(chan, am.v.data(), as);
      (void)chan.batch_open();
      close_segment(P, frame, rows, io, checking ? &held : nullptr, &am.v);
      (void)io.take(fpstream::FUNS); // the attention walk's, which this prover forms by itself
      printf("A%d: %.1f s, %u scalars, %zu bytes so far\n", g, now() - ts, as, file.size());
    }
    // The output head.
    {
      const double ts = now();
      const size_t lh = 2 * NG;
      HeadRequest q = HeadRequest::read(req, shape.T);
      q.y = y;
      HeadWitness H = head_build(pol, shape, V, tabs, q);
      if (!H.norm_ok || !H.margins_ok || !H.bounds_ok) { fprintf(stderr, "fp_prove: the head's witness fails: norm %d margins %d bounds %d\n", H.norm_ok, H.margins_ok, H.bounds_ok); return 3; }
      HeadProof hp(H, wi.at("Wh"));
      SegMask hm = make_mask(P, sh, lh);
      Arrays arrays = named(hp.arrays0());
      hold(lh, arrays);
      (void)frame.first(place_tables(P, ly, sh, lh, 0, arrays), {}, std::move(hm.table), {E::from(from_u64(y))});
      const E cm = take_index_mask();
      u32 hs = 0;
      hp.moment0(chan, hm.v.data(), hs);
      arrays.clear();
      { ArrayData U; U.over_e = true, U.e = hp.u3(), U.rows = 1, U.columns = U.e.size(); arrays["U3"] = std::move(U); }
      hold(lh, arrays);
      (void)frame.later(place_tables(P, ly, sh, lh, 1, arrays));
      hp.draw(ch);
      arrays.clear();
      if (!hp.digits(arrays["ZZ"], arrays["BZ"])) { fprintf(stderr, "fp_prove: a combination leaves its window\n"); return 3; }
      hold(lh, arrays);
      (void)frame.later(place_tables(P, ly, sh, lh, 2, arrays));
      hp.moment2(chan, hm.v.data(), hs, cm, 0);
      close_segment(P, frame, rows, io, checking ? &held : nullptr, &hm.v);
      printf("H: %.1f s, %u scalars, %zu bytes so far\n", now() - ts, hs, file.size());
    }
    frame.finish();
    const double te = now();
    // The rows on the table, as the walk has them.
    std::vector<TensorRow> pending;
    for (size_t k = 0; k < pend_slots.size(); k++) {
      const std::vector<u8> v = io.take(fpstream::PENDING);
      RecordReader r(v);
      const u64 first = r.u();
      auto words = [&](u64 n) { std::vector<E> o(n); for (E &e : o) { u64 w[3]; w[0] = r.u(), w[1] = r.u(), w[2] = r.u(); e = E{{F{w[0]}, F{w[1]}, F{w[2]}}}; } return o; };
      auto values = std::make_shared<const std::vector<E>>(words(r.u()));
      const std::vector<E> psi = words(r.u());
      const E beta = words(1)[0], target = words(1)[0];
      pending.push_back(member_pending_row(member_weights_on_rows(S, first, values), psi, std::move(pend_slots[k]), beta, target));
    }
    std::unique_ptr<LinBegun> &begun = begun_first;
    MemberMlx heavy(S, M);
    LinFirst lin_first = member_first_prove(P, x.lin, frame.at_start().at(0), std::move(pending), heavy);
    Sealed seal;
    BLK_REQUIRE(ch.seal(R_SEAL, seal), "a prover's channel does not refuse");
    ch.finish_inside_a_file();
    printf("first chain sealed: %zu bytes, %.1f s\n", file.size(), now() - t0);
    LinBinding binding(c.fixed().binding, c.fixed().configuration, *coins);
    binding.put(0, seal.digest());
    binding.write_path(file, 0);
    Binder shared_b(LABEL_PROOF_SHARED_CHAIN, c.proof().chains.at(1).plan);
    const LinPublic shared_x = bind_shared_chain(shared_b, c, name, binding.root());
    ProverRounds shared_ch(shared_b, Digest{}, file, search);
    ProverContext shared_P(shared_x.codes, shared_ch, engine, *coins);
    LinShared shared;
    lin_shared_prove(shared_P, shared_x, TableTest::LATER, table, std::move(begun), shared);
    lin_shared_tests_prove(shared_P, shared_x, shared);
    Sealed shared_seal;
    BLK_REQUIRE(shared_ch.seal(R_SEAL, shared_seal), "a prover's channel does not refuse");
    shared_ch.finish_inside_a_file();
    Binder second_b(LABEL_PROOF_SECOND_CHAIN, c.proof().chains.at(2).plan);
    bind_second_chain(second_b, c, seal, shared_seal);
    coins->bytes(salt, sizeof salt);
    digest_from_bytes(salt, random_bytes);
    ProverRounds second_ch(second_b, random_bytes, file, search);
    {
      ProverContext P2 = lin_go_on(std::move(P), second_ch);
      const LinSecond second = lin_second_prove(P2, x.lin, lin_first, shared_P, shared);
      lin_second_tests_prove(P2, x.lin, lin_first, second);
      second_ch.finish();
      P2.st.require_all_tested();
    }
    lin_release(shared_P, shared);
    printf("the end of the proof: %.1f s\n", now() - te);
    {
      std::ofstream o(outp, std::ios::binary);
      o.write((const char *)file.data(), file.size());
    }
    io.close_stream();
    printf("proof: %zu bytes, %.1f s in all\n", file.size(), now() - t0);
    printf("GPU memory: peak %.2f GB, active %.2f GB, cache %.2f GB\n", mx::get_peak_memory() / 1e9, mx::get_active_memory() / 1e9, mx::get_cache_memory() / 1e9);
    return 0;
  });
}
