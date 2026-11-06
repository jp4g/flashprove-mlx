// EngineMlx against EngineCpu, with equal coins: every root, opening, evaluation and fold must be equal.
#include <cstdio>
#include "engine/engine_cpu.h"
#include "engine_mlx.h"
#include "toy/toy_coins.h"
using namespace blk;
#define CHECK(c) do { if (!(c)) { std::printf("FAILED line %d: %s\n", __LINE__, #c); return 1; } } while (0)
static Table random_table(Coins &coins, unsigned deg, size_t rows, size_t len) { Table t(deg, rows, len); coins.fill(t.a.data(), t.a.size()); return t; }
static bool same(const Digest &a, const Digest &b) { return a == b; }
template <class Eng> static int run(Eng &e, Digest &root, Opening &op, E q[6], E &ev, Table &folded, Table &tail, Digest start) {
  ToyCoins coins(7);
  const Code code{10, 128, 37}, mask_code{7, 16, 11};
  const Table a = random_table(coins, 1, 3, code.ell), b = random_table(coins, 3, 1, code.ell);
  const Table mk = random_table(coins, 3, 2, mask_code.ell), mt = random_table(coins, 3, 2, mask_code.r);
  const TableId ta = e.put(a), tb = e.put(b);
  OraclePart pa, pb;
  pa.main = ta, pa.code = code, pa.mask = mk, pa.mask_tail = mt, pa.mask_code = mask_code;
  pb.main = tb, pb.code = code;
  OracleTreeShape s;
  s.log_n = 10, s.log_n_zk = 7, s.main_symbols = {3, 3}, s.mask_symbols = {6, 0}, s.tree_number = 5, s.salt_bytes = 40, s.run = start, s.parameters = start;
  const CommitId c = e.commit({pa, pb}, s, coins, root);
  std::vector<u64> at = {0, 3, 4, 700, 1023};
  e.open(c, PositionList::from_draws(at, 10), op);
  std::vector<E> z(9);
  for (E &x : z) x = coins.e();
  ev = e.evaluate(ta, Tensor::eq(z.data(), 9, coins.e()));
  Form f;
  std::vector<E> zz(8);
  for (E &x : zz) x = coins.e();
  f.tensors.push_back(Tensor::eq(zz.data(), 8, coins.e()));
  std::vector<FoldPart> parts = {FoldPart{c, 0, 0}, FoldPart{c, 1, 4}};
  (void)parts;
  const std::vector<FoldPart> one = {FoldPart{c, 0, 0}};
  std::vector<E> z9(9);
  for (E &x : z9) x = coins.e();
  Form g;
  g.tensors.push_back(Tensor::eq(z9.data(), 9, coins.e()));
  const FoldId fd = e.fold_begin(one, 2, g);
  e.fold_round(fd, q);
  e.fold_bind(fd, coins.e());
  e.fold_round(fd, q + 3);
  e.fold_bind(fd, coins.e());
  const TableId r = e.fold_end(fd, tail);
  folded = e.get(r);
  return 0;
}
int main() {
  Digest start{};
  for (int i = 0; i < 8; i++) start.w[i] = 0x12345 + i;
  EngineCpu ec; EngineMlx em;
  Digest r1, r2; Opening o1, o2; E q1[6], q2[6], e1, e2; Table f1, f2, t1, t2;
  CHECK(!run(ec, r1, o1, q1, e1, f1, t1, start));
  CHECK(!run(em, r2, o2, q2, e2, f2, t2, start));
  CHECK(same(r1, r2));
  CHECK(o1.columns.size() == o2.columns.size());
  for (size_t i = 0; i < o1.columns.size(); i++) CHECK(o1.columns[i] == o2.columns[i]);
  CHECK(o1.salts == o2.salts);
  CHECK(o1.given.size() == o2.given.size());
  for (size_t i = 0; i < o1.given.size(); i++) CHECK(o1.given[i] == o2.given[i]);
  CHECK(e1 == e2);
  for (int i = 0; i < 6; i++) CHECK(q1[i] == q2[i]);
  CHECK(f1.a.size() == f2.a.size());
  for (size_t i = 0; i < f1.a.size(); i++) CHECK(f1.a[i] == f2.a[i]);
  for (size_t i = 0; i < t1.a.size(); i++) CHECK(t1.a[i] == t2.a[i]);
  printf("engine_test: EngineMlx equals EngineCpu: root, %zu opened symbols, %zu digests, evaluation, 2 fold rounds, folded rows and tail\n", o1.columns.size(), o1.given.size());
}
