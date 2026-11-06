// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// A committed table on MLX: IndexCpu (blocks/iop/index_cpu.h) with the rows encoded and the tree
// hashed on the GPU, and `MemberHeavy` on the matrix of signed bytes.
#pragma once
#include <chrono>
#include <thread>
#include "engine/sumcheck_cpu.h"
#include "iop/foldeq.h"
#include "iop/lin.h"
#include "iop/member.h"
#include "mlx_kernels.h"
#include "spent_ledger.h"
namespace fpmlx {
using namespace blk;
struct CellMatrix { u64 rows, columns; const int8_t *cells; };
template <class Body> inline void in_parallel(u64 n, Body body) {
  const unsigned T = std::min<u64>(std::max(1u, std::thread::hardware_concurrency()), n ? n : 1);
  std::vector<std::thread> w;
  for (unsigned t = 0; t < T; t++) w.emplace_back([=] { for (u64 i = t; i < n; i += T) body(i); });
  for (auto &x : w) x.join();
}
class IndexMlx : public IndexHolder {
public:
  // Draws as IndexCpu: tails, pads, the tree's salts.
  IndexMlx(const LinShape &S, const CellMatrix &M, Coins &coins, const Digest &configuration) : shape_(S.fold.index), configuration_(configuration) {
    const Code &c = shape_.code;
    const u64 rows = shape_.rows(), per = u64(1) << S.k_lo, block = u64(1) << c.log_dim();
    std::vector<F> tails(rows * c.r);
    coins.fill(tails.data(), tails.size());
    std::vector<u64> tail_words(tails.size());
    for (u64 i = 0; i < tails.size(); i++) tail_words[i] = tails[i].v;
    const size_t pn = shape_.pads();
    std::vector<u64> pad_words(rows * pn);
    { std::vector<F> pads(pn);
      for (u64 r = 0; r < rows; r++) {
        coins.fill(pads.data(), pads.size());
        for (size_t k = 0; k < pn; k++) pad_words[r * pn + k] = pads[k].v;
      } }
    // Cell (y, c): entry j + 2^k_lo y of table row r, c = j + 2^k_lo r.
    message_ = std::make_unique<mx::array>(index_message(M.cells, M.rows, M.columns, rows, c.ell, per, shape_.plain(), pad_words, pn));
    TableWords msg, tl;
    msg.a = Buf(*message_), msg.deg = 1, msg.rows = rows, msg.len = c.ell;
    tail_ = std::make_unique<mx::array>(mx::array(reinterpret_cast<const uint64_t *>(tail_words.data()), {(int)tail_words.size()}, mx::uint64));
    tl.a = Buf(*tail_), tl.deg = 1, tl.rows = rows, tl.len = c.r;
    printf("table: message laid out\n");
    const auto a0 = std::chrono::steady_clock::now();
    each_ = std::min<u64>(rows, 16); // one much larger dispatch runs several times slower per row
    for (u64 r = 0; r < rows; r += each_) word_.push_back(encode_table(msg, tl, c.message_at(), c.tail_at(), c.ell, c.r, c.log_dim(), c.log_n, transform_table(c.log_n)->data(), r, each_));
    const auto a1 = std::chrono::steady_clock::now();
    tree_ = std::make_unique<OracleTreeMlx>(index_tree(shape_, configuration_), coins);
    const auto a2 = std::chrono::steady_clock::now();
    root_ = tree_->build_chunks(word_, each_);
    const auto a3 = std::chrono::steady_clock::now();
    auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    printf("table: encoded in %.1f s, salts %.1f s, tree %.1f s\n", sec(a0, a1), sec(a1, a2), sec(a2, a3));
  }
  const IndexShape &shape() const override { return shape_; }
  std::vector<u8> commitment() const override { return digest_bytes(root_); }
  std::vector<u8> name() const override { return index_name(root_, shape_, configuration_); }
  Digest configuration() const override { return configuration_; }
  IndexRecord record() override {
    const std::vector<SampleSeed> seed = record_seeds(shape_, name());
    IndexRecord rec;
    rec.y.resize(shape_.rows() * shape_.answers);
    in_parallel(shape_.rows(), [&](u64 r) {
      const F *m = reinterpret_cast<const F *>(message_->data<uint64_t>() + r * shape_.code.ell);
      for (unsigned i = 0; i < shape_.answers; i++) rec.y[r * shape_.answers + i] = record_value(shape_, seed[i], i, m);
    });
    return rec;
  }
  // From here on the count of opened columns is the ledger's, which outlives this object.
  void keep_count_in(const SpentLedger &ledger) { ledger_ = std::make_unique<SpentLedger>(ledger); }
  bool begin(u64 columns) override {
    const u64 limit = hiding_positions(shape_.code);
    if (ledger_) {
      unsigned long long now = 0;
      if (!ledger_->spend(columns, limit, &now)) return false;
      printf("table: %llu of %llu columns spent\n", now, (unsigned long long)limit);
      return true;
    }
    if (columns > limit - spent_) return false;
    spent_ += columns;
    return true;
  }
  void fold(const E *gamma, std::vector<E> &message, std::vector<E> &tail, BlockSums *blocks) override {
    std::vector<E> w(shape_.rows());
    eq_table(gamma, shape_.v, w.data());
    const size_t ell = shape_.code.ell, r = shape_.code.r;
    message = rows_left(*message_, shape_.rows(), ell, w);
    tail = rows_left(*tail_, shape_.rows(), r, w);
    if (blocks) block_sums(message, *blocks);
  }
  void open(const PositionList &at, Writer &file, std::vector<F> &columns) override {
    Opening o;
    const u64 n = shape_.code.n();
    tree_->open(at, o, [&](u64 x, std::vector<F> &col) {
      col.resize(shape_.rows());
      for (u64 r = 0; r < shape_.rows(); r++) col[r] = F{word_[r / each_].data()[(r % each_) * n + x]};
    });
    write_opening(file, index_tree(shape_, configuration_), at, o);
    columns = std::move(o.columns);
  }
private:
  IndexShape shape_;
  Digest configuration_, root_{};
  std::unique_ptr<mx::array> message_, tail_;
  std::vector<Buf> word_;
  u64 each_ = 1;
  std::unique_ptr<OracleTreeMlx> tree_;
  u64 spent_ = 0;
  std::unique_ptr<SpentLedger> ledger_;
};
// W_j = g_j^T M on the byte matrix.
class MemberMlx : public MemberHeavy {
public:
  MemberMlx(const LinShape &S, const CellMatrix &M) : S_(S), M_(M) {}
  void begin(const std::vector<std::vector<Tensor>> &g) override {
    w_.clear();
    for (const std::vector<Tensor> &x : g) {
      const unsigned vars = S_.k_row();
      std::vector<E> gv;
      if (vars >= 12) gv = e_download(tensors_written(x, vars));
      else {
        gv.assign(size_t(1) << vars, E::zero());
        for (const Tensor &t : x) t.add_to(gv.data());
      }
      gv.resize(S_.matrix_rows);
      std::vector<E> w = e_download(matrix_left(M_.cells, M_.rows * M_.columns, 0, M_.rows, M_.columns, gv));
      w_.push_back(std::move(w));
    }
  }
  std::unique_ptr<SumcheckProver<E>> sumcheck(const Summand<E> &s, const std::vector<Weight<E>> &vars, const std::vector<E> &, const std::vector<std::vector<E>> &) override {
    return std::unique_ptr<SumcheckProver<E>>(new SumcheckCpu<E>(s, vars, w_));
  }
  std::vector<E> combine(const std::vector<E> &a) override {
    std::vector<E> w(M_.columns, E::zero());
    for (size_t j = 0; j < a.size(); j++)
      for (u64 c = 0; c < M_.columns; c++) w[c] += a[j] * w_[j][c];
    return w;
  }
private:
  const LinShape &S_;
  CellMatrix M_;
  std::vector<std::vector<E>> w_;
};
}
