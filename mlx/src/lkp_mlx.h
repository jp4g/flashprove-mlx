// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// blk::LkpCpu (blocks/iop/lkp.h) with its trees and sumchecks on Metal, and blk::WfCpu with that lookup.
// The summands of a lookup are known by their step, so each is given to the kernels as monomials:
// KEYS, TABLE y0 y1; BITS y0^2 - y0; LEAVES 1 + b R, R being the table's factor, a map of the same index.
#pragma once
#include "iop/lkp.h"
#include "iop/wf.h"
#include "mlx_kernels.h"
namespace fpmlx {
using namespace blk;
class SumcheckMlx : public SumcheckProver<E> {
public:
  template <class Maps> SumcheckMlx(Maps &&maps, unsigned count, SumShape shape, const std::vector<Weight<E>> &vars, const std::vector<unsigned> &order, size_t ends)
      : rounds_(make(std::forward<Maps>(maps), count, std::move(shape), vars, order)), vars_(vars), bits_(order), ends_(ends) {}
  std::vector<E> round() override {
    BLK_REQUIRE(j_ < vars_.size(), "sum-check: no variable is left");
    return poly_from_values_at_0_1_2<E>(rounds_->nodes(j_, vars_[j_].d, size_t(1) << bits_[j_]));
  }
  void bind(const E &g) override {
    BLK_REQUIRE(j_ < vars_.size(), "sum-check: no variable is left");
    const unsigned b = bits_[j_];
    rounds_->bind(g, size_t(1) << b);
    for (size_t t = ++j_; t < bits_.size(); t++) if (bits_[t] > b) bits_[t]--;
  }
  std::vector<E> end_values() override {
    BLK_REQUIRE(j_ == vars_.size(), "sum-check: variables are left");
    std::vector<std::vector<E>> m;
    rounds_->fetch(m);
    std::vector<E> v;
    for (size_t i = 0; i < ends_; i++) v.push_back(m[i][0]);
    return v;
  }
private:
  static std::vector<SumWeight> later_of(const std::vector<Weight<E>> &vars, std::vector<unsigned> at) {
    std::vector<SumWeight> later;
    for (size_t l = 0; l + 1 < vars.size(); l++) {
      for (size_t t = l + 1; t < at.size(); t++) if (at[t] > at[l]) at[t]--;
      later.push_back({vars[l + 1].moment[0] - vars[l + 1].moment[1], vars[l + 1].moment[1], at[l + 1]});
    }
    return later;
  }
  template <class Maps> static std::unique_ptr<SumRounds> make(Maps &&maps, unsigned count, SumShape shape, const std::vector<Weight<E>> &vars, const std::vector<unsigned> &order) {
    if constexpr (std::is_same_v<std::decay_t<Maps>, mx::array>) return std::make_unique<SumRounds>(maps, count, std::move(shape), later_of(vars, order));
    else return std::make_unique<SumRounds>(maps, std::move(shape), later_of(vars, order));
  }
  std::unique_ptr<SumRounds> rounds_;
  std::vector<Weight<E>> vars_;
  std::vector<unsigned> bits_;
  size_t ends_, j_ = 0;
};
class LkpMlx : public LkpProver {
public:
  // keys: words of F, all_keys() of them, on the GPU. bits as LkpCpu's.
  LkpMlx(const LkpPublic &fixed, const mx::array &keys, const std::vector<E> &bits) : fixed_(fixed), keys_(keys), bits_(e_on_gpu(bits)), rows_(mx::array(0)) {
    BLK_REQUIRE(keys.size() == fixed.all_keys() && bits.size() == size_t(1) << (fixed.mu + fixed.d), "LKP: keys or bits have not the announced number");
  }
  void begin(const E &theta, const E &alpha) override {
    key_levels_.clear();
    std::vector<u64> blocks;
    std::vector<E> rows;
    for (const LkpKeyBlock &b : fixed_.keys) {
      blocks.push_back(b.first), blocks.push_back(b.log_trees), blocks.push_back((u64)(b.function + 1));
      rows.push_back(b.function < 0 ? lkp_row(fixed_, theta, b.padding_row) : E::zero());
    }
    size_t first = 0;
    for (unsigned part = 0; part < fixed_.part_bits().size(); part++) {
      const unsigned v = fixed_.part_bits()[part];
      mx::array leaves = lkp_leaves(keys_, first, v, fixed_.g, blocks, rows, alpha, theta);
      mx::eval(leaves);
      std::vector<mx::array> done;
      for (const LkpPadding &p : fixed_.padding)
        if (p.part == part) done.push_back(lkp_pad(leaves, p.first, p.log_step, p.length, alpha - lkp_row(fixed_, theta, p.row)));
      mx::eval(done);
      key_levels_.push_back(tree(leaves, size_t(1) << v, 1));
      first += size_t(1) << v;
    }
    const size_t w = size_t(1) << fixed_.mu;
    TableSide side = lkp_table_side(bits_, e_on_gpu(lkp_rows(fixed_, theta)), w, alpha);
    rows_ = side.bits_rows;
    table_levels_ = tree(side.leaves, bits_.size() / 3, w);
  }
  std::vector<E> key_roots() override {
    std::vector<E> roots;
    for (const auto &levels : key_levels_) roots.push_back(e_fetch(levels[0])[0]);
    return roots;
  }
  std::vector<E> plane_roots() override { return e_fetch(table_levels_[0]); }
  std::unique_ptr<SumcheckProver<E>> sumcheck(LkpStep step, unsigned part, unsigned level, const Summand<E> &, const std::vector<Weight<E>> &vars, const std::vector<unsigned> &order) override {
    SumShape shape;
    if (step == LkpStep::KEYS || step == LkpStep::TABLE) {
      shape.coeff = {E::one()}, shape.maps = {{0, 1}};
      const mx::array &below = (step == LkpStep::KEYS ? key_levels_[part] : table_levels_)[level + 1];
      return std::make_unique<SumcheckMlx>(below, 2, std::move(shape), vars, order, 2);
    }
    if (step == LkpStep::BITS) {
      shape.coeff = {E::one(), E::zero() - E::one()}, shape.maps = {{0, 0}, {0}};
      return std::make_unique<SumcheckMlx>(bits_, 1, std::move(shape), vars, order, 1);
    }
    shape.coeff = {E::one(), E::one()}, shape.maps = {{}, {0, 1}};
    return std::make_unique<SumcheckMlx>(mx::concatenate({bits_, rows_}, 0), 2, std::move(shape), vars, order, 1);
  }
private:
  // levels[l] from the top down; levels[0] has `top` entries.
  static std::vector<mx::array> tree(const mx::array &leaves, size_t n, size_t top) {
    std::vector<mx::array> up(1, leaves);
    for (; n > top; n /= 2) up.push_back(product_level(up.back()));
    std::reverse(up.begin(), up.end());
    return up;
  }
  LkpPublic fixed_;
  mx::array keys_, bits_, rows_;
  std::vector<std::vector<mx::array>> key_levels_;
  std::vector<mx::array> table_levels_;
};
// The heavy part of WF on the GPU. WfCpu stays its reference: its keys() is still the definition.
class WfMlx : public WfCpu {
public:
  WfMlx(const WfPublic &x, const WfMatrix &m, const int8_t *cells, u64 matrix_cells, u64 columns)
      : WfCpu(x, m, [=](u64 row, u64 column) { return from_i64(cells[row * columns + column]); }), x_(x), m_(m), fixed_(wf_lookups(x, m)), cells_(cells), matrix_cells_(matrix_cells), columns_(columns) {}
  bool multiplicities(size_t lookup, std::vector<u64> &count) override {
    const auto [keys, pad] = gpu_keys(lookup);
    const unsigned d = x_.lookups[lookup].d;
    std::vector<u64> k(keys.size());
    std::vector<char> p(keys.size());
    memcpy(k.data(), keys.data<uint64_t>(), keys.size() * 8);
    memcpy(p.data(), pad.data<uint8_t>(), keys.size());
    std::vector<u32> c(size_t(1) << d, 0);
    if (!lookup_counts(k, p, d, {}, {}, c.data())) return false;
    count.assign(c.begin(), c.end());
    return true;
  }
  LkpProver &prover(size_t lookup) override {
    std::vector<u64> count;
    BLK_REQUIRE(multiplicities(lookup, count), "WF: a key in no row of its table");
    const LkpPublic &p = fixed_[lookup];
    std::vector<E> bits(size_t(1) << (p.mu + p.d), E::zero());
    for (size_t t = 0; t < count.size(); t++)
      for (size_t j = 0; j < p.planes(); j++)
        if ((count[t] >> j) & 1) bits[j + (t << p.mu)] = E::one();
    now_.reset(new LkpMlx(p, gpu_keys(lookup).first, bits));
    return *now_;
  }
private:
  // The keys of keys(), a word of F each, and their padding flags.
  std::pair<mx::array, mx::array> gpu_keys(size_t lookup) const {
    const WfLookup &l = x_.lookups[lookup];
    size_t total = 0;
    for (const WfPart &p : l.parts) total += columns_ << p.log_places;
    std::vector<u64> fill(total, l.padding_key.v);
    mx::array keys(reinterpret_cast<const uint64_t *>(fill.data()), {(int)total}, mx::uint64);
    mx::array pad(mx::allocator::malloc(total), {(int)total}, mx::uint8);
    memset(pad.data<uint8_t>(), 1, total);
    mx::eval(keys);
    std::vector<mx::array> done;
    size_t first = 0;
    for (const WfPart &p : l.parts) {
      for (const WfRun &r : p.runs)
        done.push_back(wf_run(keys, pad, cells_, matrix_cells_, columns_, first + r.first_place * columns_, r.first_row, r.places, r.stride, l.offset, l.digit_weight));
      first += columns_ << p.log_places;
    }
    mx::eval(done);
    return {keys, pad};
  }
  const WfPublic x_;
  const WfMatrix m_;
  const std::vector<LkpPublic> fixed_;
  const int8_t *cells_;
  u64 matrix_cells_, columns_;
  std::unique_ptr<LkpMlx> now_;
};
}
