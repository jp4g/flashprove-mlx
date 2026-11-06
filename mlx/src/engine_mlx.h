// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// EngineMlx implements Engine (blocks/engine/engine.h) on MLX: encoding, leaf hashes, the Merkle
// tree's levels, evaluations and folds run as Metal kernels on mlx::core::array (mlx_kernels.h). Its
// bookkeeping (lifetimes, loans, checks, the order of draws from `coins`) is EngineCpu's, line for
// line, so equal `coins` give equal trees. mlx/tests/engine_test.cpp compares the two.
//
// A part (`OraclePart`) is one oracle message in a Merkle tree: main rows (in a table of the
// engine), mask rows (passed in by the caller) or both. A row's `tail` is its encoding randomness.
#pragma once
#include <algorithm>
#include <atomic>
#include <functional>
#include <optional>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "code/short_position.h"
#include "engine/engine.h"
#include "mlx_kernels.h"

namespace blk {
namespace mx = mlx::core;

class EngineMlx : public Engine {
public:
  ~EngineMlx() override {
    still();
    for (const auto &l : borrowed_trees_) l.second.lender->lent_trees_.erase(l.second.theirs); // what was borrowed goes back
    for (const auto &l : borrowed_tables_) l.second.lender->lent_tables_.erase(l.second.theirs);
    BLK_REQUIRE(lent_trees_.empty(), "engine: it ends while another engine has borrowed a tree of it");
    BLK_REQUIRE(lent_tables_.empty(), "engine: it ends while another engine has borrowed a table of it");
  }

  // ---- one thread at a time; loans (engine.h) ----
  void let_go() override {
    mine();
    holder_.store(0, std::memory_order_release);
  }
  void take_over() override {
    size_t nobody = 0;
    BLK_REQUIRE(holder_.compare_exchange_strong(nobody, me(), std::memory_order_acq_rel), "engine: taken over while a thread holds it");
  }
  TableId borrow_table(Engine &from, TableId theirs) override {
    mine();
    EngineMlx &lender = of_my_kind(from);
    lender.own_table(theirs);
    BLK_REQUIRE(!lender.lent_tables_.count(theirs) && !lender.borrowed_tables_.count(theirs), "borrow: the table is lent already, or is itself borrowed");
    lender.lent_tables_.insert(theirs);
    tables_.push_back(nullptr); // the number is taken; the table stays where it is
    borrowed_tables_[(TableId)(tables_.size() - 1)] = Loan{&lender, theirs, {}};
    return (TableId)(tables_.size() - 1);
  }
  CommitId borrow_tree(Engine &from, CommitId theirs) override {
    mine();
    EngineMlx &lender = of_my_kind(from);
    const Committed &t = lender.own_tree(theirs);
    BLK_REQUIRE(!lender.lent_trees_.count(theirs) && !lender.borrowed_trees_.count(theirs), "borrow: the tree is lent already, or is itself borrowed");
    Loan loan{&lender, theirs, {}};
    for (const OraclePart &p : t.parts) loan.tables.push_back(p.main == NO_ID ? NO_ID : borrow_table(lender, p.main));
    lender.lent_trees_.insert(theirs);
    commits_.push_back(nullptr);
    borrowed_trees_[(CommitId)(commits_.size() - 1)] = loan;
    return (CommitId)(commits_.size() - 1);
  }
  std::vector<TableId> borrowed_tables(CommitId id) override {
    mine();
    BLK_REQUIRE(borrowed_trees_.count(id), "engine: no such borrowed tree");
    return borrowed_trees_[id].tables;
  }

  // ---- main tables ----
  TableId put(const Table &t) override {
    mine();
    tables_.push_back(std::make_unique<Table>(t));
    return (TableId)(tables_.size() - 1);
  }
  Table get(TableId t) override { return mine(), table(t); }
  void drop(TableId t) override {
    mine();
    table(t);
    for (const auto &c : commits_)
      if (c)
        for (const OraclePart &p : c->parts) BLK_REQUIRE(p.main != t, "drop: a tree that is not closed stands over this table");
    for (const auto &l : borrowed_trees_)
      for (TableId under : l.second.tables) BLK_REQUIRE(under != t, "drop: a tree that is not closed stands over this table");
    BLK_REQUIRE(!lent_tables_.count(t), "drop: another engine has borrowed this table");
    for (const auto &f : folds_)
      if (f)
        for (TableId h : f->hinted) BLK_REQUIRE(h != t, "drop: a fold that has not ended reads this table");
    if (borrowed_tables_.count(t)) borrowed_tables_[t].lender->lent_tables_.erase(borrowed_tables_[t].theirs), borrowed_tables_.erase(t); // the loan ends
    tables_[t].reset();
  }
  TableId reshape(TableId id, unsigned log_rows) override {
    mine();
    const Table &t = table(id);
    const u64 total = t.entries();
    BLK_REQUIRE((total >> log_rows) << log_rows == total && (total >> log_rows) > 0, "reshape: the rows do not divide the table");
    Table s(t.deg, size_t(1) << log_rows, total >> log_rows);
    for (u64 j = 0; j < total; j++) s.set_entry(j, t.entry(j));
    return put(s);
  }
  std::vector<E> entries(TableId id, const std::vector<u64> &at) override {
    mine();
    const Table &t = table(id);
    std::vector<E> out;
    out.reserve(at.size());  // (no growth: what is fetched may be a one-time pad, and a vector that grows frees a buffer unwiped)
    for (u64 j : at) {
      BLK_REQUIRE(j < t.entries(), "entries: an index beyond the table");
      out.push_back(t.entry(j));
    }
    return out;
  }
  TableId with_entries(TableId id, const std::vector<std::pair<u64, E>> &changed) override {
    mine();
    Table t = table(id);
    std::vector<u64> seen;
    for (const std::pair<u64, E> &c : changed) {
      BLK_REQUIRE(c.first < t.entries(), "with_entries: an index beyond the table");
      seen.push_back(c.first);
      t.set_entry(c.first, c.second);
    }
    std::sort(seen.begin(), seen.end());
    BLK_REQUIRE(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "with_entries: an index comes twice");
    return put(t);
  }
  void set_entries(TableId id, const std::vector<std::pair<u64, E>> &changed) override {
    mine();
    Table &t = table(id);
    BLK_REQUIRE(!borrowed_tables_.count(id) && !lent_tables_.count(id), "set_entries: a borrowed table, or one that is lent");
    for (const auto &c : commits_)
      if (c)
        for (const OraclePart &p : c->parts) BLK_REQUIRE(p.main != id, "set_entries: a tree that is not closed stands over this table");
    for (const auto &f : folds_)
      if (f)
        for (TableId h : f->hinted) BLK_REQUIRE(h != id, "set_entries: a fold that has not ended reads this table");
    std::vector<u64> seen;
    for (const std::pair<u64, E> &c : changed) {
      BLK_REQUIRE(c.first < t.entries(), "set_entries: an index beyond the table");
      seen.push_back(c.first);
    }
    std::sort(seen.begin(), seen.end());
    BLK_REQUIRE(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "set_entries: an index comes twice");
    for (const std::pair<u64, E> &c : changed) t.set_entry(c.first, c.second);
  }
  TableId laid(TableId id, u64 count, u64 first_row, unsigned bits, size_t rows, size_t len) override {
    mine();
    const Table &t = table(id);
    BLK_REQUIRE(t.deg == E_DEGREE && t.rows == 1 && count <= t.len, "laid: one row over E, and no more entries than it has");
    BLK_REQUIRE(bits < 64 && first_row + (u64(1) << bits) <= rows && ((count + (u64(1) << bits) - 1) >> bits) <= len,
                "laid: the block of rows does not lie in the new table, or is too short");
    Table s(E_DEGREE, rows, len);
    for (u64 j = 0; j < count; j++) s.set(first_row + (j & ((u64(1) << bits) - 1)), j >> bits, t.at(0, j));
    return put(s);
  }
  E evaluate(TableId id, const Tensor &v) override {
    mine();
    const Table &t = table(id);
    BLK_REQUIRE((u64(1) << v.vars()) == t.entries(), "evaluate: the vector and the table differ in length");
    if (!v.is_held() && v.vars() >= 12 && !fpmlx::in_forked_child() && !fpmlx::reference_paths()) return dot_with_table(fpmlx::tensors_written({v}, v.vars()), t);
    std::vector<E> w(t.entries(), E::zero());
    (v.is_held() ? readable(v) : v).add_to(w.data());
    return dot_with_table(w, t);
  }

  E evaluate(TableId id, const Form &v) override {
    mine();
    const Table &t = table(id);
    BLK_REQUIRE((u64(1) << ceil_log2(t.entries())) == t.entries() && v.fits(ceil_log2(t.entries())),
                "evaluate: the vector and the table differ in length");
    return dot_with_table(written_form(v, t.entries(), ceil_log2(t.entries())), t);
  }

  // ---- oracles ----
  CommitId commit(const std::vector<OraclePart> &parts, const OracleTreeShape &shape, Coins &coins, Digest &root) override {
    mine();
    BLK_REQUIRE(parts.size() == shape.main_symbols.size() && parts.size() == shape.mask_symbols.size(),
                "commit: the shape speaks of another number of parts");
    auto c = std::make_unique<Committed>();
    c->parts = parts;
    c->shape = shape;
    for (size_t i = 0; i < parts.size(); i++) {
      const OraclePart &p = parts[i];
      Words main, mask;
      Table tail;
      if (p.main != NO_ID) {
        const Table &m = table(p.main);
        BLK_REQUIRE(m.len == p.code.ell && p.code.log_n == shape.log_n, "commit: a main row does not fit its code or its tree");
        tail = Table(m.deg, m.rows, p.code.r);
        coins.fill(tail.a.data(), tail.a.size());
        main = encode_rows(p.code, m, tail);
      }
      if (p.mask.rows) {
        BLK_REQUIRE(p.mask.deg == E_DEGREE && p.mask_tail.deg == E_DEGREE, "commit: mask rows are over E");
        BLK_REQUIRE(p.mask.len == p.mask_code.ell && p.mask_tail.len == p.mask_code.r && p.mask_tail.rows == p.mask.rows &&
                        p.mask_code.log_n == shape.log_n_zk,
                    "commit: a mask row does not fit the mask code or its tree");
        mask = encode_rows(p.mask_code, p.mask, p.mask_tail);
      }
      BLK_REQUIRE(main.rows == shape.main_symbols[i] && mask.rows == shape.mask_symbols[i], "commit: rows other than the shape says");
      c->main.push_back(std::move(main));
      c->mask.push_back(std::move(mask));
      c->tail.push_back(std::move(tail));
    }
    c->tree = std::make_unique<fpmlx::OracleTreeMlx>(shape, coins);
    std::vector<fpmlx::Buf> main_words, mask_words;
    for (const Words &w : c->main) if (w.rows) main_words.push_back(w.a);
    for (const Words &w : c->mask) if (w.rows) mask_words.push_back(w.a);
    root = c->tree->build(main_words, mask_words);
    commits_.push_back(std::move(c));
    return (CommitId)(commits_.size() - 1);
  }
  void open(CommitId id, const PositionList &at, Opening &out) override {
    mine();
    Committed &c = committed(id);
    c.tree->open(at, out, [&](u64 x, std::vector<F> &col) { leaf_column(c, x, col); });
  }
  Table tail(CommitId id, unsigned part) override {
    mine();
    Committed &c = committed(id);
    BLK_REQUIRE(part < c.tail.size(), "engine: no such part");
    return c.tail[part];
  }
  void close(CommitId id) override {
    mine();
    committed(id);
    for (const auto &f : folds_)
      if (f)
        for (CommitId c : f->over) BLK_REQUIRE(c != id, "close: a fold over this tree's rows has not ended");
    BLK_REQUIRE(!lent_trees_.count(id), "close: another engine has borrowed this tree");
    if (borrowed_trees_.count(id)) { // the loan ends: the tree's, then its tables'
      const Loan loan = borrowed_trees_[id];
      loan.lender->lent_trees_.erase(loan.theirs);
      borrowed_trees_.erase(id);
      for (TableId t : loan.tables)
        if (t != NO_ID) drop(t);
      return;
    }
    commits_[id].reset();
  }

  // ---- the fold ----
  // Not checked: each FoldPart::part exists in its tree; Code::ell is a power of two, which
  // write_out() needs.
  FoldId fold_begin(const std::vector<FoldPart> &parts, unsigned k, const Form &c) override {
    mine();
    BLK_REQUIRE(!parts.empty(), "fold: no rows");
    auto f = std::make_unique<Folding>();
    const Code &code = committed(parts[0].commit).parts[parts[0].part].code;
    const size_t rows = size_t(1) << k;
    f->ell = code.ell, f->r = code.r, f->rows = rows;
    // On the GPU a fold's rows are gathered from its tables' words; a tiny one is not worth it.
    const bool gather = !fpmlx::in_forked_child() && !fpmlx::reference_paths() && rows * f->ell >= 8 && rows * f->r >= 8;
    std::optional<fpmlx::mx::array> gm, gt; // no array is made in a forked child: Metal does not survive a fork
    std::vector<fpmlx::mx::array> gathered;
    if (gather) gm = fpmlx::e_rows_buffer(rows * f->ell), gt = fpmlx::e_rows_buffer(rows * f->r);
    else f->m.assign(rows * f->ell, E::zero()), f->tail.assign(rows * f->r, E::zero());
    std::vector<bool> taken(rows, false);
    for (const FoldPart &p : parts) {
      f->over.push_back(p.commit);
      const Committed &cm = committed(p.commit);
      const OraclePart &op = cm.parts[p.part];
      BLK_REQUIRE(op.main != NO_ID && op.code.ell == f->ell && op.code.r == f->r && op.code.log_n == code.log_n,
                  "fold: the rows of a group are under one code");
      const Table &m = borrowed_trees_.count(p.commit) ? borrowed_trees_[p.commit].lender->own_table(op.main) : table(op.main); // a part's number is its owner's
      BLK_REQUIRE((p.first_row & ((u64(1) << m.log_rows) - 1)) == 0 && p.first_row + (u64(1) << m.log_rows) <= rows,
                  "fold: a block of rows is not aligned or does not fit");
      for (size_t b = 0; b < (size_t(1) << m.log_rows); b++) {
        BLK_REQUIRE(!taken[p.first_row + b], "fold: two parts on the same rows");
        taken[p.first_row + b] = true;
      }
      if (gather) {
        gathered.push_back(fpmlx::put_rows(*gm, fpmlx::table_words(m), p.first_row));
        gathered.push_back(fpmlx::put_rows(*gt, fpmlx::table_words(cm.tail[p.part]), p.first_row));
      } else for (size_t b = 0; b < m.rows; b++) {
        for (size_t i = 0; i < f->ell; i++) f->m[(p.first_row + b) * f->ell + i] = m.at(b, i);
        for (size_t i = 0; i < f->r; i++) f->tail[(p.first_row + b) * f->r + i] = cm.tail[p.part].at(b, i);
      }
    }
    const size_t length = rows * f->ell;
    const unsigned vars = ceil_log2(length);
    BLK_REQUIRE(c.fits(vars), "fold: a coefficient vector has not the group's length");
    fpmlx::Buf dense = written_form(c, length, vars);
    for (const Held &h : c.held) check_hint(c, h, vars), f->hinted.push_back(h.vector);
    for (const EngineTerm &t : c.engine_terms) f->hinted.push_back(t.vector);
    for (const Tensor &v : c.tensors)
      if (v.is_held()) f->hinted.push_back(v.held);  // a dense factor stored in a table
    if (gather) fpmlx::mx::eval(gathered), f->gm = fpmlx::Buf(*gm), f->gt = fpmlx::Buf(*gt);
    else f->gm = fpmlx::e_upload(f->m), f->gt = fpmlx::e_upload(f->tail);
    if (dense.on_gpu()) f->gc = fpmlx::rows_first(dense, rows, f->ell);
    else {
      const E *d = reinterpret_cast<const E *>(dense.data());
      std::vector<E> cc(length);
      for (size_t b = 0; b < rows; b++)
        for (size_t i = 0; i < f->ell; i++) cc[b * f->ell + i] = d[b + rows * i];
      f->gc = fpmlx::e_upload(cc);
    }
    f->m.clear(), f->tail.clear();
    folds_.push_back(std::move(f));
    return (FoldId)(folds_.size() - 1);
  }
  void fold_round(FoldId id, E q[3]) override {
    mine();
    const Folding &f = folding(id);
    BLK_REQUIRE(f.rows >= 2, "fold: no row variable is left");
    fpmlx::fold_round(f.gc, f.gm, f.rows, f.ell, q);
  }
  void fold_bind(FoldId id, const E &gamma) override {
    mine();
    Folding &f = folding(id);
    BLK_REQUIRE(f.rows >= 2, "fold: no row variable is left");
    f.gm = fpmlx::halve(f.gm, f.rows, f.ell, gamma);
    f.gc = fpmlx::halve(f.gc, f.rows, f.ell, gamma);
    f.gt = fpmlx::halve(f.gt, f.rows, f.r, gamma);
    f.rows /= 2;
  }
  TableId fold_end(FoldId id, Table &tail, TableId *c_folded = nullptr) override {
    mine();
    Folding &f = folding(id);
    BLK_REQUIRE(f.rows == 1, "fold: row variables are left");
    const std::vector<E> fm = fpmlx::e_download(f.gm), ft = fpmlx::e_download(f.gt);
    Table m(E_DEGREE, 1, f.ell);
    for (size_t i = 0; i < f.ell; i++) m.set(0, i, fm[i]);
    tail = Table(E_DEGREE, 1, f.r);
    for (size_t i = 0; i < f.r; i++) tail.set(0, i, ft[i]);
    if (c_folded) {
      const std::vector<E> fc = fpmlx::e_download(f.gc);
      Table c(E_DEGREE, 1, f.ell);
      for (size_t i = 0; i < f.ell; i++) c.set(0, i, fc[i]);
      *c_folded = put(c);
    }
    folds_[id].reset();
    return put(m);
  }

  // ---- work ----
  u64 search_nonce(const ChainPrefix &prefix, unsigned bits) override {
    mine();
    return fpmlx::search_nonce(prefix, bits);
  }

private:
  // Codewords over the field F: a row over E, its cubic extension, gives three, one for each
  // coordinate, in order.
  struct Words {
    size_t rows = 0, n = 0;
    fpmlx::Buf a; // rows x n
  };
  struct Committed {
    std::vector<OraclePart> parts;
    std::vector<Table> tail; // encoding randomness (from `coins`) of the main rows, part by part
    std::vector<Words> main, mask;
    OracleTreeShape shape;
    std::unique_ptr<fpmlx::OracleTreeMlx> tree;
  };
  // A fold in progress. m: the folded main rows, as one table; c: the coefficient vector; tail: the
  // encoding randomness. Each has `rows` rows, of ell, ell and r entries. fold_begin() copies all
  // three; `over` and `hinted` only enforce engine.h's rules for close(), drop() and set_entries(),
  // which engines that do not copy need.
  struct Folding {
    size_t ell = 0, r = 0, rows = 0;
    std::vector<E> m, tail;   // only while fold_begin() gathers the rows
    fpmlx::Buf gm, gc, gt; // m, c and tail on the GPU, E entries, row by row
    std::vector<CommitId> over;   // the trees whose rows are folded
    std::vector<TableId> hinted;  // the tables named by the `Form` that c was computed from
  };

  // Returns v with the table of its dense factor (Tensor::held) copied into a host vector.
  Tensor readable(const Tensor &v) {
    const Table &t = table(v.held);
    auto entries = std::make_shared<std::vector<E>>(t.entries());
    for (u64 j = 0; j < t.entries(); j++) (*entries)[j] = t.entry(j);
    return v.with_table(std::move(entries));
  }
  // Adds c, written out on `vars` variables, to `out`, with Form::engine_terms, which only an
  // engine can read. Not checked: `out` has 2^vars entries.
  // The Form's vector, `length` entries at the least. The Tensors are written on the GPU; what is left
  // of a Form, its few single entries and the engine's terms, is added to the same memory.
  fpmlx::Buf written_form(const Form &c, size_t length, unsigned vars) {
    if (fpmlx::in_forked_child() || fpmlx::reference_paths() || vars < 12 || !c.frozen.empty() || c.has_held_factor()) {
      std::vector<E> w(length, E::zero());
      write_out(c, w, vars);
      const u64 *p = reinterpret_cast<const u64 *>(w.data());
      if (fpmlx::in_forked_child()) return fpmlx::Buf(std::vector<u64>(p, p + 3 * length));
      return fpmlx::e_upload(w);
    }
    fpmlx::Buf out = fpmlx::tensors_written(c.tensors, vars);
    for (const EngineTerm &term : c.engine_terms) {
      const Table &t = table(term.vector);
      BLK_REQUIRE(t.entries() << __builtin_popcountll(term.choice_mask) == length, "engine: a term's vector has not the length that the form gives it");
      fpmlx::add_term(out, fpmlx::table_words(t), term.scale, term.choice_mask, term.choice_value);
    }
    E *o = const_cast<E *>(reinterpret_cast<const E *>(out.data()));
    for (const std::pair<u64, E> &e : c.entries) {
      BLK_REQUIRE(e.first >> vars == 0, "form: an entry beyond the vector's length");
      o[e.first] += e.second;
    }
    return out;
  }
  void write_out(const Form &c, std::vector<E> &out, unsigned vars) {
    if (c.has_held_factor()) {
      Form host = c;
      for (Tensor &v : host.tensors)
        if (v.is_held()) v = readable(v);
      host.add_written_to(out.data(), vars);
    } else {
      c.add_written_to(out.data(), vars);
    }
    for (const EngineTerm &term : c.engine_terms) {
      const Table &t = table(term.vector);
      std::vector<E> w(t.entries());
      for (u64 j = 0; j < t.entries(); j++) w[j] = t.entry(j);
      BLK_REQUIRE(w.size() << __builtin_popcountll(term.choice_mask) == out.size(), "engine: a term's vector has not the length that the form gives it");
      for (u64 j = 0; j < out.size(); j++) out[j] += engine_term_at(term, w, j, vars);
    }
  }
  // This engine computes every part of a Form itself and uses a hint only to check it: the tensors
  // and entries that h names must add up to the vector that h represents (held_at()).
  void check_hint(const Form &c, const Held &h, unsigned vars) {
    BLK_REQUIRE(h.from <= h.to && h.to <= c.tensors.size() && h.entries_from <= h.entries_to && h.entries_to <= c.entries.size(),
                "fold: a hint names parts that the form has not");
    Form named;
    named.tensors.assign(c.tensors.begin() + h.from, c.tensors.begin() + h.to);
    named.entries.assign(c.entries.begin() + h.entries_from, c.entries.begin() + h.entries_to);
    std::vector<E> sum(size_t(1) << vars, E::zero());
    write_out(named, sum, vars);
    const Table &t = table(h.vector);
    std::vector<E> w(t.entries());
    for (u64 j = 0; j < t.entries(); j++) w[j] = t.entry(j);
    for (u64 j = 0; j < sum.size(); j++) BLK_REQUIRE(sum[j] == held_at(h, w, j, vars), "fold: a hint differs from the parts that it names");
  }
  // Not checked here: the sizes of `msg` and `tail`; commit() makes sure of them.
  static Words encode_rows(const Code &code, const Table &msg, const Table &tail) {
    Words out;
    out.n = size_t(1) << code.log_n;
    out.rows = msg.rows * msg.deg;
    const size_t block = size_t(1) << code.log_dim();
    if (!fpmlx::in_forked_child() && msg.rows) {
      out.a = fpmlx::encode_table(fpmlx::table_words(msg), fpmlx::table_words(tail), code.message_at(), code.tail_at(), code.ell, code.r, code.log_dim(), code.log_n, transform_table(code.log_n)->data());
      return out;
    }
    std::vector<u64> coefficients(out.rows * block, 0);
    for (size_t b = 0; b < msg.rows; b++)
      for (unsigned d = 0; d < msg.deg; d++) {
        u64 *row = &coefficients[(b * msg.deg + d) * block];
        for (size_t i = 0; i < code.ell; i++) row[code.message_at() + i] = msg.a[(b * code.ell + i) * msg.deg + d].v;
        for (size_t i = 0; i < code.r; i++) row[code.tail_at() + i] = tail.a[(b * code.r + i) * msg.deg + d].v;
      }
    out.a = fpmlx::encode_rows(coefficients.data(), out.rows, code.log_dim(), code.log_n, transform_table(code.log_n)->data());
    return out;
  }
  // Appends the symbols at x of all rows, part by part, row by row.
  static void column(const std::vector<Words> &parts, u64 x, std::vector<F> &col) {
    for (const Words &p : parts)
      for (size_t b = 0; b < p.rows; b++) col.push_back(F{p.a.data()[b * p.n + x]});
  }
  // What the leaf of position x holds: the main rows' symbols at x, behind them the mask rows' at the short position of x. So a
  // mask row's symbol at s is copied into every leaf whose x has the short position s (x modulo the
  // mask codewords' length). In a tree of mask rows only, x is their own position.
  static void leaf_column(const Committed &c, u64 x, std::vector<F> &col) {
    col.clear();
    if (c.shape.has_main()) column(c.main, x, col);
    if (!c.shape.has_mask()) return;
    const size_t main_width = col.size();
    column(c.mask, c.shape.has_main() ? short_position(c.shape.log_n, c.shape.log_n_zk, x) : x, col);
    (void)main_width;
  }
  E dot_with_table(const std::vector<E> &w, const Table &t) { return dot_with_table(fpmlx::e_upload(w), t); }
  E dot_with_table(const fpmlx::Buf &w, const Table &t) {
    if (fpmlx::in_forked_child() || !w.on_gpu() || fpmlx::reference_paths()) {
      std::vector<E> entries(t.entries());
      for (u64 j = 0; j < t.entries(); j++) entries[j] = t.entry(j);
      return fpmlx::dot(w, fpmlx::e_upload(entries));
    }
    return fpmlx::dot_table(w, fpmlx::table_words(t));
  }
  // table() and committed() follow a loan to the lender; own_table() and own_tree() do not.
  Table &table(TableId t) {
    const auto loan = borrowed_tables_.find(t);
    if (loan != borrowed_tables_.end()) return loan->second.lender->still(), loan->second.lender->own_table(loan->second.theirs);
    return own_table(t);
  }
  Table &own_table(TableId t) {
    BLK_REQUIRE(t < tables_.size() && tables_[t], "engine: no such table");
    return *tables_[t];
  }
  Committed &committed(CommitId c) {
    const auto loan = borrowed_trees_.find(c);
    if (loan != borrowed_trees_.end()) return loan->second.lender->still(), loan->second.lender->own_tree(loan->second.theirs);
    return own_tree(c);
  }
  Committed &own_tree(CommitId c) {
    BLK_REQUIRE(c < commits_.size() && commits_[c], "engine: no such tree");
    return *commits_[c];
  }
  // A borrowed table or tree: its owner, its number there and, for a tree, the numbers in this
  // engine of its parts' main tables.
  struct Loan {
    EngineMlx *lender = nullptr;
    u32 theirs = NO_ID;
    std::vector<TableId> tables;
  };
  EngineMlx &of_my_kind(Engine &from) {
    EngineMlx *lender = dynamic_cast<EngineMlx *>(&from);
    BLK_REQUIRE(lender && lender != this, "borrow: from an engine of another kind, or from oneself");
    lender->still();
    return *lender;
  }
  // The calling thread, as a number that is not 0. Two threads could get the same number; mine()
  // and still() would then not tell them apart.
  static size_t me() { return std::hash<std::thread::id>()(std::this_thread::get_id()) | 1; }
  void mine() const { BLK_REQUIRE(holder_.load(std::memory_order_acquire) == me(), "engine: called by a thread that does not hold it"); }
  void still() const { // for a lender in use and for the destructor: no other thread holds it
    const size_t h = holder_.load(std::memory_order_acquire);
    BLK_REQUIRE(h == 0 || h == me(), "engine: another thread holds this engine (a lender at a borrower's call, or an engine at its end)");
  }
  std::atomic<size_t> holder_{me()}; // 0: nobody
  std::map<TableId, Loan> borrowed_tables_;
  std::map<CommitId, Loan> borrowed_trees_;
  std::set<TableId> lent_tables_;
  std::set<CommitId> lent_trees_;
  Folding &folding(FoldId f) {
    BLK_REQUIRE(f < folds_.size() && folds_[f], "engine: no such fold");
    return *folds_[f];
  }
  std::vector<std::unique_ptr<Table>> tables_;
  std::vector<std::unique_ptr<Committed>> commits_;
  std::vector<std::unique_ptr<Folding>> folds_;
};

} // namespace blk
