// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Model of docs/model.md, in exact integers, on MLX arrays: elementwise steps are MLX's int64 ops;
// the exact integer product and the wide norm are Metal kernels (MLX's matmul is floating point).
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <mlx/mlx.h>
namespace fpmlx {
namespace mx = mlx::core;
struct LayerScales { int kq, kk, kv, ko, kg, ku, kd, M1, K1, M2, K2, sh, rp, kr, kgR, kuR; bool experts; };
struct ModelShape {
  int T, L, d, n_h, n_g, dh, f, fe, NE, V, cs, dq, dkv, MF, KF;
  long long eps1, eps2, epsF;
  bool gcl, rcl;
  std::vector<LayerScales> layers;
};
using Tensors = std::unordered_map<std::string, mx::array>;
using Note = std::function<void(const std::string &, const mx::array &)>;
ModelShape read_shape(const std::string &params_txt);
// x (n x k), w (o x k): the exact product, int64.
mx::array exact_product(const mx::array &x, const mx::array &w);
// Runs tokens through the model; returns the logits (T x V). note() gets the recorded names of model.py.
mx::array forward(const ModelShape &p, const Tensors &w, const mx::array &tokens, const Note &note);
}
