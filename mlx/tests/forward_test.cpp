// The MLX forward pass against model.py's recorded values, value by value.
// usage: forward_test <directory of make_reference.py>
#include <chrono>
#include <cstdio>
#include <set>
#include "integer_forward.h"
using namespace fpmlx;
int main(int c, char **v) {
  const std::string dir = v[1];
  const ModelShape p = read_shape(dir + "/params.txt");
  auto weights = mx::load_safetensors(dir + "/weights.safetensors").first;
  auto record = mx::load_safetensors(dir + "/record.safetensors").first;
  for (auto &[k, a] : weights) if (a.dtype() != mx::int64 && k.rfind("table", 0) == 0) a = mx::astype(a, mx::int64);
  long compared = 0, values = 0, bad = 0;
  std::set<std::string> seen;
  const Note note = [&](const std::string &name, const mx::array &a) {
    if (name.rfind("sum_", 0) == 0) return; // no name in model.py
    auto it = record.find(name);
    if (it == record.end()) { printf("no reference for %s\n", name.c_str()); bad++; return; }
    mx::array want = mx::astype(it->second, mx::int64), got = mx::astype(a, mx::int64);
    if (want.shape() != got.shape()) { printf("%s: shapes differ\n", name.c_str()); bad++; return; }
    const long diff = mx::sum(mx::not_equal(want, got)).item<int>();
    compared++, values += want.size(), seen.insert(name);
    if (diff && bad++ < 8) printf("%s: %ld of %zu values differ\n", name.c_str(), diff, want.size());
  };
  const auto t0 = std::chrono::steady_clock::now();
  mx::array lam = forward(p, weights, weights.at("tokens"), note);
  mx::eval(lam);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  for (auto &[k, a] : record) if (!seen.count(k)) printf("not compared: %s\n", k.c_str()), bad++;
  printf("forward_test: %s: %ld tensors, %ld values compared with model.py (%.2f s, comparisons included)\n", bad ? "FAILED" : "all equal", compared, values, s);
  return bad != 0;
}
