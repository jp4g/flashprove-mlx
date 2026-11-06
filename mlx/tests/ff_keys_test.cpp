// Every key of a feed-forward segment's lookup is a row of its call's table, on the reference's model.
// usage: ff_keys_test <directory of make_reference.py> <policy.txt>
#include <cstdio>
#include "seg_ff_build.h"
using namespace fpmlx;
static std::string slurp(const char *p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
int main(int c, char **v) {
  const std::string dir = v[1];
  const ModelShape m = read_shape(dir + "/params.txt");
  auto w = mx::load_safetensors(dir + "/weights.safetensors").first;
  for (auto &[k, a] : w) if (k.rfind("table", 0) == 0) a = mx::astype(a, mx::int64);
  const Policy pol = Policy::read(slurp(v[2]));
  const Values V = capture_forward(m, w, w.at("tokens"));
  const CallTables tabs = call_tables(w);
  const int groups = m.L >> pol.get("ll");
  int bad = 0;
  for (int g = 0; g < groups; g++) {
    FFGroup G = ff_build(pol, m, V, tabs, g);
    { // The GPU's witness against the host's, word by word.
      const FFGroup H = ff_build(pol, m, V, tabs, g, false);
      size_t diff = 0, cells = 0, keys = 0;
      const FeatMat *a[] = {&G.af, &G.ad, &G.ar, &G.qa, &G.xbI, &G.xbO}, *b[] = {&H.af, &H.ad, &H.ar, &H.qa, &H.xbI, &H.xbO};
      for (int q = 0; q < 6; q++)
        for (size_t p = 0; p < a[q]->part.size(); p++) {
          const auto &x = a[q]->part[p].cell, &y = b[q]->part[p].cell;
          cells += x.size();
          for (size_t i = 0; i < x.size(); i++) diff += x[i] != y[i];
        }
      printf("group %d: %zu of %zu cells differ\n", g, diff, cells);
      bad += diff != 0, diff = 0;
      for (size_t c = 0; c < G.calls.size(); c++) {
        keys += G.calls[c].keys.size();
        for (size_t i = 0; i < G.calls[c].keys.size(); i++) diff += G.calls[c].keys[i] != H.calls[c].keys[i] || G.calls[c].padding[i] != H.calls[c].padding[i];
      }
      printf("group %d: %zu of %zu keys differ\n", g, diff, keys);
      bad += diff != 0, diff = 0;
      for (size_t i = 0; i < G.rows.size(); i++)
        for (size_t t = 0; t < G.rows[i].size(); t++) {
          const NormRow &x = G.rows[i][t], &y = H.rows[i][t];
          diff += x.D != y.D || x.a != y.a || x.b != y.b || x.sig != y.sig || x.Rlo != y.Rlo || x.Rhi != y.Rhi || x.R != y.R || x.W != y.W;
        }
      printf("group %d: %zu norm rows differ\n", g, diff);
      bad += diff != 0;
    }
    for (size_t i = 0; i < G.calls.size(); i++) {
      std::vector<LookupCall> one{G.calls[i]};
      LookupSizes sz = lookup_sizes(one, 28);
      std::vector<std::vector<u32>> mult;
      const bool ok = lookup_multiplicities(one, sz, mult);
      u64 live = 0; for (char p : G.calls[i].padding) live += !p;
      printf("group %d call %s: 2^%u keys, %llu live: %s\n", g, pol.call[i].name.c_str(), G.calls[i].n, (unsigned long long)live, ok ? "every key is in the table" : "A KEY IS IN NO ROW");
      bad += !ok;
    }
  }
  printf("ff_keys_test: %s\n", bad ? "FAILED" : "all keys in their tables");
  return bad != 0;
}
