// Compiles every kernel of a pack, in parallel, and reports failures and times.
// usage: compile_pack <pack> <prelude> [threads] [substring]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
using namespace std;
struct K { string name; size_t off, len; double ms; bool ok; };
int main(int c, char **v) {
  ifstream f(v[1], ios::binary); stringstream ss; ss << f.rdbuf(); string d = ss.str();
  ifstream pf(v[2]); stringstream ps; ps << pf.rdbuf(); string prelude = ps.str();
  int T = c > 3 ? atoi(v[3]) : 8;
  uint64_t n[3]; memcpy(n, d.data() + 8, 24);
  string idx = d.substr(32, n[0]); size_t blob = 32 + n[0] + n[1];
  vector<K> ks; { istringstream is(idx); string l; while (getline(is, l)) if (l.rfind("kernel ", 0) == 0) { istringstream t(l.substr(7)); K k; t >> k.name >> k.off >> k.len; if (c <= 4 || k.name.find(v[4]) != string::npos) ks.push_back(k); } }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  atomic<size_t> next{0}; mutex m; atomic<int> bad{0};
  auto t0 = chrono::steady_clock::now();
  vector<thread> th;
  for (int i = 0; i < T; i++) th.emplace_back([&] { @autoreleasepool {
    for (;;) { size_t i = next++; if (i >= ks.size()) break;
      K &k = ks[i]; string body = d.substr(blob + k.off, k.len); size_t at = body.find("//PRELUDE\n");
      string s = body.substr(0, at) + prelude + body.substr(at + 10);
      auto a = chrono::steady_clock::now();
      MTLCompileOptions *o = [MTLCompileOptions new]; o.languageVersion = MTLLanguageVersion3_2; o.mathMode = MTLMathModeSafe;
      NSError *e = nil;
      id<MTLLibrary> l = [dev newLibraryWithSource:[NSString stringWithUTF8String:s.c_str()] options:o error:&e];
      id<MTLComputePipelineState> p = nil;
      if (l) p = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:@"K"] error:&e];
      k.ms = chrono::duration<double, milli>(chrono::steady_clock::now() - a).count(); k.ok = p != nil;
      if (!p) { bad++; lock_guard<mutex> g(m); string msg = e.localizedDescription.UTF8String; size_t q = msg.find("error:"); printf("FAIL %s: %s\n", k.name.c_str(), msg.substr(q == string::npos ? 0 : q, 300).c_str()); fflush(stdout); }
    }
  }});
  for (auto &t : th) t.join();
  double wall = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
  sort(ks.begin(), ks.end(), [](const K &a, const K &b) { return a.ms > b.ms; });
  printf("%zu kernels, %d failed, %.1f s wall\n", ks.size(), (int)bad, wall);
  for (size_t i = 0; i < 8 && i < ks.size(); i++) printf("  %8.0f ms %8zu B %s\n", ks[i].ms, ks[i].len, ks[i].name.substr(0, 90).c_str());
}
