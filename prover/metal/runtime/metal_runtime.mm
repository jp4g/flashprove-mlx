// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The CUDA runtime calls of FlashProve's programs, on Metal. prover/metal/README.md describes the
// design; in short:
//  - Memory. An allocation is an MTLBuffer whose pages are mapped a second time at the address the
//    GPU uses for it (mach_vm_remap), so one number means the same on CPU and GPU, as under CUDA's
//    unified addressing, and pointers inside structures need no translation.
//  - Order. All streams share one in-order Metal queue. That is stricter than CUDA's order.
//  - Kernels. __cudaRegisterFunction ties a kernel's host stub to its source in the pack that
//    ptx2msl.py made; it is compiled at its first launch.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <Accelerate/Accelerate.h>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define __CUDA_METAL_RUNTIME__ 1
#include "metal_api.h"

extern const char mtl_prelude_text[];
extern const char mtl_builtin_text[];

bool verbose_now(); bool sync_each_now(); bool trace_now(); const char *dump_path();
namespace {
struct LaunchInfo {
  uint32_t gx, gy, gz, bx, by, bz, dyn, pad;
  uint64_t G, trapw, gsync, locks, scratch, stride;
};
struct Pack;
struct Kernel {
  std::string name;
  Pack *pack;
  const char *src;
  size_t len;
  uint32_t statics, dyn_at, maxntid, param_size;
  std::vector<std::pair<uint32_t, uint32_t>> params; // offset, size
  id<MTLComputePipelineState> pso[2] = {nil, nil};   // threadgroup, device shared memory
  int max_dyn = 0;
  uint64_t launches = 0;
  double compile_ms = 0;
};
struct Pack {
  uint64_t G = 0;
  std::unordered_map<std::string, Kernel *> by_name;
  std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> syms;
};

id<MTLDevice> dev;
id<MTLCommandQueue> queue;
id<MTLResidencySet> resident;
id<MTLFence> fence;
id<MTLCommandBuffer> cb, last_cb;
id<MTLComputeCommandEncoder> enc;
int in_cb = 0;
struct Alloc { id<MTLBuffer> buf; size_t size; };
// Module constructors of the programs call into this file before its own static constructors have
// run: what has a constructor lives here, made at first use.
struct State {
  std::recursive_mutex big;
  std::unordered_map<const void *, Kernel *> kernels;
  std::unordered_map<const void *, std::pair<Pack *, std::string>> vars;
  std::map<uint64_t, Alloc> allocs;    // by address
  std::map<uint64_t, size_t> small_of; // a small allocation's address -> its class
  std::vector<uint64_t> free_small[32];
};
State &st() { static State *s = new State; return *s; }
#define big (st().big)
#define kernels (st().kernels)
#define vars (st().vars)
#define allocs (st().allocs)
#define small_of (st().small_of)
#define free_small (st().free_small)
uint64_t arena_at = 0, arena_left = 0;
size_t used_bytes = 0, peak_bytes = 0;
uint32_t *control = nullptr; // trap word, grid barrier, locks
thread_local cudaError_t last_error = cudaSuccess;
struct Config { dim3 g, b; size_t s; void *st; };
thread_local std::vector<Config> configs;
id<MTLComputePipelineState> ps_fill, ps_copy, ps_gemm;
bool verbose = false;
uint64_t scratch_addr = 0, scratch_size = 0;
std::atomic<uint64_t> n_dispatch{0}, n_sync{0};
double compile_total_ms = 0;

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

[[noreturn]] void die(const char *what, const char *more = "") {
  fprintf(stderr, "metal runtime: %s %s\n", what, more);
  fflush(stderr);
  _exit(70);
}

id<MTLLibrary> compile(const std::string &src, NSError **e) {
  MTLCompileOptions *o = [MTLCompileOptions new];
  o.languageVersion = MTLLanguageVersion3_2;
  o.mathMode = MTLMathModeSafe;
  o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  return [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:e];
}
id<MTLComputePipelineState> pipeline_of(id<MTLLibrary> l, NSString *name, unsigned threads) {
  NSError *e = nil;
  id<MTLFunction> f = [l newFunctionWithName:name];
  if (!f) die("no function", name.UTF8String);
  MTLComputePipelineDescriptor *d = [MTLComputePipelineDescriptor new];
  d.computeFunction = f;
  if (threads) d.maxTotalThreadsPerThreadgroup = threads;
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithDescriptor:d options:MTLPipelineOptionNone reflection:nil error:&e];
  if (!p) die("pipeline:", e.localizedDescription.UTF8String);
  return p;
}
void start() {
  if (dev) return;
  std::lock_guard<std::recursive_mutex> g(big);
  if (dev) return;
  verbose = verbose_now();
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  if (!d) die("no Metal device");
  queue = [d newCommandQueue];
  MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
  rd.initialCapacity = 256;
  NSError *e = nil;
  resident = [d newResidencySetWithDescriptor:rd error:&e];
  if (!resident) die("residency set:", e.localizedDescription.UTF8String);
  [queue addResidencySet:resident];
  fence = [d newFence];
  dev = d;
  id<MTLLibrary> l = compile(mtl_builtin_text, &e);
  if (!l) die("builtin kernels:", e.localizedDescription.UTF8String);
  ps_fill = pipeline_of(l, @"mtl_fill", 0);
  ps_copy = pipeline_of(l, @"mtl_copy", 0);
  ps_gemm = pipeline_of(l, @"mtl_gemm_tn_i8", 0);
  void *p = nullptr;
  mtl_raw_alloc(&p, 65536);
  control = (uint32_t *)p;
  memset(control, 0, 65536);
}
void open_encoder() {
  if (enc) return;
  cb = [queue commandBufferWithUnretainedReferences];
  enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
  [enc waitForFence:fence];
  in_cb = 0;
}
void commit() {
  if (!enc) return;
  [enc updateFence:fence];
  [enc endEncoding];
  [cb commit];
  last_cb = cb;
  enc = nil;
  cb = nil;
}
void check_done(id<MTLCommandBuffer> c) {
  if (c.status == MTLCommandBufferStatusError) die("GPU error:", c.error.localizedDescription.UTF8String);
}
void sync_all() {
  std::lock_guard<std::recursive_mutex> g(big);
  n_sync++;
  commit();
  if (last_cb) {
    [last_cb waitUntilCompleted];
    check_done(last_cb);
    last_cb = nil;
  }
  if (control && control[0]) {
    fprintf(stderr, "metal runtime: a kernel trapped\n");
    fflush(stderr);
    _exit(71);
  }
}
void after_dispatch() {
  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  n_dispatch++;
  if (++in_cb >= 128) commit();
}
void fill_gpu(uint64_t dst, int v, size_t n) {
  struct { uint64_t d, n; uint32_t v, pad; } a = {dst, n, (uint32_t)(v & 255) * 0x01010101u, 0};
  open_encoder();
  [enc setComputePipelineState:ps_fill];
  [enc setBytes:&a length:sizeof a atIndex:0];
  const uint64_t words = (n + 15) / 16, tg = std::min<uint64_t>((words + 255) / 256, 65535);
  [enc dispatchThreadgroups:MTLSizeMake(std::max<uint64_t>(tg, 1), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  after_dispatch();
}
void copy_gpu(uint64_t dst, uint64_t src, size_t n) {
  struct { uint64_t d, s, n; } a = {dst, src, n};
  open_encoder();
  [enc setComputePipelineState:ps_copy];
  [enc setBytes:&a length:sizeof a atIndex:0];
  const uint64_t tg = std::min<uint64_t>((n / 16 + 256) / 256, 65535);
  [enc dispatchThreadgroups:MTLSizeMake(std::max<uint64_t>(tg, 1), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  after_dispatch();
}
bool is_ours(const void *p) {
  const uint64_t a = (uint64_t)p;
  auto it = allocs.upper_bound(a);
  if (it == allocs.begin()) return false;
  --it;
  return a < it->first + it->second.size;
}
Kernel *kernel_of(const void *f) {
  auto it = kernels.find(f);
  if (it == kernels.end()) die("launch of a kernel with no source");
  return it->second;
}
id<MTLComputePipelineState> get_pso(Kernel *k, int on_device, unsigned threads) {
  id<MTLComputePipelineState> p = k->pso[on_device];
  if (p && p.maxTotalThreadsPerThreadgroup >= threads) return p;
  const double t0 = now_ms();
  std::string s;
  s.reserve(k->len + 65536);
  if (on_device) s += "#define SMEM_DEVICE 1\n";
  std::string body(k->src, k->len);
  const size_t at = body.find("//PRELUDE\n");
  s += body.substr(0, at);
  s += mtl_prelude_text;
  s += body.substr(at + 10);
  NSError *e = nil;
  id<MTLLibrary> l = compile(s, &e);
  if (!l) {
    if (const char *dump = dump_path()) { FILE *f = fopen(dump, "w"); if (f) { fwrite(s.data(), 1, s.size(), f); fclose(f); } }
    die(k->name.c_str(), e.localizedDescription.UTF8String);
  }
  p = pipeline_of(l, @"K", 0);
  if (trace_now()) fprintf(stderr, "  compile %.0f ms\n", now_ms() - t0);
  if (p.maxTotalThreadsPerThreadgroup < threads) p = pipeline_of(l, @"K", threads);
  if (p.maxTotalThreadsPerThreadgroup < threads) die("too many threads for", k->name.c_str());
  k->pso[on_device] = p;
  const double ms = now_ms() - t0;
  k->compile_ms += ms;
  compile_total_ms += ms;
  if (verbose) fprintf(stderr, "metal runtime: compiled %s in %.0f ms (%zu bytes)\n", k->name.c_str(), ms, s.size());
  return p;
}
uint64_t get_scratch(uint64_t n) {
  if (n > scratch_size) {
    sync_all();
    if (scratch_addr) cudaFree((void *)scratch_addr);
    void *p = nullptr;
    if (mtl_raw_alloc(&p, n)) die("no memory for shared-memory scratch");
    scratch_addr = (uint64_t)p;
    scratch_size = n;
  }
  return scratch_addr;
}
cudaError_t launch(const void *f, dim3 g, dim3 b, void **args, size_t dyn) {
  start();
  std::lock_guard<std::recursive_mutex> lock(big);
  Kernel *k = kernel_of(f);
  const uint64_t threads = (uint64_t)b.x * b.y * b.z, blocks = (uint64_t)g.x * g.y * g.z;
  if (!threads || !blocks) return cudaErrorInvalidValue;
  const uint64_t smem = (uint64_t)k->dyn_at + dyn;
  const int on_device = smem > 32768 ? 1 : 0;
  id<MTLComputePipelineState> p = get_pso(k, on_device, (unsigned)threads);
  uint8_t small[4096];
  std::vector<uint8_t> big_params;
  uint8_t *pb = small;
  if (k->param_size > sizeof small) { big_params.resize(k->param_size); pb = big_params.data(); }
  for (size_t i = 0; i < k->params.size(); i++) memcpy(pb + k->params[i].first, args[i], k->params[i].second);
  LaunchInfo li = {g.x, g.y, g.z, b.x, b.y, b.z, (uint32_t)dyn, 0, k->pack->G, (uint64_t)control, (uint64_t)(control + 4), (uint64_t)(control + 16), 0, 0};
  if (on_device) {
    li.stride = (smem + 255) & ~255ull;
    li.scratch = get_scratch(li.stride * blocks);
  }
  open_encoder();
  [enc setComputePipelineState:p];
  if (k->param_size > sizeof small) {
    id<MTLBuffer> t = [dev newBufferWithBytes:pb length:k->param_size options:MTLResourceStorageModeShared];
    [enc setBuffer:t offset:0 atIndex:0];
    [enc useResource:t usage:MTLResourceUsageRead];
    commit(); // the unretained command buffer must see the buffer live
    sync_all();
    open_encoder();
  } else {
    [enc setBytes:pb length:std::max<uint32_t>(k->param_size, 16) atIndex:0];
  }
  [enc setBytes:&li length:sizeof li atIndex:1];
  if (!on_device) [enc setThreadgroupMemoryLength:std::max<uint64_t>((smem + 15) & ~15ull, 16) atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(g.x, g.y, g.z) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  k->launches++;
  after_dispatch();
  if (trace_now()) fprintf(stderr, "launch %s %u %u %u x %u %u %u dyn %zu\n", k->name.c_str(), g.x, g.y, g.z, b.x, b.y, b.z, dyn);
  if (sync_each_now()) { const double t0 = now_ms(); sync_all(); if (trace_now()) fprintf(stderr, "  ran %.2f ms\n", now_ms() - t0); }
  return cudaSuccess;
}
} // namespace

// ---- memory
int mtl_raw_alloc(void **out, size_t n) {
  std::lock_guard<std::recursive_mutex> g(big);
  n = (n + 16383) & ~(size_t)16383;
  id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
  if (!b) {
    if (verbose) fprintf(stderr, "metal runtime: no buffer of %zu bytes (in use: %zu)\n", n, used_bytes);
    return 1;
  }
  mach_vm_address_t target = b.gpuAddress;
  vm_prot_t cur, max;
  const kern_return_t r = mach_vm_remap(mach_task_self(), &target, n, 0, VM_FLAGS_FIXED, mach_task_self(), (mach_vm_address_t)b.contents, FALSE, &cur, &max, VM_INHERIT_SHARE);
  if (r != KERN_SUCCESS) die("mach_vm_remap failed");
  [resident addAllocation:b];
  [resident commit];
  allocs[target] = Alloc{b, n};
  *out = (void *)target;
  used_bytes += n;
  peak_bytes = std::max(peak_bytes, used_bytes);
  return 0;
}
static cudaError_t alloc(void **p, size_t n) {
  start();
  std::lock_guard<std::recursive_mutex> g(big);
  if (n == 0) n = 1;
  if (n <= (1u << 20)) { // a small one: power-of-two classes in a shared arena
    int c = 8;
    while ((1ull << c) < n) c++;
    if (!free_small[c].empty()) {
      *p = (void *)free_small[c].back();
      free_small[c].pop_back();
    } else {
      if (arena_left < (1ull << c)) {
        void *a = nullptr;
        if (mtl_raw_alloc(&a, 64u << 20)) return last_error = cudaErrorMemoryAllocation;
        arena_at = (uint64_t)a;
        arena_left = 64u << 20;
      }
      *p = (void *)arena_at;
      arena_at += 1ull << c;
      arena_left -= 1ull << c;
    }
    small_of[(uint64_t)*p] = c;
    return cudaSuccess;
  }
  if (mtl_raw_alloc(p, n)) { *p = nullptr; return last_error = cudaErrorMemoryAllocation; }
  return cudaSuccess;
}
static cudaError_t release(void *p) {
  if (!p) return cudaSuccess;
  sync_all();
  std::lock_guard<std::recursive_mutex> g(big);
  auto s = small_of.find((uint64_t)p);
  if (s != small_of.end()) {
    free_small[s->second].push_back((uint64_t)p);
    small_of.erase(s);
    return cudaSuccess;
  }
  auto it = allocs.find((uint64_t)p);
  if (it == allocs.end()) return cudaErrorInvalidValue;
  mach_vm_deallocate(mach_task_self(), it->first, it->second.size);
  [resident removeAllocation:it->second.buf];
  [resident commit];
  used_bytes -= it->second.size;
  allocs.erase(it);
  return cudaSuccess;
}
size_t mtl_peak_bytes() { return peak_bytes; }

// ---- exit handlers (see metal_compat.h)
static std::vector<std::pair<mtl_exit_fn, void *>> &exit_list_() { static auto *v = new std::vector<std::pair<mtl_exit_fn, void *>>; return *v; }
#define exit_list exit_list_()
int mtl_exit_status = 0;
int mtl_second_start = 0;
static bool env_verbose, env_sync_each, env_trace;
static std::string *env_dump;
__attribute__((constructor)) static void see_second_start() {
  if (getenv("MTL_SECOND_START")) mtl_second_start = 1;
  env_verbose = getenv("MTL_VERBOSE");
  env_sync_each = getenv("MTL_SYNC_EACH");
  env_trace = getenv("MTL_TRACE");
  if (const char *d = getenv("MTL_DUMP")) env_dump = new std::string(d);
}
bool verbose_now() { return env_verbose; }
bool sync_each_now() { return env_sync_each; }
bool trace_now() { return env_trace; }
const char *dump_path() { return env_dump ? env_dump->c_str() : nullptr; }
// What the system adds on its own, and the backend's switches, are no part of the reduced environment.
void mtl_scrub_environment() {
  if (!mtl_second_start) return;
  extern char **environ;
  std::vector<std::string> gone;
  for (char **e = environ; e && *e; e++)
    if (!strncmp(*e, "MTL_", 4) || !strncmp(*e, "__CF", 4)) gone.push_back(std::string(*e, strchr(*e, '=') - *e));
  for (auto &n : gone) unsetenv(n.c_str());
}
static void run_exit_list() {
  while (!exit_list.empty()) {
    auto [f, a] = exit_list.back();
    exit_list.pop_back();
    f(mtl_exit_status, a);
  }
}
int mtl_on_exit_add(mtl_exit_fn f, void *a) {
  static bool once = (atexit(run_exit_list), true);
  (void)once;
  exit_list.push_back({f, a});
  return 0;
}

extern "C" {
const char *cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : e == cudaErrorMemoryAllocation ? "out of memory" : "error"; }
const char *cudaGetErrorName(cudaError_t e) { return cudaGetErrorString(e); }
cudaError_t cudaGetLastError() { const cudaError_t e = last_error; last_error = cudaSuccess; return e; }
cudaError_t cudaPeekAtLastError() { return last_error; }
cudaError_t cudaGetDevice(int *d) { *d = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaGetDeviceCount(int *n) { *n = 1; return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int) {
  start();
  memset(p, 0, sizeof *p);
  strncpy(p->name, dev.name.UTF8String, 255);
  p->totalGlobalMem = dev.recommendedMaxWorkingSetSize;
  p->sharedMemPerBlock = 32768;
  p->sharedMemPerBlockOptin = 232448;
  p->sharedMemPerMultiprocessor = 233472;
  p->warpSize = 32;
  p->maxThreadsPerBlock = 1024;
  p->maxThreadsPerMultiProcessor = 2048;
  p->multiProcessorCount = 20;
  p->major = 9;
  p->minor = 0;
  p->clockRate = 1500000;
  p->canMapHostMemory = 1;
  p->unifiedAddressing = 1;
  p->managedMemory = 1;
  p->concurrentManagedAccess = 1;
  return cudaSuccess;
}
cudaError_t cudaDeviceGetAttribute(int *v, cudaDeviceAttr a, int) {
  *v = a == cudaDevAttrMultiProcessorCount ? 20 : a == cudaDevAttrClockRate ? 1500000 : a == cudaDevAttrMaxSharedMemoryPerBlockOptin ? 232448 : 0;
  return cudaSuccess;
}
cudaError_t cudaDeviceGetPCIBusId(char *s, int n, int) { strncpy(s, "0000:00:00.0", n); return cudaSuccess; }
cudaError_t cudaDeviceGetStreamPriorityRange(int *l, int *h) { if (l) *l = 0; if (h) *h = 0; return cudaSuccess; }
cudaError_t cudaDeviceSynchronize() { start(); sync_all(); return cudaSuccess; }
cudaError_t cudaDeviceSetLimit(cudaLimit, size_t) { return cudaSuccess; }
cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *m, int) { *m = (cudaMemPool_t)1; return cudaSuccess; }
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t, cudaMemPoolAttr a, void *v) { *(uint64_t *)v = a == cudaMemPoolAttrReleaseThreshold ? 0 : used_bytes; return cudaSuccess; }
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t, cudaMemPoolAttr, void *) { return cudaSuccess; }
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t, size_t) { return cudaSuccess; }
cudaError_t cudaMallocFromPoolAsync(void **p, size_t n, cudaMemPool_t, cudaStream_t) { return alloc(p, n); }
cudaError_t cudaFreeAsync(void *p, cudaStream_t) { return release(p); }
cudaError_t cudaMalloc(void **p, size_t n) { return alloc(p, n); }
cudaError_t cudaMallocHost(void **p, size_t n) { return alloc(p, n); }
cudaError_t cudaMallocManaged(void **p, size_t n, unsigned) { return alloc(p, n); }
cudaError_t cudaMallocPitch(void **p, size_t *pitch, size_t w, size_t h) { *pitch = (w + 255) & ~(size_t)255; return alloc(p, *pitch * h); }
cudaError_t cudaHostAlloc(void **p, size_t n, unsigned) { return alloc(p, n); }
cudaError_t cudaHostRegister(void *, size_t, unsigned) { return cudaSuccess; }
cudaError_t cudaHostUnregister(void *) { return cudaSuccess; }
cudaError_t cudaHostGetDevicePointer(void **d, void *h, unsigned) {
  std::lock_guard<std::recursive_mutex> g(big);
  if (!is_ours(h)) return last_error = cudaErrorInvalidValue;
  *d = h;
  return cudaSuccess;
}
cudaError_t cudaFree(void *p) { return release(p); }
cudaError_t cudaFreeHost(void *p) { return release(p); }
cudaError_t cudaMemGetInfo(size_t *fr, size_t *total) {
  start();
  // One allocation is one buffer, so no more than a buffer's limit counts as the GPU's memory.
  const size_t t = std::min<size_t>(dev.recommendedMaxWorkingSetSize, dev.maxBufferLength - (1ull << 30));
  if (total) *total = t;
  if (fr) *fr = t > used_bytes ? t - used_bytes : 0;
  return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, cudaMemcpyKind, cudaStream_t) {
  if (!n) return cudaSuccess;
  start();
  std::lock_guard<std::recursive_mutex> g(big);
  if (is_ours(d) && is_ours(s)) { copy_gpu((uint64_t)d, (uint64_t)s, n); return cudaSuccess; }
  sync_all();
  memcpy(d, s, n);
  return cudaSuccess;
}
cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind k) {
  if (!n) return cudaSuccess;
  start();
  sync_all();
  memcpy(d, s, n);
  return cudaSuccess;
}
cudaError_t cudaMemcpy2DAsync(void *d, size_t dp, const void *s, size_t sp, size_t w, size_t h, cudaMemcpyKind, cudaStream_t) {
  start();
  sync_all();
  for (size_t i = 0; i < h; i++) memcpy((char *)d + i * dp, (const char *)s + i * sp, w);
  return cudaSuccess;
}
cudaError_t cudaMemsetAsync(void *p, int v, size_t n, cudaStream_t) {
  if (!n) return cudaSuccess;
  start();
  std::lock_guard<std::recursive_mutex> g(big);
  fill_gpu((uint64_t)p, v, n);
  return cudaSuccess;
}
cudaError_t cudaMemset(void *p, int v, size_t n) { cudaMemsetAsync(p, v, n, 0); sync_all(); return cudaSuccess; }
cudaError_t cudaMemset2DAsync(void *p, size_t pitch, int v, size_t w, size_t h, cudaStream_t) {
  for (size_t i = 0; i < h; i++) cudaMemsetAsync((char *)p + i * pitch, v, w, 0);
  return cudaSuccess;
}
cudaError_t cudaStreamCreate(cudaStream_t *s) { *s = (cudaStream_t)16; return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned) { *s = (cudaStream_t)16; return cudaSuccess; }
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *s, unsigned, int) { *s = (cudaStream_t)16; return cudaSuccess; }
cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { start(); sync_all(); return cudaSuccess; }
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }
cudaError_t cudaStreamGetFlags(cudaStream_t, unsigned *f) { *f = 0; return cudaSuccess; }
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus *s) { *s = cudaStreamCaptureStatusNone; return cudaSuccess; }
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { die("graphs are not supported"); }
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t *) { die("graphs are not supported"); }
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t, unsigned long long) { die("graphs are not supported"); }
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { die("graphs are not supported"); }
cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
struct CUevent_st { double t; };
cudaError_t cudaEventCreate(cudaEvent_t *e) { *e = new CUevent_st{0}; return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned) { return cudaEventCreate(e); }
cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) { e->t = now_ms(); return cudaSuccess; }
cudaError_t cudaEventRecordWithFlags(cudaEvent_t e, cudaStream_t s, unsigned) { return cudaEventRecord(e, s); }
cudaError_t cudaEventSynchronize(cudaEvent_t) { start(); sync_all(); return cudaSuccess; }
cudaError_t cudaEventQuery(cudaEvent_t) { start(); sync_all(); return cudaSuccess; }
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) { *ms = (float)(b->t - a->t); return cudaSuccess; }
cudaError_t cudaFuncSetAttribute(const void *, cudaFuncAttribute, int) { return cudaSuccess; }
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *a, const void *f) {
  memset(a, 0, sizeof *a);
  Kernel *k = kernel_of(f);
  a->sharedSizeBytes = k->statics;
  a->maxThreadsPerBlock = k->maxntid ? k->maxntid : 1024;
  a->numRegs = 32;
  return cudaSuccess;
}
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *n, const void *, int, size_t) { *n = 1; return cudaSuccess; }
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *n, const void *, int, size_t, unsigned) { *n = 1; return cudaSuccess; }
cudaError_t cudaLaunchKernel(const void *f, dim3 g, dim3 b, void **a, size_t s, cudaStream_t) { return launch(f, g, b, a, s); }
cudaError_t cudaLaunchCooperativeKernel(const void *f, dim3 g, dim3 b, void **a, size_t s, cudaStream_t) { return launch(f, g, b, a, s); }
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *c, const void *f, void **a) { return launch(f, c->gridDim, c->blockDim, a, c->dynamicSmemBytes); }
static uint64_t var_addr(const void *h) {
  auto it = vars.find(h);
  if (it == vars.end()) die("unknown device variable");
  return it->second.first->G + it->second.first->syms[it->second.second].first;
}
cudaError_t cudaGetSymbolAddress(void **p, const void *s) { *p = (void *)var_addr(s); return cudaSuccess; }
cudaError_t cudaMemcpyToSymbol(const void *s, const void *src, size_t n, size_t off, cudaMemcpyKind) { start(); sync_all(); memcpy((char *)var_addr(s) + off, src, n); return cudaSuccess; }
cudaError_t cudaMemcpyFromSymbol(void *d, const void *s, size_t n, size_t off, cudaMemcpyKind) { start(); sync_all(); memcpy(d, (char *)var_addr(s) + off, n); return cudaSuccess; }
unsigned __cudaPushCallConfiguration(dim3 g, dim3 b, size_t s, void *st) { configs.push_back({g, b, s, st}); return 0; }
cudaError_t __cudaPopCallConfiguration(dim3 *g, dim3 *b, size_t *s, void *st) {
  Config c = configs.back();
  configs.pop_back();
  *g = c.g; *b = c.b; *s = c.s; *(void **)st = c.st;
  return cudaSuccess;
}
struct Wrapper { int magic, version; const char *data; void *unused; };
void **__cudaRegisterFatBinary(void *w) {
  start();
  const char *d = ((Wrapper *)w)->data;
  if (memcmp(d, "MTLPACK1", 8)) die("not a pack");
  uint64_t n[3];
  memcpy(n, d + 8, 24);
  const char *idx = d + 32, *gdata = idx + n[0], *blob = gdata + n[1];
  Pack *p = new Pack;
  std::lock_guard<std::recursive_mutex> g(big);
  if (n[1]) {
    void *a = nullptr;
    if (mtl_raw_alloc(&a, n[1])) die("no memory for globals");
    p->G = (uint64_t)a;
    memcpy(a, gdata, n[1]);
  }
  std::string text(idx, n[0]);
  size_t at = 0;
  while (at < text.size()) {
    const size_t e = text.find('\n', at);
    char *line = &text[at];
    text[e] = 0;
    if (!strncmp(line, "sym ", 4)) {
      char name[2048];
      unsigned long long o, sz;
      sscanf(line + 4, "%2047s %llu %llu", name, &o, &sz);
      p->syms[name] = {o, sz};
    } else if (!strncmp(line, "reloc ", 6)) {
      unsigned long long o, v;
      sscanf(line + 6, "%llu %llu", &o, &v);
      *(uint64_t *)(p->G + o) = p->G + v;
    } else if (!strncmp(line, "kernel ", 7)) {
      static char name[1 << 16], ps[1 << 16];
      unsigned long long off, len;
      unsigned st, dy, mx, psz;
      sscanf(line + 7, "%65535s %llu %llu %u %u %u %u %65535s", name, &off, &len, &st, &dy, &mx, &psz, ps);
      Kernel *k = new Kernel;
      k->name = name; k->pack = p; k->src = blob + off; k->len = len; k->statics = st; k->dyn_at = dy; k->maxntid = mx; k->param_size = psz;
      if (strcmp(ps, "-")) {
        char *q = ps;
        while (*q) {
          unsigned o = strtoul(q, &q, 10);
          unsigned s = strtoul(q + 1, &q, 10);
          k->params.push_back({o, s});
          if (*q == ',') q++;
        }
      }
      p->by_name[k->name] = k;
    }
    at = e + 1;
  }
  return (void **)p;
}
void __cudaRegisterFatBinaryEnd(void **) {}
void __cudaUnregisterFatBinary(void **) {}
int __cudaRegisterFunction(void **h, const void *fn, char *, const char *name, int, void *, void *, void *, void *, int *) {
  Pack *p = (Pack *)h;
  auto it = p->by_name.find(name);
  if (it == p->by_name.end()) {
    if (verbose_now()) fprintf(stderr, "metal runtime: kernel %s has no source\n", name);
    return 0;
  }
  kernels[fn] = it->second;
  return 0;
}
void __cudaRegisterVar(void **h, char *var, char *, const char *name, int, size_t, int, int) { vars[var] = {(Pack *)h, name}; }
void __cudaRegisterManagedVar(void **, void **, char *, const char *, int, size_t, int, int) {}
// ---- the driver API
CUresult cuGetErrorString(CUresult, const char **s) { *s = "error"; return 0; }
CUresult cuEventElapsedTime(float *ms, CUevent a, CUevent b) { return cudaEventElapsedTime(ms, a, b); }
CUresult cuMemAddressReserve(CUdeviceptr *p, size_t n, size_t, CUdeviceptr, unsigned long long) { void *q = nullptr; const cudaError_t e = alloc(&q, n); *p = (CUdeviceptr)q; return e; }
CUresult cuMemAddressFree(CUdeviceptr p, size_t) { return release((void *)p); }
CUresult cuMemCreate(CUmemGenericAllocationHandle *h, size_t, const CUmemAllocationProp *, unsigned long long) { *h = 1; return 0; }
CUresult cuMemRelease(CUmemGenericAllocationHandle) { return 0; }
CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long) { return 0; }
CUresult cuMemUnmap(CUdeviceptr, size_t) { return 0; }
CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t) { return 0; }
CUresult cuMemGetAllocationGranularity(size_t *g, const CUmemAllocationProp *, int) { *g = 2u << 20; return 0; }
// ---- integer products
struct cublasContext { int unused; };
cublasStatus_t cublasCreate(cublasHandle_t *h) { *h = new cublasContext; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasDestroy(cublasHandle_t h) { delete h; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetWorkspace(cublasHandle_t, void *, size_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) { return CUBLAS_STATUS_SUCCESS; }
static cublasStatus_t gemm_tn(int m, int n, int k, int alpha, const void *A, int lda, long long sa, const void *B, int ldb, long long sb, int beta, void *C, int ldc, long long sc, int batch) {
  std::lock_guard<std::recursive_mutex> g(big);
  struct { uint64_t a, b, c; int64_t sa, sb, sc; int32_t m, n, k, lda, ldb, ldc, alpha, beta; } p = {(uint64_t)A, (uint64_t)B, (uint64_t)C, sa, sb, sc, m, n, k, lda, ldb, ldc, alpha, beta};
  open_encoder();
  [enc setComputePipelineState:ps_gemm];
  [enc setBytes:&p length:sizeof p atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake((m + 63) / 64, (n + 63) / 64, batch) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
  after_dispatch();
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void *al, const void *A, cudaDataType, int lda, const void *B, cudaDataType, int ldb, const void *be, void *C, cudaDataType, int ldc, cublasComputeType_t, cublasGemmAlgo_t) {
  if (ta != CUBLAS_OP_T || tb != CUBLAS_OP_N) return CUBLAS_STATUS_NOT_SUPPORTED;
  return gemm_tn(m, n, k, *(const int *)al, A, lda, 0, B, ldb, 0, *(const int *)be, C, ldc, 0, 1);
}
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void *al, const void *A, cudaDataType, int lda, long long sa, const void *B, cudaDataType, int ldb, long long sb, const void *be, void *C, cudaDataType, int ldc, long long sc, int batch, cublasComputeType_t, cublasGemmAlgo_t) {
  if (ta != CUBLAS_OP_T || tb != CUBLAS_OP_N) return CUBLAS_STATUS_NOT_SUPPORTED;
  return gemm_tn(m, n, k, *(const int *)al, A, lda, sa, B, ldb, sb, *(const int *)be, C, ldc, sc, batch);
}
cublasStatus_t cublasDgemm(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const double *al, const double *A, int lda, const double *B, int ldb, const double *be, double *C, int ldc) {
  sync_all(); // double precision: the CPU's, on the same memory
  cblas_dgemm(CblasColMajor, ta == CUBLAS_OP_T ? CblasTrans : CblasNoTrans, tb == CUBLAS_OP_T ? CblasTrans : CblasNoTrans, m, n, k, *al, A, lda, B, ldb, *be, C, ldc);
  return CUBLAS_STATUS_SUCCESS;
}
struct cublasLtContext { int unused; };
struct cublasLtMatmulDescOpaque { int unused; };
struct cublasLtLayoutOpaque { uint64_t rows, cols; int64_t ld; int batch; int64_t stride; };
cublasStatus_t cublasLtCreate(cublasLtHandle_t *h) { *h = new cublasLtContext; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *d, cublasComputeType_t, cudaDataType) { *d = new cublasLtMatmulDescOpaque; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *, size_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *l, cudaDataType, uint64_t r, uint64_t c, int64_t ld) { *l = new cublasLtLayoutOpaque{r, c, ld, 1, 0}; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t l, cublasLtMatrixLayoutAttribute_t a, const void *v, size_t) {
  if (a == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT) l->batch = *(const int *)v;
  if (a == CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET) l->stride = *(const int64_t *)v;
  return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *p) { *p = nullptr; return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void *, size_t) { return CUBLAS_STATUS_SUCCESS; }
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t *r, int *got) {
  memset(r, 0, sizeof *r);
  *got = 1;
  return CUBLAS_STATUS_SUCCESS;
}
// As FlashProve calls it: A and B stored k x m and k x n with TRANSA set, C m x n, all column-major.
cublasStatus_t cublasLtMatmul(cublasLtHandle_t, cublasLtMatmulDesc_t, const void *al, const void *A, cublasLtMatrixLayout_t la, const void *B, cublasLtMatrixLayout_t lb, const void *be, const void *, cublasLtMatrixLayout_t lc, void *D, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *, size_t, cudaStream_t) {
  return gemm_tn((int)la->cols, (int)lb->cols, (int)la->rows, *(const int *)al, A, (int)la->ld, la->stride, B, (int)lb->ld, lb->stride, *(const int *)be, D, (int)lc->ld, lc->stride, la->batch);
}
} // extern "C"
