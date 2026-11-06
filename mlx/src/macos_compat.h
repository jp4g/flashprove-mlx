// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// getrandom(2) for macOS, where blocks/ asks for it: the system's getentropy().
#pragma once
#include <sys/random.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef __cplusplus
static inline ssize_t getrandom(void *p, size_t n, unsigned) {
  size_t done = 0;
  while (done < n) {
    const size_t k = n - done > 256 ? 256 : n - done;
    if (getentropy((char *)p + done, k)) return done ? (ssize_t)done : -1;
    done += k;
  }
  return (ssize_t)n;
}
#include <string.h>
// CPU affinity: macOS has none to set; the tests' helpers read how many CPUs they may use.
#define CPU_SETSIZE 1024
typedef struct { unsigned long bits[CPU_SETSIZE / 64]; } cpu_set_t;
#define CPU_ZERO(s) memset((s), 0, sizeof(cpu_set_t))
#define CPU_SET(i, s) ((s)->bits[(i) / 64] |= 1ul << ((i) % 64))
#define CPU_ISSET(i, s) (((s)->bits[(i) / 64] >> ((i) % 64)) & 1)
static inline int mlx_cpu_count(const cpu_set_t *s) { int n = 0; for (int i = 0; i < CPU_SETSIZE / 64; i++) n += __builtin_popcountl(s->bits[i]); return n; }
#define CPU_COUNT(s) mlx_cpu_count(s)
static inline int sched_getaffinity(pid_t, size_t, cpu_set_t *s) { CPU_ZERO(s); long n = sysconf(_SC_NPROCESSORS_ONLN); for (long i = 0; i < n && i < CPU_SETSIZE; i++) CPU_SET(i, s); return 0; }
static inline int sched_setaffinity(pid_t, size_t, const cpu_set_t *) { return 0; }
#endif
