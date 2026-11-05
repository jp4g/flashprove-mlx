// What the sources take from glibc and Linux and macOS lacks or names otherwise.
#pragma once
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#ifdef __APPLE__
#define __STDC_WANT_LIB_EXT1__ 1
static inline void explicit_bzero(void *p, size_t n) {
  volatile unsigned char *q = (volatile unsigned char *)p;
  while (n--) *q++ = 0;
}
#endif
#ifdef __APPLE__
#include <sys/mman.h>
#ifndef MADV_WIPEONFORK
#define MADV_WIPEONFORK 18 // No counterpart on macOS: see madvise_compat below
#define MADV_DONTDUMP 16
#define MADV_DONTFORK 10
#endif
#define CPU_SETSIZE 1024
typedef struct { unsigned long bits[CPU_SETSIZE / 64]; } cpu_set_t;
#define CPU_ZERO(s) memset((s), 0, sizeof(cpu_set_t))
#define CPU_SET(i, s) ((s)->bits[(i) / 64] |= 1ul << ((i) % 64))
#define CPU_ISSET(i, s) (((s)->bits[(i) / 64] >> ((i) % 64)) & 1)
#define CPU_COUNT(s) __cpu_count(s)
static inline int __cpu_count(const cpu_set_t *s) { int n = 0; for (int i = 0; i < CPU_SETSIZE / 64; i++) n += __builtin_popcountl(s->bits[i]); return n; }
static inline int sched_setaffinity(pid_t, size_t, const cpu_set_t *) { return 0; }
static inline int sched_getaffinity(pid_t, size_t, cpu_set_t *s) { memset(s, 0, sizeof *s); long n = sysconf(_SC_NPROCESSORS_ONLN); for (long i = 0; i < n && i < CPU_SETSIZE; i++) CPU_SET(i, s); return 0; }
#endif
#ifdef __APPLE__
// Linux calls that the sources make, with their nearest macOS meaning. prover/metal/README.md lists
// which protections these leave out.
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/fcntl.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <mach/vm_inherit.h>
#ifdef __cplusplus
extern "C++" {
static inline ssize_t getrandom(void *p, size_t n, unsigned) {
  size_t done = 0;
  while (done < n) { size_t k = n - done > 256 ? 256 : n - done; if (getentropy((char *)p + done, k)) return done ? (ssize_t)done : -1; done += k; }
  return (ssize_t)n;
}
static inline int fdatasync(int fd) { return fsync(fd); }
#define st_mtim st_mtimespec
#define st_atim st_atimespec
#define st_ctim st_ctimespec
#define POSIX_FADV_DONTNEED 4
#define POSIX_FADV_SEQUENTIAL 2
#define POSIX_FADV_WILLNEED 3
static inline int posix_fadvise(int, off_t, off_t, int) { return 0; }
#define MADV_HUGEPAGE 14
#define MADV_NOHUGEPAGE 15
// madvise: a page that a fork wipes becomes a page that a fork does not hand on at all (the child
// faults on it), and advice with no counterpart is taken as done.
static inline int mtl_madvise(void *p, size_t n, int a) {
  if (a == MADV_WIPEONFORK) return minherit(p, n, VM_INHERIT_NONE);
  if (a == MADV_DONTDUMP || a == MADV_DONTFORK || a == MADV_HUGEPAGE || a == MADV_NOHUGEPAGE) return 0;
  return madvise(p, n, a);
}
#define madvise(p, n, a) mtl_madvise((void *)(p), (n), (a))
#define O_PATH O_RDONLY
#define SIGPWR SIGTERM
#define SIGSTKFLT SIGTERM
#define SIGRTMIN 1
#define SIGRTMAX 0
#define SYS_renameat2 100000
#include <stdarg.h>
static inline long mtl_renameat2(int a, const char *f, int b, const char *t, unsigned flags) { return renameatx_np(a, f, b, t, flags & 1 ? RENAME_EXCL : 0); }
#define syscall(n, ...) mtl_renameat2(__VA_ARGS__)
// /proc/self/exe has no counterpart on macOS.
#include <mach-o/dyld.h>
#include <spawn.h>
static inline const char *mtl_self_path() { static char p[4096]; uint32_t n = sizeof p; return _NSGetExecutablePath(p, &n) ? "" : p; }
static inline ssize_t mtl_readlink(const char *a, char *b, size_t n) {
  if (!strcmp(a, "/proc/self/exe")) { const char *s = mtl_self_path(); const size_t l = strlen(s) < n ? strlen(s) : n; memcpy(b, s, l); return (ssize_t)l; }
  return readlink(a, b, n);
}
#define readlink mtl_readlink
static inline int mtl_posix_spawn(pid_t *p, const char *path, const posix_spawn_file_actions_t *a, const posix_spawnattr_t *t, char *const *v, char *const *e) {
  return posix_spawn(p, strcmp(path, "/proc/self/exe") ? path : mtl_self_path(), a, t, v, e);
}
#define posix_spawn mtl_posix_spawn
// The second start of reduced_env_start(): Linux tells it by the name of the file; here a variable,
// which the runtime removes before main().
extern int mtl_second_start;
static inline int mtl_execve(const char *path, char *const *v, char *const *e) {
  if (strcmp(path, "/proc/self/exe")) return execve(path, v, e);
  extern char **environ;
  int n = 0, m = 0;
  while (e[n]) n++;
  for (char **x = environ; *x; x++) m++;
  char **w = (char **)calloc(n + m + 2, sizeof *w);
  for (int i = 0; i < n; i++) w[i] = e[i];
  for (char **x = environ; *x; x++) if (!strncmp(*x, "MTL_", 4)) w[n++] = *x; // the backend's own switches
  w[n] = (char *)"MTL_SECOND_START=1";
  return execve(mtl_self_path(), v, w);
}
#define execve mtl_execve
#define RENAME_NOREPLACE 1
// A file in memory: here an unlinked file; closefrom: descriptors the sources open are close-on-exec.
#define MFD_CLOEXEC 1u
static inline int memfd_create(const char *, unsigned) {
  char name[] = "/tmp/flashprove.XXXXXX";
  const int fd = mkstemp(name);
  if (fd >= 0) { unlink(name); fcntl(fd, F_SETFD, FD_CLOEXEC); }
  return fd;
}
#define posix_spawn_file_actions_addclosefrom_np(a, n) (0)
static inline int renameat2(int a, const char *f, int b, const char *t, unsigned flags) { return (int)mtl_renameat2(a, f, b, t, flags); }
// on_exit: atexit without the status, which the handlers named here do not read on this path.
typedef void (*mtl_exit_fn)(int, void *);
static inline int mtl_on_exit(mtl_exit_fn f, void *a);
#define on_exit mtl_on_exit
}
#include <vector>
#include <mutex>
extern "C++" {
void mtl_run_on_exit_list();
int mtl_on_exit_add(mtl_exit_fn, void *);
static inline int mtl_on_exit(mtl_exit_fn f, void *a) { return mtl_on_exit_add(f, a); }
// exit(): remembers the status for the handlers, which read it.
extern int mtl_exit_status;
[[noreturn]] static inline void mtl_exit(int s) { mtl_exit_status = s; ::exit(s); }
}
#include <cstdlib>
namespace std { using ::mtl_exit; }
#define exit(s) mtl_exit(s)
#endif
#endif
