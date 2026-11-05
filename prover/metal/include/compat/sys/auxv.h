// Linux's <sys/auxv.h>, for macOS.
#pragma once
#define AT_EXECFN 31
#define AT_SECURE 23
extern int mtl_second_start;
void mtl_scrub_environment();
static inline unsigned long getauxval(unsigned long w) { mtl_scrub_environment(); return w == AT_EXECFN ? (unsigned long)(mtl_second_start ? "/proc/self/exe" : "first") : 0; }
