// Linux's <sys/prctl.h>, for macOS: these calls have no counterpart there and succeed without effect.
// docs on the Metal backend: prover/metal/README.md, "What differs from the Linux programs".
#pragma once
#define PR_SET_DUMPABLE 4
#define PR_SET_PDEATHSIG 1
#define PR_SET_NAME 15
#define PR_GET_DUMPABLE 3
static inline int prctl(int, ...) { return 0; }
