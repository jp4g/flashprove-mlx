// Linux's <malloc.h>, for macOS.
#pragma once
#include <stdlib.h>
#include <malloc/malloc.h>
#ifdef __cplusplus
extern "C++" { static inline size_t malloc_usable_size(void *p) { return malloc_size(p); } }
#else
static inline size_t malloc_usable_size(void *p) { return malloc_size(p); }
#endif
