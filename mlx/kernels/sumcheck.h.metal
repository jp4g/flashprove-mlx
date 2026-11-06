// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Elements of E in arrays of words, three words each.
#pragma once
inline GlE e_ld(const device ulong *a, ulong i) { return GlE{a[3 * i], a[3 * i + 1], a[3 * i + 2]}; }
inline void e_st(device ulong *a, ulong i, GlE x) { a[3 * i] = x.c0; a[3 * i + 1] = x.c1; a[3 * i + 2] = x.c2; }
inline GlE e_ld(const constant ulong *a, ulong i) { return GlE{a[3 * i], a[3 * i + 1], a[3 * i + 2]}; }
