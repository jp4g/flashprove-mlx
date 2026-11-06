// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// BLAKE2b's compression function for Metal, as blocks/hash/blake2b_f.h has it (RFC 7693).
#pragma once
constant ulong B2_IV[8] = {0x6a09e667f3bcc908ul,0xbb67ae8584caa73bul,0x3c6ef372fe94f82bul,0xa54ff53a5f1d36f1ul,
                           0x510e527fade682d1ul,0x9b05688c2b3e6c1ful,0x1f83d9abfb41bd6bul,0x5be0cd19137e2179ul};
inline ulong b2_rotr(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }
// Every index is a literal, so the state and the block stay in registers.
#define B2_G(x,y,a,b,c,d) \
  a = a + b + m[x]; d = b2_rotr(d ^ a, 32); c = c + d; b = b2_rotr(b ^ c, 24); \
  a = a + b + m[y]; d = b2_rotr(d ^ a, 16); c = c + d; b = b2_rotr(b ^ c, 63);
#define B2_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15) \
  B2_G(s0,s1,v0,v4,v8,v12) B2_G(s2,s3,v1,v5,v9,v13) B2_G(s4,s5,v2,v6,v10,v14) B2_G(s6,s7,v3,v7,v11,v15) \
  B2_G(s8,s9,v0,v5,v10,v15) B2_G(s10,s11,v1,v6,v11,v12) B2_G(s12,s13,v2,v7,v8,v13) B2_G(s14,s15,v3,v4,v9,v14)
inline void b2_compress(thread ulong* h, thread const ulong* m, ulong t_low, ulong t_high, bool last) {
  ulong v0 = h[0], v1 = h[1], v2 = h[2], v3 = h[3], v4 = h[4], v5 = h[5], v6 = h[6], v7 = h[7];
  ulong v8 = B2_IV[0], v9 = B2_IV[1], v10 = B2_IV[2], v11 = B2_IV[3], v12 = B2_IV[4] ^ t_low, v13 = B2_IV[5] ^ t_high;
  ulong v14 = last ? ~B2_IV[6] : B2_IV[6], v15 = B2_IV[7];
  B2_ROUND(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
  B2_ROUND(14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3)
  B2_ROUND(11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4)
  B2_ROUND(7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8)
  B2_ROUND(9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13)
  B2_ROUND(2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9)
  B2_ROUND(12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11)
  B2_ROUND(13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10)
  B2_ROUND(6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5)
  B2_ROUND(10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0)
  B2_ROUND(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
  B2_ROUND(14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3)
  h[0] ^= v0 ^ v8; h[1] ^= v1 ^ v9; h[2] ^= v2 ^ v10; h[3] ^= v3 ^ v11;
  h[4] ^= v4 ^ v12; h[5] ^= v5 ^ v13; h[6] ^= v6 ^ v14; h[7] ^= v7 ^ v15;
}
// The first chaining value for a digest of 32 bytes; start: BLAKE2b's salt and personalization.
inline void b2_begin(thread ulong* h, thread const ulong* start) {
  h[0] = B2_IV[0] ^ 0x01010020ul; h[1] = B2_IV[1]; h[2] = B2_IV[2]; h[3] = B2_IV[3];
  for (int i = 0; i < 4; i++) h[4 + i] = B2_IV[4 + i] ^ start[i];
}
