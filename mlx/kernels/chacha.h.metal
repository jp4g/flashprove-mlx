// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// ChaCha20's block, blocks/coins/chacha.h: RFC 8439, 2.3.
#pragma once
inline uint cc_rotl(uint x, uint n) { return (x << n) | (x >> (32 - n)); }
#define CC_QR(a, b, c, d) a += b; d = cc_rotl(d ^ a, 16); c += d; b = cc_rotl(b ^ c, 12); a += b; d = cc_rotl(d ^ a, 8); c += d; b = cc_rotl(b ^ c, 7);
// The first eight words of the block: the four lanes of a call.
inline void chacha_lanes(thread ulong *lane, thread const uint *key, uint i12, uint i13, uint i14, uint i15) {
  const uint c0 = 0x61707865u, c1 = 0x3320646eu, c2 = 0x79622d32u, c3 = 0x6b206574u;
  uint x0 = c0, x1 = c1, x2 = c2, x3 = c3, x4 = key[0], x5 = key[1], x6 = key[2], x7 = key[3];
  uint x8 = key[4], x9 = key[5], x10 = key[6], x11 = key[7], x12 = i12, x13 = i13, x14 = i14, x15 = i15;
  for (int r = 0; r < 10; r++) {
    CC_QR(x0, x4, x8, x12) CC_QR(x1, x5, x9, x13) CC_QR(x2, x6, x10, x14) CC_QR(x3, x7, x11, x15)
    CC_QR(x0, x5, x10, x15) CC_QR(x1, x6, x11, x12) CC_QR(x2, x7, x8, x13) CC_QR(x3, x4, x9, x14)
  }
  x0 += c0; x1 += c1; x2 += c2; x3 += c3; x4 += key[0]; x5 += key[1]; x6 += key[2]; x7 += key[3];
  lane[0] = (ulong)x0 | ((ulong)x1 << 32); lane[1] = (ulong)x2 | ((ulong)x3 << 32);
  lane[2] = (ulong)x4 | ((ulong)x5 << 32); lane[3] = (ulong)x6 | ((ulong)x7 << 32);
}
