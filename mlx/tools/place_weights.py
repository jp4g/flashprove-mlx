#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Writes the committed matrix W (docs/format/11-5-weights-in-the-table.md): one signed byte per cell.
usage: place_weights.py <directory of make_reference.py> <lk> <D>  -> <directory>/matrix.i8, matrix.txt"""
import sys, pathlib
import numpy as np, mlx.core as mx

d = pathlib.Path(sys.argv[1]); lk = int(sys.argv[2]); D = int(sys.argv[3])
P = {}; layers = []
for line in open(d / "params.txt"):
    t = line.split()
    if t[0] == "layer": layers.append(int(t[-1]))
    else: P[t[0]] = int(t[1])
W = {k: np.array(v) for k, v in mx.load(str(d / "weights.safetensors")).items()}
L, dd, dh, f, fe, NE, V, H, G = (P[k] for k in ("L", "d", "dh", "f", "fe", "NE", "V", "n_h", "n_g"))
dq, dkv = P["dq"], P["dkv"]

def clog2(x): return (x - 1).bit_length()
def cut(groups):
    """Column bits of a unit's groups: (cb0, t, z, lambda) for each."""
    left, cb, out = lk, 0, []
    for M, cls in groups:
        if cls == "A":
            t = clog2(M) if clog2(M) <= left else 0
            lam, z = (1 if t else M), 0
        else:
            z = (M & -M).bit_length() - 1
            t = min(z, left); lam = M >> t
        out.append((cb, t, z, lam)); left -= t; cb += t
    return out
def split(v, M, cls, t, z):
    if cls == "low": return v % (1 << t), v >> t
    if cls == "high":
        zz = z - t
        return (v >> zz) % (1 << t), ((v >> z) << zz) + (v % (1 << zz))
    return (v, 0 * v) if t else (0 * v, v)
def digits(a, nd):
    a = a.astype(np.int64); out = []
    for q in range(nd):
        low = ((a + 128) % 256) - 128
        out.append(low); a = (a - low) >> 8
    assert not a.any(), "a value does not fit its digits"
    return out
units = []
for l in range(L):
    S = f".{l}"
    qk = np.zeros((64, dh, dd), np.int64)
    qk[:H] = W["Wq" + S].reshape(H, dh, dd); qk[H:H + G] = W["Wk" + S].reshape(G, dh, dd)
    units.append(([(64, "A"), (dh, "A"), (dd, "low")], qk, 2))
    units.append(([(dd, "low"), (dkv, "high")], W["Wv" + S].T, 2))
    units.append(([(dq, "low"), (dd, "high")], W["Wo" + S].T, 2))
    if layers[l]:
        units.append(([(dd, "low"), (fe, "high")], W["Wg" + S].T, 2))
        units.append(([(dd, "low"), (fe, "high")], W["Wu" + S].T, 2))
        for n in ("Wgr", "Wur"):
            t = np.stack([W[f"{n}{S}.{e}"] for e in range(NE)])  # e, x, kk
            units.append(([(dd, "low"), (NE, "low"), (fe, "high")], t.transpose(2, 0, 1), 2))
        units.append(([(fe, "low"), (dd, "high")], W["Wd" + S].T, 2))
        t = np.stack([W[f"Wdr{S}.{e}"] for e in range(NE)])  # e, x, kk
        units.append(([(fe, "low"), (NE, "low"), (dd, "high")], t.transpose(2, 0, 1), 2))
        units.append(([(dd, "low"), (NE, "high")], W["Wr" + S].T, 2))
    else:
        units.append(([(dd, "low"), (f, "high")], W["Wg" + S].T, 2))
        units.append(([(dd, "low"), (f, "high")], W["Wu" + S].T, 2))
        units.append(([(f, "low"), (dd, "high")], W["Wd" + S].T, 2))
units.append(([(V, "A"), (dd, "high")], W["Emb"], 4))
units.append(([(V, "A"), (dd, "low")], W["Wh"], 2))
rows = [u[2] * int(np.prod([c[3] for c in cut(u[0])])) for u in units]
total = sum(rows); m0 = (total + D - 1) // D * D
M = np.zeros((m0, 1 << lk), np.int8)
off = 0
for (groups, tensor, nd), r in zip(units, rows):
    cs = cut(groups); n = len(groups)
    c = 0; y = 0; mult = 1
    for g, ((Mx, cls), (cb0, t, z, lam)) in enumerate(zip(groups, cs)):
        shape = [1] * n; shape[g] = Mx
        v = np.arange(Mx, dtype=np.int64).reshape(shape)
        col, rho = split(v, Mx, cls, t, z)
        c = c + (col << cb0); y = y + rho * mult; mult *= lam
    for q, cell in enumerate(digits(tensor, nd)):
        M[off + y + mult * q, c] = cell.astype(np.int8)
    off += r
M.tofile(d / "matrix.i8")
(d / "matrix.txt").write_text(f"{m0} {lk}\n")
print(f"matrix: {len(units)} units, {total} rows, m0 {m0}, 2^{lk} columns")
