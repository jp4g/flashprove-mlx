#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Runs integrations/vllm/integer_model/model.py, the repository's independent integer reference, on
a kit's synthetic weights and saves weights, tables, tokens and every recorded value for
mlx/tests/forward_test.cpp.  usage: make_reference.py <kit directory> <out directory> [tokens]"""
import sys, pathlib, struct
root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "integrations/vllm/integer_model"))
import numpy as np, torch, mlx.core as mx
from integer_model.parameters import Parameters
from integer_model.model import IntegerModel
from integer_model.synthetic_weights import SyntheticWeights

kit, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
p = Parameters.read(kit / "policy.txt", kit / "KL.bin")
w = SyntheticWeights(p, "cpu")
m = IntegerModel(p, w, "cpu")
req = (pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else kit / "own_choices.req").read_bytes()
T = p.T
tokens = list(struct.unpack("<%dI" % T, req[:4 * T])) if len(req) >= 4 * T else list(range(T))  # T tokens, then r and e
tokens = [t % p.V for t in tokens]
rec = {}
def record(name, layer, t):
    rec[name if layer is None else f"{name}.{layer}"] = t.to(torch.int64).numpy().copy()
seq = m.new_sequence()
choice = m.extend(seq, tokens, record=record)
A = lambda t: mx.array(np.ascontiguousarray(t))
W = {}
for l in range(p.L):
    ex = p.has_experts(l)
    for n in ("Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"):
        W[f"{n}.{l}"] = A(w.matrix(n, l).to(torch.int32).numpy())
    if ex:
        W[f"Wr.{l}"] = A(w.matrix("Wr", l).to(torch.int32).numpy())
        for e in range(p.NE):
            for n in ("Wgr", "Wur", "Wdr"):
                W[f"{n}.{l}.{e}"] = A(w.matrix(n, l, e).to(torch.int32).numpy())
W["Wh"] = A(w.matrix("Wh").to(torch.int32).numpy())
W["Emb"] = A(w.embedding_rows(torch.arange(p.V)).to(torch.int64).numpy())
t = m.tables
for n in ("cos", "sin", "softmax", "silu", "sigmoid"):
    W["table." + n] = A(getattr(t, n).numpy())
W["tokens"] = A(np.array(tokens, dtype=np.int64))
mx.save_safetensors(str(out / "weights.safetensors"), W)
mx.save_safetensors(str(out / "record.safetensors"), {k: A(v) for k, v in rec.items()})
K = ["kq","kk","kv","ko","kg","ku","kd","M1","K1","M2","K2","sh","rp","kr","kgR","kuR"]
with open(out / "params.txt", "w") as f:
    for n in ("T","L","d","n_h","n_g","dh","f","fe","NE","V","cs"):
        f.write(f"{n} {getattr(p, n)}\n")
    f.write(f"dq {p.dq}\ndkv {p.dkv}\neps1 {p.eps1}\neps2 {p.eps2}\nepsF {p.epsF}\nMF {p.MF}\nKF {p.KF}\n")
    f.write(f"gcl {int(p.gate_clamp)}\nrcl {int(p.router_clamp)}\n")
    for l, k in enumerate(p.layers):
        f.write(f"layer {l} " + " ".join(str(getattr(k, n)) for n in K) + f" {int(p.has_experts(l))}\n")
print("reference:", len(rec), "recorded values; first choices end", choice[-3:].tolist())
