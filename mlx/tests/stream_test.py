#!/usr/bin/env python3
# Feeds a stored proof to the companion through the shared stream, a few KB at a time, and counts the
# equations it exports. usage: stream_test.py <accepted-proof directory> <companion> <scratch dir>
import mmap, os, struct, subprocess, sys, time
d, comp, tmp = sys.argv[1:4]
PROOF_AT, EQ_AT, SIZE, MARK = 4096, 1 << 30, 1 << 33, 1048583
path = tmp + "/stream.bin"
fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600); os.ftruncate(fd, SIZE)
m = mmap.mmap(fd, SIZE)
proof = open(d + "/example.proof", "rb").read()
open(tmp + "/mark.proof", "wb").write(proof[:12] + b"\0" * (MARK - 12))
env = dict(os.environ, FP_STREAM=path)
p = subprocess.Popen([comp, d + "/config", d + "/ivk", d + "/example.req", d + "/example.y", tmp + "/mark.proof", "-id", open(d + "/ID.txt").read().strip()],
                     env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
at = 0
while at < len(proof):
    n = min(4096, len(proof) - at)
    m[PROOF_AT + at:PROOF_AT + at + n] = proof[at:at + n]
    at += n
    struct.pack_into("<Q", m, 0, at)
    time.sleep(0.0002)
struct.pack_into("<Q", m, 8, 1)
out, _ = p.communicate(timeout=300)
print("last line:", out.strip().splitlines()[-1], "| exit", p.returncode)
print("equations exported:", struct.unpack_from("<Q", m, 24)[0], "bytes:", struct.unpack_from("<Q", m, 16)[0])
