#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Translates the PTX that clang makes from FlashProve's CUDA sources into Metal Shading Language.

usage: ptx2msl.py <input.ptx> <output.pack>

Each kernel becomes one self-contained MSL function with the functions that it calls. Every PTX
instruction becomes the C expression with the same bits, so the arithmetic of the kernels is the
arithmetic of the sources; nothing is approximated. The instructions Metal has no counterpart of
(carry chains, mma.sync, ldmatrix, double precision) are written out in metal_prelude.metal.

PTX control flow is a graph; MSL has no goto. The graph is rebuilt as nested loops and blocks by
the algorithm of N. Ramsey, "Beyond Relooper" (ICFP 2022), which needs a reducible graph (the
translator stops on any other).

Memory. Global pointers are the numbers that the runtime hands out, which mean the same on the CPU
and the GPU. A generic address carries a tag in its top bits (SH_TAG, LOC_TAG) for shared and local
memory. Local memory of a thread is one stack: each function has a frame at offset fp in it.

The pack (read by metal_runtime.mm): prover/metal/README.md, "The pack".
"""
import re
import struct
import sys
from collections import defaultdict

sys.setrecursionlimit(1000000)

TY = {
    'b8': 'uchar', 'u8': 'uchar', 's8': 'char', 'b16': 'ushort', 'u16': 'ushort', 's16': 'short',
    'f16': 'ushort', 'b32': 'uint', 'u32': 'uint', 's32': 'int', 'f32': 'float', 'b64': 'ulong',
    'u64': 'ulong', 's64': 'long', 'f64': 'ulong', 'pred': 'bool',
}
SIZE = {'b8': 1, 'u8': 1, 's8': 1, 'b16': 2, 'u16': 2, 's16': 2, 'f16': 2, 'b32': 4, 'u32': 4,
        's32': 4, 'f32': 4, 'b64': 8, 'u64': 8, 's64': 8, 'f64': 8, 'b128': 16}
BITS = {k: v * 8 for k, v in SIZE.items()}
SPECIAL = {
    '%tid.x': 'c.tx', '%tid.y': 'c.ty', '%tid.z': 'c.tz', '%ntid.x': 'c.bx', '%ntid.y': 'c.by',
    '%ntid.z': 'c.bz', '%ctaid.x': 'c.cx', '%ctaid.y': 'c.cy', '%ctaid.z': 'c.cz',
    '%nctaid.x': 'c.gx', '%nctaid.y': 'c.gy', '%nctaid.z': 'c.gz', '%laneid': 'c.lane',
    '%warpid': '(c.tidx>>5)', '%clock64': '0ul', '%clock': '0u', '%globaltimer': '0ul',
    '%dynamic_smem_size': 'c.dyn', '%total_smem_size': 'c.dyn', '%nwarpid': '32u', '%smid': '0u',
}
ALIGN_RE = re.compile(r'\.align\s+(\d+)')


class Sym:
    def __init__(self, name, space, size, align, init=None, dyn=False):
        self.name, self.space, self.size, self.align, self.init, self.dyn = name, space, size, align, init, dyn


class Func:
    def __init__(self, name, is_kernel):
        self.name, self.is_kernel = name, is_kernel
        self.params = []  # (name, offset, size)
        self.param_size = 0
        self.ret = None  # (name, size)
        self.maxntid = 0
        self.stmts = []
        self.regs = {}  # prefix -> type
        self.frame = 0
        self.calls = set()
        self.shared_used = set()
        self.body = None
        self.needs = set()


def mangle(n):
    return re.sub(r'[^A-Za-z0-9_]', '_', n)


def parse_decl_params(text, fn):
    off = 0
    for raw in text.split(','):
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r'\.param\s+(.*)$', raw)
        spec = m.group(1)
        a = ALIGN_RE.search(spec)
        name = spec.split()[-1]
        arr = re.search(r'\[(\d+)\]', name)
        ty = re.search(r'\.(b8|b16|b32|b64|u8|u16|u32|u64|s8|s16|s32|s64|f32|f64)\b', spec).group(1)
        if arr:
            size = int(arr.group(1)) * SIZE[ty]
            name = name[:name.index('[')]
            align = int(a.group(1)) if a else SIZE[ty]
        else:
            size = SIZE[ty]
            align = SIZE[ty]
        off = (off + align - 1) // align * align
        fn.params.append((name, off, size))
        off += size
    fn.param_size = off


def strip_comments(text):
    return re.sub(r'//[^\n]*', '', text)


def parse_init(s, sz, unit):
    """{1, 2, ...} of unit-byte numbers, or a string of them."""
    data = bytearray()
    relocs = []
    items = [x.strip() for x in s.strip().strip('{}').split(',') if x.strip()]
    for it in items:
        m = re.match(r'generic\((\S+)\)(?:\+(\d+))?$', it)
        if m or not re.match(r'^-?(0[xX][0-9a-fA-F]+|\d+)U?$', it):
            sym = m.group(1) if m else it
            add = int(m.group(2)) if m and m.group(2) else 0
            relocs.append((len(data), sym, add))
            data += b'\0' * unit
        else:
            v = int(it.rstrip('U'), 0)
            data += (v & ((1 << (8 * unit)) - 1)).to_bytes(unit, 'little')
    data += b'\0' * (sz - len(data))
    return bytes(data), relocs


def parse_module(text):
    text = strip_comments(text)
    syms, funcs = {}, {}
    n, i = len(text), 0
    relocs = {}
    # Statements end in ';' or open a body; a body is matched by braces.
    while True:
        while i < n and text[i] in ' \t\r\n':
            i += 1
        if i >= n:
            break
        j = i
        while j < n and text[j] not in ';{':
            j += 1
        head = text[i:j].strip()
        # `= { ... }` initializers hold a brace inside a declaration
        if text[j:j + 1] == '{' and '=' in head:
            j = text.index(';', j)
            head = text[i:j].strip()
            i = j + 1
            sym_decl(head, syms, relocs)
            continue
        if text[j:j + 1] == ';':
            i = j + 1
            if head.startswith(('.version', '.target', '.address_size', '.file', '.section')):
                continue
            if re.search(r'\.(func|entry)\b', head):
                continue  # a prototype
            sym_decl(head, syms, relocs)
            continue
        # A function body.
        depth, k = 0, j
        while True:
            c = text[k]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = text[j + 1:k]
        i = k + 1
        fn = parse_func_head(head)
        fn.src = body
        funcs[fn.name] = fn
    return syms, funcs, relocs


def sym_decl(head, syms, relocs):
    m = re.match(r'(?:\.(?:visible|weak|extern)\s+)*\.(global|const|shared)\s+(.*)$', head, re.S)
    if not m:
        if head.strip():
            raise SystemExit('ptx2msl: declaration not understood: ' + head[:120])
        return
    space, rest = m.group(1), m.group(2)
    a = ALIGN_RE.search(rest)
    align = int(a.group(1)) if a else 1
    init = None
    if '=' in rest:
        rest, init = rest.split('=', 1)
    t = re.search(r'\.(b8|b16|b32|b64|u8|u16|u32|u64|s8|s16|s32|s64|f32|f64)\b', rest)
    ty = t.group(1)
    tail = rest[t.end():].strip()
    arr = re.match(r'([^\s\[]+)\s*((?:\[\d*\])*)', tail)
    name = arr.group(1)
    dims = re.findall(r'\[(\d*)\]', arr.group(2))
    dyn = bool(dims) and dims[0] == ''
    count = 1
    for d in dims:
        count *= int(d) if d else 0
    size = count * SIZE[ty]
    align = max(align, SIZE[ty] if not dims else 1)
    data = None
    if init is not None:
        data, rel = parse_init(init, size, SIZE[ty])
        if rel:
            relocs[name] = rel
    syms[name] = Sym(name, space, size, align, data, dyn)


def parse_func_head(head):
    is_kernel = '.entry' in head
    h = head
    ret = None
    m = re.search(r'\.func\s*\(\s*\.param\s+(.*?)\)', h, re.S)
    if m and not is_kernel:
        spec = m.group(1)
        arr = re.search(r'func_retval0\[(\d+)\]', spec)
        ty = re.search(r'\.(b8|b16|b32|b64)\b', spec).group(1)
        ret = int(arr.group(1)) if arr else SIZE[ty]
        h = h[:m.start()] + '.func ' + h[m.end():]
    m = re.search(r'\.(?:entry|func)\s+([^\s(]+)', h, re.S)
    fn = Func(m.group(1), is_kernel)
    fn.ret = ret
    pm = re.search(r'\((.*)\)', h[m.end():], re.S)
    if pm:
        parse_decl_params(pm.group(1), fn)
    nt = re.search(r'\.(?:maxntid|reqntid)\s+(\d+)(?:,\s*(\d+))?(?:,\s*(\d+))?', h)
    if nt:
        fn.maxntid = int(nt.group(1)) * int(nt.group(2) or 1) * int(nt.group(3) or 1)
    return fn


def split_operands(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '[{(':
            depth += 1
        elif ch in ']})':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Inst:
    __slots__ = ('pred', 'neg', 'op', 'args')

    def __init__(self, pred, neg, op, args):
        self.pred, self.neg, self.op, self.args = pred, neg, op, args


def tokenize_body(fn):
    """Fills fn.stmts: labels as strings, ('{'), ('}'), and Inst; takes the declarations."""
    src = fn.src
    stmts = []
    i, n = 0, len(src)
    while i < n:
        while i < n and src[i] in ' \t\r\n':
            i += 1
        if i >= n:
            break
        c = src[i]
        if c == '{' or c == '}':
            stmts.append(c)
            i += 1
            continue
        j = i
        while j < n and src[j] not in ';:{}':
            j += 1
        # A label ends in ':'; a statement in ';'. `{` starts an operand list inside a statement.
        if j < n and src[j] == ':':
            lab = src[i:j].strip()
            stmts.append(lab + ':')
            i = j + 1
            continue
        j = src.index(';', i)
        stmts.append(src[i:j].strip())
        i = j + 1
    out = []
    for s in stmts:
        if s in ('{', '}') or s.endswith(':'):
            out.append(s)
            continue
        if s.startswith('.reg'):
            m = re.match(r'\.reg\s+\.(\w+)\s+(.*)$', s, re.S)
            ty = m.group(1)
            for r in m.group(2).split(','):
                r = r.strip()
                pm = re.match(r'(%[A-Za-z_]+)<\d+>', r)
                if pm:
                    fn.regs[pm.group(1)] = ty
                else:
                    fn.regs[r] = ty
            continue
        if s.startswith('.local'):
            m = re.search(r'\.align\s+(\d+)\s+\.b8\s+\S+\[(\d+)\]', s)
            fn.frame = int(m.group(2))
            continue
        if s.startswith('.param'):
            out.append(s)
            continue
        if s.startswith(('.loc', '.pragma', '.shared', '.section')):
            if s.startswith('.shared'):
                sym_decl(s, fn.module_syms, {})
            continue
        pred, neg = None, False
        m = re.match(r'@(!?)(%\w+)\s+(.*)$', s, re.S)
        if m:
            neg, pred, s = m.group(1) == '!', m.group(2), m.group(3)
        parts = s.split(None, 1)
        out.append(Inst(pred, neg, parts[0], split_operands(parts[1]) if len(parts) > 1 else []))
    fn.stmts = out


def regname(r):
    return r[1:].replace('.', '_')


def imm(tok, ty):
    """A number as a literal of type ty."""
    t = tok
    if t.startswith('0f'):
        v = int(t[2:], 16)
        return v
    if t.startswith('0d'):
        return int(t[2:], 16)
    neg = t.startswith('-')
    if neg:
        t = t[1:]
    t = t.rstrip('U')
    v = int(t, 0)
    return -v if neg else v


class Translator:
    def __init__(self, syms, funcs, gl_offsets):
        self.syms, self.funcs, self.gl = syms, funcs, gl_offsets

    # ---- operands
    def addr_of_sym(self, name, fn):
        s = self.syms.get(name)
        if s is None:
            raise KeyError(name)
        if s.space == 'shared':
            fn.shared_used.add(name)
            return f'SHOFF_{mangle(name)}'
        return f'(c.G+{self.gl[name]}ul)'

    def is_num(self, tok):
        return bool(re.match(r'^-?(\d|0[xXfd])', tok))

    def val(self, tok, fn, ty='b64'):
        """Value of an operand, as an expression of the unsigned type of ty's width (raw bits)."""
        bits = BITS[ty]
        if tok.startswith('%'):
            if tok in SPECIAL:
                return SPECIAL[tok]
            return regname(tok)
        if self.is_num(tok):
            v = imm(tok, ty) & ((1 << bits) - 1)
            return f'{v:#x}ul' if bits == 64 else f'{v:#x}u'
        m = re.match(r'([^+]+)(?:\+(\d+))?$', tok)
        name, add = m.group(1), int(m.group(2) or 0)
        if name in self.syms:
            return f'({self.addr_of_sym(name, fn)}+{add}ul)'
        for pn, off, _ in fn.params:
            if pn == name:
                return f'(PARAM_ADDR+{off + add}ul)'
        raise SystemExit(f'ptx2msl: operand {tok} in {fn.name}')

    def rd(self, tok, fn, ty):
        """Operand as an expression of PTX type ty."""
        v = self.val(tok, fn, ty)
        if ty == 'f32':
            return f'as_type<float>((uint){v})'
        if ty == 'f64':
            return f'((ulong){v})'
        return f'(({TY[ty]}){v})'

    def wr(self, dst, expr, ty, fn):
        d = regname(dst)
        dty = fn.regs.get(re.match(r'%[A-Za-z_]+', dst).group(0), fn.regs.get(dst))
        if ty == 'f32':
            return f'{d}=as_type<uint>({expr});'
        return f'{d}=({TY[dty if dty else ty]})({expr});' if dty != 'pred' else f'{d}=({expr});'

    # ---- memory
    def address(self, tok, fn):
        """[...] -> (base expression, constant offset, name of a parameter or None)."""
        inner = tok.strip()[1:-1].strip()
        m = re.match(r'^(.*?)\s*\+\s*(-?\d+)$', inner)
        off = 0
        if m:
            inner, off = m.group(1).strip(), int(m.group(2))
        return inner, off

    def mem_expr(self, space, ty, base, off, fn, volatile=False):
        t = TY[ty] if ty not in ('f32',) else 'uint'
        if ty.startswith('b') or ty.startswith('u'):
            pass
        if base.startswith('%'):
            a = regname(base)
        elif self.is_num(base):
            a = f'{int(base, 0)}ul'
        else:
            a = None
        if a is None:
            s = self.syms.get(base)
            if s is None:
                raise SystemExit(f'ptx2msl: address {base} in {fn.name}')
            if s.space == 'shared':
                fn.shared_used.add(base)
                a = f'SHOFF_{mangle(base)}'
            else:
                a = f'(c.G+{self.gl[base]}ul)'
                space = 'global'
        o = f'+{off}' if off > 0 else (f'{off}' if off < 0 else '')
        v = 'volatile ' if volatile else ''
        if space in ('global', 'const'):
            return f'(*reinterpret_cast<{v}device {t}*>({a}{o}))'
        if space == 'shared':
            return f'SMEM({v}, {t}, ({a}{o}))'
        if space == 'local':
            return f'(*reinterpret_cast<thread {t}*>(c.stk+(uint)({a}{o})))'
        return None, f'({a}{o})', t  # generic: the caller uses gld/gst

    def param_ref(self, base, off, ty, fn, scope):
        t = TY[ty] if ty != 'f32' else 'uint'
        if base.startswith('%'):
            if not fn.is_kernel:  # the copy of the parameters at the start of the frame
                return f'(*reinterpret_cast<thread {t}*>(c.stk+(uint){regname(base)}+{off}))'
            return f'(*reinterpret_cast<constant {t}*>(kp+{regname(base)}+{off}))'
        if base in scope:
            b, o = scope[base]
            return f'(*reinterpret_cast<thread {t}*>({b}+{o + off}))'
        if base == 'func_retval0':
            return f'(*reinterpret_cast<thread {t}*>(R+{off}))'
        for pn, o, _ in fn.params:
            if pn == base:
                sp = 'constant' if fn.is_kernel else 'thread'
                b = 'kp' if fn.is_kernel else 'P'
                return f'(*reinterpret_cast<{sp} {t}*>({b}+{o + off}))'
        raise SystemExit(f'ptx2msl: parameter {base} in {fn.name}')

    # ---- one instruction
    def inst(self, ins, fn, scope):
        op = ins.op
        p = op.split('.')
        a = ins.args
        b = p[0]
        m = getattr(self, 'i_' + b, None)
        if m is None:
            raise SystemExit(f'ptx2msl: no translation for {op} {a} in {fn.name}')
        return m(p, a, fn, scope, op)

    def i_mov(self, p, a, fn, sc, op):
        ty = p[-1]
        d, s = a[0], a[1]
        if d.startswith('{'):  # unpack
            ds = [x.strip() for x in d[1:-1].split(',')]
            w = 64 // len(ds) if ty == 'b64' else 32 // len(ds)
            if ty == 'b128':
                w = 64
            sv = self.val(s, fn, ty)
            o = []
            for i, x in enumerate(ds):
                if x == '_':
                    continue
                o.append(f'{regname(x)}=(ushort)({sv}>>{i * w});' if w == 16 else f'{regname(x)}=(uint)({sv}>>{i * w});' if w == 32 else f'{regname(x)}=(uchar)({sv}>>{i * w});')
            return ' '.join(o)
        if s.startswith('{'):  # pack
            ss = [x.strip() for x in s[1:-1].split(',')]
            w = BITS[ty] // len(ss)
            big = 'ulong' if BITS[ty] == 64 else 'uint'
            e = '|'.join(f'(({big}){self.val(x, fn, "b" + str(w))}<<{i * w})' for i, x in enumerate(ss))
            return self.wr(d, e, ty if ty != 'f32' else 'b32', fn)
        if ty == 'pred':
            return f'{regname(d)}={self.pv(s)};'
        if s == '__local_depot':
            return ''
        if s.startswith('__local_depot'):
            return f'{regname(d)}=fp+{fn.param_copy}ul;'
        v = self.val(s, fn, ty if ty not in ('f32', 'f64') else ('b32' if ty == 'f32' else 'b64'))
        return self.wr(d, v, 'b64' if BITS[ty] == 64 else ('b32' if BITS[ty] == 32 else 'b16'), fn)

    def pv(self, s):
        if s.startswith('!'):
            return '!' + regname(s[1:])
        if s in ('0', '1'):
            return 'true' if s == '1' else 'false'
        return regname(s)

    def arith(self, p, a, fn, f):
        ty = p[-1]
        uty = {'s16': 'u16', 's32': 'u32', 's64': 'u64'}.get(ty, ty)
        return self.wr(a[0], f(self.rd(a[1], fn, uty), self.rd(a[2], fn, uty)), uty, fn)

    def i_add(self, p, a, fn, sc, op):
        if 'cc' in p:
            return self.carry(p, a, fn, True, False)
        if p[-1] == 'f64':
            fn.needs.add('f64')
            return self.wr(a[0], f'sf_add({self.rd(a[1], fn, "f64")},{self.rd(a[2], fn, "f64")},{self.rmode(p)})', 'f64', fn)
        if p[-1] == 'f32':
            return self.wr(a[0], f'{self.rd(a[1], fn, "f32")}+{self.rd(a[2], fn, "f32")}', 'f32', fn)
        return self.arith(p, a, fn, lambda x, y: f'{x}+{y}')

    def rmode(self, p):
        return 1 if 'rp' in p else 2 if 'rm' in p else 3 if 'rz' in p else 0

    def i_sub(self, p, a, fn, sc, op):
        if 'cc' in p:
            return self.carry(p, a, fn, False, False)
        if p[-1] == 'f64':
            fn.needs.add('f64')
            return self.wr(a[0], f'sf_add({self.rd(a[1], fn, "f64")},sf_neg({self.rd(a[2], fn, "f64")}),{self.rmode(p)})', 'f64', fn)
        if p[-1] == 'f32':
            return self.wr(a[0], f'{self.rd(a[1], fn, "f32")}-{self.rd(a[2], fn, "f32")}', 'f32', fn)
        return self.arith(p, a, fn, lambda x, y: f'{x}-{y}')

    def carry(self, p, a, fn, is_add, with_c):
        ty = p[-1]
        uty = 'u64' if BITS[ty] == 64 else 'u32'
        x, y = self.rd(a[1], fn, uty), self.rd(a[2], fn, uty)
        d = regname(a[0])
        w = 'ulong' if uty == 'u64' else 'uint'
        cc_out = 'cc' in p
        if is_add:
            if not with_c:
                s = f'{{{w} t_={x}+{y}; {("CC=t_<" + x + ";") if cc_out else ""} {d}=({TY[fn.regs[re.match(r"%[A-Za-z_]+", a[0]).group(0)]]})t_;}}'
            else:
                s = f'{{{w} u_={x}; {w} t_=u_+{y}; bool c1=t_<u_; {w} v_=t_+({w})CC; {("CC=c1|(v_<t_);") if cc_out else ""} {d}=v_;}}'
        else:
            if not with_c:
                s = f'{{{w} u_={x}; {w} w_={y}; {("CC=u_<w_;") if cc_out else ""} {d}=u_-w_;}}'
            else:
                s = f'{{{w} u_={x}; {w} w_={y}; {w} t_=u_-w_; bool b1=u_<w_; {w} v_=t_-({w})CC; {("CC=b1|(t_<({0})CC);".format(w)) if cc_out else ""} {d}=v_;}}'
        return s

    def i_addc(self, p, a, fn, sc, op):
        return self.carry(p, a, fn, True, True)

    def i_subc(self, p, a, fn, sc, op):
        return self.carry(p, a, fn, False, True)

    def i_mad(self, p, a, fn, sc, op):
        return self.madlike(p, a, fn, False)

    def i_madc(self, p, a, fn, sc, op):
        return self.madlike(p, a, fn, True)

    def madlike(self, p, a, fn, with_c):
        ty = p[-1]
        mode = p[1]  # lo, hi, wide
        d = regname(a[0])
        cc_out = 'cc' in p
        if mode == 'wide':
            s = ty[0] == 's'
            big = 'long' if s else 'ulong'
            src = ('s' if s else 'u') + str(BITS[ty])
            return self.wr(a[0], f'({big}){self.rd(a[1], fn, src)}*({big}){self.rd(a[2], fn, src)}+({big}){self.val(a[3], fn, "b" + str(2 * BITS[ty]))}', 'b' + str(2 * BITS[ty]), fn)
        uty = ty if ty[0] == 's' and mode == 'hi' else ('u32' if BITS[ty] == 32 else 'u64' if BITS[ty] == 64 else 'u16')
        x, y = self.rd(a[1], fn, uty), self.rd(a[2], fn, uty)
        z = self.rd(a[3], fn, 'u32' if BITS[ty] == 32 else 'u64' if BITS[ty] == 64 else 'u16')
        w = {16: 'ushort', 32: 'uint', 64: 'ulong'}[BITS[ty]]
        prod = f'mulhi({x},{y})' if mode == 'hi' else f'{x}*{y}'
        if not (with_c or cc_out):
            return self.wr(a[0], f'({w})({prod})+{z}', ty, fn)
        cin = f'+({w})CC' if with_c else ''
        return (f'{{{w} p_=({w})({prod}); {w} z_={z}; {w} t_=p_+z_; bool c1=t_<p_; {w} v_=t_{cin}; '
                + (f'CC=c1|(v_<t_);' if cc_out else '') + f' {d}=v_;}}')

    def i_mul(self, p, a, fn, sc, op):
        ty = p[-1]
        if ty == 'f64':
            fn.needs.add('f64')
            return self.wr(a[0], f'sf_mul({self.rd(a[1], fn, "f64")},{self.rd(a[2], fn, "f64")},{self.rmode(p)})', 'f64', fn)
        if ty == 'f32':
            return self.wr(a[0], f'{self.rd(a[1], fn, "f32")}*{self.rd(a[2], fn, "f32")}', 'f32', fn)
        x, y = self.rd(a[1], fn, ty), self.rd(a[2], fn, ty)
        if p[1] == 'wide':
            big = {'u32': 'ulong', 's32': 'long', 'u16': 'uint', 's16': 'int'}[ty]
            return self.wr(a[0], f'({big}){x}*({big}){y}', 'b64' if BITS[ty] == 32 else 'b32', fn)
        if p[1] == 'hi':
            if BITS[ty] == 16:
                return self.wr(a[0], f'(((int){x}*(int){y})>>16)', ty, fn)
            return self.wr(a[0], f'mulhi({x},{y})', ty, fn)
        u = 'u' + str(BITS[ty])
        return self.wr(a[0], f'{self.rd(a[1], fn, u)}*{self.rd(a[2], fn, u)}', u, fn)

    def i_div(self, p, a, fn, sc, op):
        ty = p[-1]
        if ty == 'f64':
            fn.needs.add('f64')
            return self.wr(a[0], f'sf_div({self.rd(a[1], fn, "f64")},{self.rd(a[2], fn, "f64")},{self.rmode(p)})', 'f64', fn)
        if ty == 'f32':
            return self.wr(a[0], f'{self.rd(a[1], fn, "f32")}/{self.rd(a[2], fn, "f32")}', 'f32', fn)
        return self.wr(a[0], f'{self.rd(a[1], fn, ty)}/{self.rd(a[2], fn, ty)}', ty, fn)

    def i_rem(self, p, a, fn, sc, op):
        ty = p[-1]
        return self.wr(a[0], f'{self.rd(a[1], fn, ty)}%{self.rd(a[2], fn, ty)}', ty, fn)

    def i_sqrt(self, p, a, fn, sc, op):
        fn.needs.add('f64')
        return self.wr(a[0], f'sf_sqrt({self.rd(a[1], fn, "f64")},{self.rmode(p)})', 'f64', fn)

    def logic(self, p, a, fn, sym):
        ty = p[-1]
        if ty == 'pred':
            return f'{regname(a[0])}={self.pv(a[1])}{sym}{self.pv(a[2])};'
        return self.wr(a[0], f'{self.rd(a[1], fn, ty)}{sym}{self.rd(a[2], fn, ty)}', ty, fn)

    def i_and(self, p, a, fn, sc, op):
        return self.logic(p, a, fn, '&')

    def i_or(self, p, a, fn, sc, op):
        return self.logic(p, a, fn, '|')

    def i_xor(self, p, a, fn, sc, op):
        ty = p[-1]
        if ty == 'pred':
            return f'{regname(a[0])}={self.pv(a[1])}!={self.pv(a[2])};'
        return self.logic(p, a, fn, '^')

    def i_not(self, p, a, fn, sc, op):
        ty = p[-1]
        if ty == 'pred':
            return f'{regname(a[0])}=!{self.pv(a[1])};'
        return self.wr(a[0], f'~{self.rd(a[1], fn, ty)}', ty, fn)

    def i_neg(self, p, a, fn, sc, op):
        ty = p[-1]
        if ty == 'f32':
            return self.wr(a[0], f'as_type<float>({self.val(a[1], fn, "b32")}^0x80000000u)', 'f32', fn)
        if ty == 'f64':
            return self.wr(a[0], f'{self.val(a[1], fn, "b64")}^(1ul<<63)', 'b64', fn)
        u = 'u' + str(BITS[ty])
        return self.wr(a[0], f'0-{self.rd(a[1], fn, u)}', u, fn)

    def i_abs(self, p, a, fn, sc, op):
        ty = p[-1]
        if ty == 'f32':
            return self.wr(a[0], f'as_type<float>({self.val(a[1], fn, "b32")}&0x7FFFFFFFu)', 'f32', fn)
        if ty == 'f64':
            return self.wr(a[0], f'{self.val(a[1], fn, "b64")}&~(1ul<<63)', 'b64', fn)
        x = self.rd(a[1], fn, ty)
        return self.wr(a[0], f'{x}<0?-{x}:{x}', ty, fn)

    def i_shl(self, p, a, fn, sc, op):
        ty = p[-1]
        bits = BITS[ty]
        x = self.rd(a[1], fn, 'u' + str(bits))
        if self.is_num(a[2]):
            n = int(a[2], 0)
            return self.wr(a[0], f'{x}<<{n}' if n < bits else '0', 'u' + str(bits), fn)
        n = self.val(a[2], fn, 'b32')
        return self.wr(a[0], f'(uint){n}<{bits}u?({x}<<(uint){n}):0', 'u' + str(bits), fn)

    def i_shr(self, p, a, fn, sc, op):
        ty = p[-1]
        bits = BITS[ty]
        x = self.rd(a[1], fn, ty)
        signed = ty[0] == 's'
        if self.is_num(a[2]):
            n = min(int(a[2], 0), bits - 1 if signed else bits)
            return self.wr(a[0], f'{x}>>{n}' if n < bits else '0', ty, fn)
        n = self.val(a[2], fn, 'b32')
        if signed:
            return self.wr(a[0], f'{x}>>min((uint){n},{bits - 1}u)', ty, fn)
        return self.wr(a[0], f'(uint){n}<{bits}u?({x}>>(uint){n}):0', ty, fn)

    def i_shf(self, p, a, fn, sc, op):
        lo, hi = self.val(a[1], fn, 'b32'), self.val(a[2], fn, 'b32')
        n = self.val(a[3], fn, 'b32')
        f = 'shf_l' if 'l' in p else 'shf_r'
        return f'{regname(a[0])}={f}({lo},{hi},{n},{1 if "wrap" in p else 0});'

    def i_prmt(self, p, a, fn, sc, op):
        return f'{regname(a[0])}=prmt({self.val(a[1], fn, "b32")},{self.val(a[2], fn, "b32")},{self.val(a[3], fn, "b32")});'

    def i_bfe(self, p, a, fn, sc, op):
        ty = p[-1]
        f = {'u32': 'bfe_u32', 's32': 'bfe_s32', 'u64': 'bfe_u64', 's64': 'bfe_s64'}[ty]
        return self.wr(a[0], f'{f}({self.val(a[1], fn, ty)},{self.val(a[2], fn, "b32")},{self.val(a[3], fn, "b32")})', 'u' + str(BITS[ty]), fn)

    def i_min(self, p, a, fn, sc, op):
        ty = p[-1]
        return self.wr(a[0], f'min({self.rd(a[1], fn, ty)},{self.rd(a[2], fn, ty)})', ty, fn)

    def i_max(self, p, a, fn, sc, op):
        ty = p[-1]
        return self.wr(a[0], f'max({self.rd(a[1], fn, ty)},{self.rd(a[2], fn, ty)})', ty, fn)

    def i_popc(self, p, a, fn, sc, op):
        return f'{regname(a[0])}=(uint)popcount({self.rd(a[1], fn, p[-1])});'

    def i_clz(self, p, a, fn, sc, op):
        return f'{regname(a[0])}=(uint)clz({self.rd(a[1], fn, p[-1])});'

    def i_brev(self, p, a, fn, sc, op):
        return self.wr(a[0], f'reverse_bits({self.rd(a[1], fn, p[-1])})', p[-1], fn)

    def i_setp(self, p, a, fn, sc, op):
        cmp_, ty = p[1], p[-1]
        d = a[0]
        comb = None
        if len(p) == 4:
            comb = p[2]
        x, y = a[1], a[2]
        if ty == 'f64':
            fn.needs.add('f64')
            e = f'sf_cmp_{cmp_}({self.rd(x, fn, ty)},{self.rd(y, fn, ty)})'
        elif ty == 'f32':
            X, Y = self.rd(x, fn, ty), self.rd(y, fn, ty)
            e = {'eq': f'{X}=={Y}', 'ne': f'({X}!={Y}&&{X}=={X}&&{Y}=={Y})', 'lt': f'{X}<{Y}', 'le': f'{X}<={Y}', 'gt': f'{X}>{Y}',
                 'ge': f'{X}>={Y}', 'equ': f'!({X}!={Y})||{X}!={X}||{Y}!={Y}', 'neu': f'{X}!={Y}', 'ltu': f'!({X}>={Y})',
                 'leu': f'!({X}>{Y})', 'gtu': f'!({X}<={Y})', 'geu': f'!({X}<{Y})', 'num': f'{X}=={X}&&{Y}=={Y}',
                 'nan': f'{X}!={X}||{Y}!={Y}'}[cmp_]
        else:
            tt = ty
            if ty[0] == 'b':
                tt = 'u' + ty[1:]
            X, Y = self.rd(x, fn, tt), self.rd(y, fn, tt)
            sym = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=', 'lo': '<', 'ls': '<=', 'hi': '>', 'hs': '>='}[cmp_]
            e = f'{X}{sym}{Y}'
        if '|' in d:
            d1, d2 = d.split('|')
            return f'{regname(d1)}=({e}); {regname(d2)}=!{regname(d1)};'
        if comb:
            c = self.pv(a[3])
            return f'{regname(d)}=({e}){"&&" if comb == "and" else "||" if comb == "or" else "!="}{c};'
        return f'{regname(d)}=({e});'

    def i_selp(self, p, a, fn, sc, op):
        ty = p[-1]
        u = 'b' + str(BITS[ty])
        return self.wr(a[0], f'{self.pv(a[3])}?{self.rd(a[1], fn, "u" + str(BITS[ty]))}:{self.rd(a[2], fn, "u" + str(BITS[ty]))}', u, fn)

    def i_cvt(self, p, a, fn, sc, op):
        dty, sty = p[-2], p[-1]
        mods = p[1:-2]
        src = a[1]
        if 'sat' in mods:
            raise SystemExit('ptx2msl: cvt.sat')
        if dty == 'f64' or sty == 'f64':
            fn.needs.add('f64')
            x = self.rd(src, fn, sty)
            if dty == 'f64' and sty == 'f64':
                r = {'rmi': 'sf_floor', 'rzi': 'sf_trunc', 'rni': 'sf_rint', 'rpi': 'sf_ceil'}[mods[0]]
                return self.wr(a[0], f'{r}({x})', 'f64', fn)
            if dty == 'f64':
                if sty == 'f32':
                    return self.wr(a[0], f'sf_from_f32(as_type<uint>({x}))', 'f64', fn)
                sgn = sty[0] == 's'
                f = 'sf_from_i64' if sgn else 'sf_from_u64'
                return self.wr(a[0], f'{f}(({"long" if sgn else "ulong"}){x},{self.rmode(mods)})', 'f64', fn)
            if dty == 'f32':
                return self.wr(a[0], f'as_type<float>(sf_to_f32({x},{self.rmode(mods)}))', 'f32', fn)
            rm = {'rni': 0, 'rzi': 3, 'rmi': 2, 'rpi': 1}[mods[0]]
            f = 'sf_to_i64' if dty[0] == 's' else 'sf_to_u64'
            return self.wr(a[0], f'{f}({x},{rm})', dty, fn)
        if sty == 'f16':
            return self.wr(a[0], f'(float)as_type<half>((ushort){self.val(src, fn, "b16")})', 'f32', fn)
        if dty == 'f16':
            return self.wr(a[0], f'as_type<ushort>((half){self.rd(src, fn, sty)})', 'b16', fn)
        if dty == 'f32' and sty == 'f32':
            r = {'rni': 'rint', 'rzi': 'trunc', 'rmi': 'floor', 'rpi': 'ceil'}.get(mods[0] if mods else '')
            return self.wr(a[0], f'{r}({self.rd(src, fn, "f32")})' if r else self.rd(src, fn, 'f32'), 'f32', fn)
        if dty == 'f32':
            if mods and mods[0] != 'rn':
                raise SystemExit('ptx2msl: cvt ' + '.'.join(p))
            return self.wr(a[0], f'(float){self.rd(src, fn, sty)}', 'f32', fn)
        if sty == 'f32':
            r = {'rni': 'rint', 'rzi': 'trunc', 'rmi': 'floor', 'rpi': 'ceil'}[mods[0]]
            x = self.rd(src, fn, 'f32')
            f = f'f2i_{dty}'
            return self.wr(a[0], f'{f}({r}({x}))', dty, fn)
        return self.wr(a[0], f'({TY[dty]}){self.rd(src, fn, sty)}', dty, fn)

    def i_cvta(self, p, a, fn, sc, op):
        d, s = a[0], a[1]
        if p[1] == 'to':
            space = p[2]
            if space == 'global':
                return self.wr(d, self.val(s, fn, 'b64'), 'b64', fn)
            return self.wr(d, f'{self.val(s, fn, "b64")}&0xFFFFFFFFul', 'b64', fn)
        space = p[1]
        if space == 'global':
            return self.wr(d, self.val(s, fn, 'b64'), 'b64', fn)
        if space == 'shared':
            return self.wr(d, f'{self.val(s, fn, "b64")}|SH_TAG', 'b64', fn)
        if space == 'local':
            return self.wr(d, f'{self.val(s, fn, "b64")}|LOC_TAG', 'b64', fn)
        if space == 'param':
            return self.wr(d, f'{self.val(s, fn, "b64")}|LOC_TAG', 'b64', fn)
        raise SystemExit('ptx2msl: ' + op)

    def spacety(self, p):
        space = None
        vec = 1
        vol = False
        for q in p[1:-1]:
            if q in ('global', 'shared', 'local', 'param', 'const'):
                space = q
            elif q == 'v2':
                vec = 2
            elif q == 'v4':
                vec = 4
            elif q == 'volatile':
                vol = True
        return space, vec, vol, p[-1]

    def i_ld(self, p, a, fn, sc, op):
        space, vec, vol, ty = self.spacety(p)
        base, off = self.address(a[1], fn)
        dsts = [x.strip() for x in a[0][1:-1].split(',')] if a[0].startswith('{') else [a[0]]
        o = []
        for i, d in enumerate(dsts):
            if d == '_':
                continue
            e = off + i * SIZE[ty]
            if space == 'param':
                x = self.param_ref(base, e, ty, fn, sc)
            else:
                x = self.mem_expr(space, ty, base, e, fn, vol)
                if isinstance(x, tuple):
                    x = f'gld<{x[2]}>(c,{x[1]})'
            dty = fn.regs.get(re.match(r'%[A-Za-z_]+', d).group(0), fn.regs.get(d))
            cast = TY[dty]
            o.append(f'{regname(d)}=({cast})({x});')
        return ' '.join(o)

    def i_st(self, p, a, fn, sc, op):
        space, vec, vol, ty = self.spacety(p)
        base, off = self.address(a[0], fn)
        srcs = [x.strip() for x in a[1][1:-1].split(',')] if a[1].startswith('{') else [a[1]]
        o = []
        t = TY[ty] if ty != 'f32' else 'uint'
        for i, s in enumerate(srcs):
            e = off + i * SIZE[ty]
            v = f'({t}){self.val(s, fn, "b" + str(BITS[ty]) if BITS[ty] >= 16 else "b16")}'
            if space == 'param':
                x = self.param_ref(base, e, ty, fn, sc)
                o.append(f'{x}={v};')
            else:
                x = self.mem_expr(space, ty, base, e, fn, vol)
                if isinstance(x, tuple):
                    o.append(f'gst<{x[2]}>(c,{x[1]},{v});')
                else:
                    o.append(f'{x}={v};')
        return ' '.join(o)

    def i_atom(self, p, a, fn, sc, op):
        q = [x for x in p[1:] if x not in ('relaxed', 'acq_rel', 'acquire', 'release', 'sys', 'gpu', 'cta')]
        space = 'global' if 'global' in q else 'shared' if 'shared' in q else 'gen'
        q = [x for x in q if x not in ('global', 'shared')]
        kind, ty = q[0], q[-1]
        base, off = self.address(a[1], fn)
        addr = self.val(base, fn, 'b64') if not base.startswith('%') else regname(base)
        ad = f'({addr}+{off})' if off else addr
        w = BITS[ty]
        args = [self.val(x, fn, 'b' + str(w)) for x in a[2:]]
        f = f'atom_{kind}_{w}_{ty[0]}_{space}'
        fn.needs.add('atom')
        return self.wr(a[0], f'{f}(c,{ad},{",".join(args)})', 'b' + str(w), fn)

    def i_red(self, p, a, fn, sc, op):
        q = [x for x in p[1:] if x not in ('relaxed', 'acq_rel', 'acquire', 'release', 'sys', 'gpu', 'cta')]
        space = 'global' if 'global' in q else 'shared' if 'shared' in q else 'gen'
        q = [x for x in q if x not in ('global', 'shared')]
        kind, ty = q[0], q[-1]
        base, off = self.address(a[0], fn)
        addr = regname(base)
        ad = f'({addr}+{off})' if off else addr
        w = BITS[ty]
        return f'atom_{kind}_{w}_{ty[0]}_{space}(c,{ad},{self.val(a[1], fn, "b" + str(w))});'

    def i_bar(self, p, a, fn, sc, op):
        if 'warp' in p:
            return 'simdgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);'
        if 'red' in p:
            d = regname(a[0])
            q = self.pv(a[-1])
            k = p[2]
            fn.needs.add('barred')
            if k == 'popc':
                return f'{d}=bar_red_popc(c,{q});'
            return f'{d}=bar_red_popc(c,{q})' + ('!=0u;' if k == 'or' else '==(c.bx*c.by*c.bz);')
        return 'BARRIER();'

    def i_barrier(self, p, a, fn, sc, op):
        return self.i_bar(p, a, fn, sc, op)

    def i_membar(self, p, a, fn, sc, op):
        return 'atomic_thread_fence(mem_flags::mem_device|mem_flags::mem_threadgroup,memory_order_seq_cst,thread_scope_device);'

    def i_fence(self, p, a, fn, sc, op):
        return self.i_membar(p, a, fn, sc, op)

    def i_shfl(self, p, a, fn, sc, op):
        mode = p[2]
        d = a[0]
        pd = None
        if '|' in d:
            d, pd = d.split('|')
        s = f'{regname(d)}=shfl_{mode}(c,{self.val(a[1], fn, "b32")},{self.val(a[2], fn, "b32")},{self.val(a[3], fn, "b32")});'
        if pd:
            s += f' {regname(pd)}=true;'
        return s

    def i_vote(self, p, a, fn, sc, op):
        k = p[2]
        if k == 'ballot':
            return f'{regname(a[0])}=(uint)(ulong)simd_ballot({self.pv(a[1])});'
        if k == 'any':
            return f'{regname(a[0])}=simd_any({self.pv(a[1])});'
        if k == 'all':
            return f'{regname(a[0])}=simd_all({self.pv(a[1])});'
        raise SystemExit('ptx2msl: ' + op)

    def i_activemask(self, p, a, fn, sc, op):
        return f'{regname(a[0])}=(uint)(ulong)simd_active_threads_mask();'

    def i_mma(self, p, a, fn, sc, op):
        # mma.sync.aligned.m16n8k32.row.col.s32.{s8,u8}.{s8,u8}.s32 {d}, {a}, {b}, {c}
        sa, sb = p[-3], p[-2]
        R = lambda s: [x.strip() for x in s[1:-1].split(',')]
        d, av, bv, cv = R(a[0]), R(a[1]), R(a[2]), R(a[3])
        V = lambda r: self.val(r, fn, 'b32')
        fn.needs.add('mma')
        return ('{ uint4 d_=mma_m16n8k32_%s%s(c,uint4(%s),uint2(%s),uint4(%s)); %s}' % (
            sa, sb, ','.join(V(x) for x in av), ','.join(V(x) for x in bv), ','.join(V(x) for x in cv),
            ' '.join(f'{regname(x)}=d_[{i}];' for i, x in enumerate(d))))

    def i_ldmatrix(self, p, a, fn, sc, op):
        R = [x.strip() for x in a[0][1:-1].split(',')]
        base, off = self.address(a[1], fn)
        addr = regname(base) if base.startswith('%') else self.val(base, fn, 'b64')
        ad = f'(uint)({addr}+{off})' if off else f'(uint)({addr})'
        fn.needs.add('mma')
        o = [f'{regname(x)}=ldmatrix_part(c,{ad},{i});' for i, x in enumerate(R)]
        return ' '.join(o)

    def i_cp(self, p, a, fn, sc, op):
        raise SystemExit('ptx2msl: ' + op)

    def i_trap(self, p, a, fn, sc, op):
        return 'TRAP();'

    def i_brkpt(self, p, a, fn, sc, op):
        return 'TRAP();'


# ---- control flow


class Block:
    __slots__ = ('id', 'label', 'insts', 'term', 'succ', 'cond')

    def __init__(self, i, label):
        self.id, self.label, self.insts, self.succ = i, label, [], []
        self.term = None  # 'ret' | 'br' | 'cbr' | 'trap'
        self.cond = None


def build_cfg(fn):
    blocks, cur = [], None
    labels = {}

    def new(label):
        nonlocal cur
        b = Block(len(blocks), label)
        blocks.append(b)
        if label:
            labels[label] = b
        cur = b
        return b

    new(None)
    fall = {}
    for s in fn.stmts:
        if isinstance(s, str) and s.endswith(':') and not s.startswith('.'):
            lab = s[:-1]
            old = cur
            if not old.insts and old.term is None and old.label is None and old.id == len(blocks) - 1 and old.id != 0:
                old.label = lab
                labels[lab] = old
                continue
            b = new(lab)
            if old.term is None:
                old.term = 'br'
                old.succ = [b]
            continue
        if isinstance(s, Inst) and s.op.split('.')[0] == 'bra':
            tgt = s.args[0]
            if s.pred:
                cur.term = 'cbr'
                cur.cond = ('!' if s.neg else '') + s.pred
                cur.succ = [tgt, None]
                old = cur
                b = new(None)
                old.succ[1] = b
            else:
                cur.term = 'br'
                cur.succ = [tgt]
                new(None)
            continue
        if isinstance(s, Inst) and s.op.split('.')[0] == 'brx':
            raise SystemExit(f'ptx2msl: brx in {fn.name}: compile with -fno-jump-tables')
        if isinstance(s, Inst) and s.op in ('ret', 'exit', 'ret.uni'):
            if s.pred:
                # A predicated return: a branch to a block of its own.
                old = cur
                r = Block(len(blocks), None)
                blocks.append(r)
                r.term = 'ret'
                cur.term = 'cbr'
                cur.cond = ('!' if s.neg else '') + s.pred
                b = new(None)
                old.succ = [r, b]
            else:
                cur.term = 'ret'
                new(None)
            continue
        if isinstance(s, Inst) and s.op == 'trap' and not s.pred:
            cur.insts.append(s)
            cur.term = 'ret'
            new(None)
            continue
        cur.insts.append(s)
    if cur.term is None:
        cur.term = 'ret'
    for b in blocks:
        b.succ = [labels[x] if isinstance(x, str) else x for x in b.succ]
    return blocks


def structure(fn, blocks, emit_block, cond_expr):
    entry = blocks[0]
    # reverse postorder
    seen, post = set(), []
    stack = [(entry, 0)]
    seen.add(entry.id)
    while stack:
        b, i = stack.pop()
        if i < len(b.succ):
            stack.append((b, i + 1))
            s = b.succ[i]
            if s.id not in seen:
                seen.add(s.id)
                stack.append((s, 0))
        else:
            post.append(b)
    rpo = post[::-1]
    num = {b.id: i for i, b in enumerate(rpo)}
    preds = defaultdict(list)
    for b in rpo:
        for s in b.succ:
            preds[s.id].append(b)
    # dominators (Cooper, Harvey, Kennedy)
    idom = {entry.id: entry}

    def inter(a, b):
        while a is not b:
            while num[a.id] > num[b.id]:
                a = idom[a.id]
            while num[b.id] > num[a.id]:
                b = idom[b.id]
        return a

    ch = True
    while ch:
        ch = False
        for b in rpo[1:]:
            new = None
            for q in preds[b.id]:
                if q.id in idom:
                    new = q if new is None else inter(new, q)
            if idom.get(b.id) is not new:
                idom[b.id] = new
                ch = True

    def dominates(a, b):
        while True:
            if a is b:
                return True
            if b is entry:
                return False
            b = idom[b.id]

    back = set()
    loops = set()
    fwd_in = defaultdict(int)
    for b in rpo:
        for s in b.succ:
            if num[s.id] <= num[b.id]:
                if not dominates(s, b):
                    raise SystemExit(f'ptx2msl: irreducible control flow in {fn.name}')
                back.add((b.id, s.id))
                loops.add(s.id)
            else:
                fwd_in[s.id] += 1
    merge = {b.id for b in rpo if fwd_in[b.id] >= 2}
    kids = defaultdict(list)
    for b in rpo[1:]:
        kids[idom[b.id].id].append(b)

    # constructs: nested C loops/blocks.
    class K:
        __slots__ = ('kind', 'target', 'passing', 'id')
        n = 0

        def __init__(self, kind, target):
            self.kind, self.target, self.passing = kind, target, set()
            K.n += 1
            self.id = K.n

    out = []
    maxdepth = [0]

    def jump(tgt, kind, ctx):
        # kind: 'c' continue the loop of tgt, 'b' leave the block before tgt.
        for i in range(len(ctx) - 1, -1, -1):
            k = ctx[i]
            if k.target == tgt.id and ((kind == 'c' and k.kind == 'loop') or (kind == 'b' and k.kind == 'block')):
                if i == len(ctx) - 1:
                    out.append('continue;' if kind == 'c' else 'break;')
                else:
                    code = k.id * 2 + (1 if kind == 'c' else 0)
                    for m in ctx[i + 1:]:
                        m.passing.add((code, k))
                    out.append(f'J={code}; break;')
                return
        raise SystemExit(f'ptx2msl: lost a jump in {fn.name}')

    def after(k, ctx):
        """After construct k closed, inside ctx."""
        if not k.passing:
            return
        here = ctx[-1] if ctx else None
        out.append('if(J){')
        rest = False
        for code, target in sorted(k.passing, key=lambda x: x[0]):
            if target is here:
                out.append(f'if(J=={code}){{J=0; {"continue" if code & 1 else "break"};}}')
            else:
                rest = True
        if rest:
            out.append('break;')
        out.append('}')

    def branch(src, dst, ctx):
        if (src.id, dst.id) in back:
            jump(dst, 'c', ctx)
        elif dst.id in merge:
            jump(dst, 'b', ctx)
        else:
            do_tree(dst, ctx)

    def within(x, ms, ctx):
        if ms:
            y = ms[0]
            k = K('block', y.id)
            out.append('do{')
            within(x, ms[1:], ctx + [k])
            out.append('}while(false);')
            after(k, ctx)
            do_tree(y, ctx)
            return
        emit_block(x, out)
        if x.term == 'ret':
            out.append('return;')
        elif x.term == 'br':
            branch(x, x.succ[0], ctx)
        else:
            out.append(f'if({cond_expr(x.cond)}){{')
            branch(x, x.succ[0], ctx)
            out.append('}else{')
            branch(x, x.succ[1], ctx)
            out.append('}')

    def do_tree(x, ctx):
        maxdepth[0] = max(maxdepth[0], len(ctx))
        ms = [b for b in kids[x.id] if b.id in merge]
        ms.sort(key=lambda b: -num[b.id])
        if x.id in loops:
            k = K('loop', x.id)
            out.append('while(true){')
            within(x, ms, ctx + [k])
            out.append('}')
            after(k, ctx)
        else:
            within(x, ms, ctx)

    do_tree(entry, [])
    return out, maxdepth[0]


def translate_func(fn, tr):
    fn.module_syms = tr.syms
    tokenize_body(fn)
    # A function that takes the address of a parameter gets a copy of them at the start of its frame.
    fn.param_copy = 0
    if not fn.is_kernel and re.search(r'mov\.b64\s+%\w+,\s*\S+_param_\d+', fn.src):
        fn.param_copy = (fn.param_size + 15) // 16 * 16
        fn.frame = (fn.frame + 15) // 16 * 16 + fn.param_copy
    blocks = build_cfg(fn)
    # Scopes of call sequences: walk statements in order within a block and keep the current scope.
    pmax = [0, 0]
    state = {'scope': {}, 'poff': 0, 'roff': 0}
    used = set()

    def emit_block(b, out):
        sc = state['scope']
        for s in b.insts:
            if s == '{':
                state['scope'] = sc = {}
                state['poff'] = state['roff'] = 0
                continue
            if s == '}':
                continue
            if isinstance(s, str) and s.startswith('.param'):
                m = re.match(r'\.param\s+(.*)$', s, re.S)
                spec = m.group(1)
                a = ALIGN_RE.search(spec)
                name = spec.split()[-1]
                ar = re.search(r'\[(\d+)\]', name)
                ty = re.search(r'\.(b8|b16|b32|b64)\b', spec).group(1)
                if ar:
                    size = int(ar.group(1)) * SIZE[ty]
                    name = name[:name.index('[')]
                    al = int(a.group(1)) if a else SIZE[ty]
                else:
                    size, al = SIZE[ty], SIZE[ty]
                if name.startswith('retval'):
                    sc[name] = ('RB', 0)
                    pmax[1] = max(pmax[1], size)
                else:
                    o = (state['poff'] + al - 1) // al * al
                    sc[name] = ('PB', o)
                    state['poff'] = o + size
                    pmax[0] = max(pmax[0], state['poff'])
                continue
            if isinstance(s, str):
                raise SystemExit('ptx2msl: statement ' + s)
            if s.op.split('.')[0] == 'call':
                args = s.args
                ret = None
                if args[0].startswith('('):
                    i = 1
                    if args[0] != '()':
                        ret = args[0][1:-1].strip()
                else:
                    i = 0
                name = args[i].strip()
                if name == '__mtl_grid_sync':
                    fn.needs.add('gridsync')
                    out.append('grid_sync(c);')
                    continue
                if name in ('vprintf', 'malloc', 'free', '__assertfail'):
                    if name == '__assertfail':
                        out.append('TRAP();')
                    continue
                if name not in tr.funcs:
                    raise SystemExit(f'ptx2msl: call of {name} in {fn.name}')
                fn.calls.add(name)
                out.append(f'F_{mangle(name)}(PBp,RBp,c,fp+{(fn.frame + 15) // 16 * 16});')
                continue
            t = tr.inst(s, fn, sc)
            if s.pred:
                t = f'if({"!" if s.neg else ""}{regname(s.pred)}){{{t}}}'
            if t:
                out.append(t)

    def cond_expr(c):
        return ('!' + regname(c[1:])) if c.startswith('!') else regname(c)

    lines, depth = structure(fn, blocks, emit_block, cond_expr)
    fn.depth = depth
    # register declarations: only names that occur
    text = '\n'.join(lines)
    names = set(re.findall(r'\b(?:rd|rs|rq|r|p|f|fd|h|hh|t|tmp|SP|SPL|SPP)\d*\b', text))
    decl = defaultdict(list)
    for prefix, ty in fn.regs.items():
        pn = prefix[1:]
        if '<' in pn:
            continue
        for nm in names:
            if re.fullmatch(re.escape(pn) + r'\d*', nm) and (nm == pn or prefix in fn.regs and nm[len(pn):].isdigit()):
                decl[ty].append(nm)
    longest = {}
    for prefix, ty in fn.regs.items():
        longest[prefix[1:]] = ty
    decl = defaultdict(set)
    for nm in names:
        m = re.match(r'([A-Za-z]+)', nm)
        if m.group(1) in longest:
            decl[longest[m.group(1)]].add(nm)
    d = []
    for ty, ns in decl.items():
        t = TY.get(ty, 'uint')
        d.append(f'{t} ' + ','.join(sorted(n + ('=false' if t == 'bool' else '=0') for n in ns)) + ';')
    uses_cc = 'CC' in text
    pre = ''.join(x + '\n' for x in d)
    if uses_cc:
        pre += 'bool CC=false;\n'
    if 'J=' in text or '(J)' in text:
        pre += 'int J=0;\n'
    if pmax[0]:
        pre += f'ulong PB_[{(pmax[0] + 7) // 8}]; thread uchar* PB=(thread uchar*)PB_; thread uchar* PBp=PB;\n'
    else:
        pre += 'thread uchar* PBp=(thread uchar*)nullptr;\n'
    if pmax[1]:
        pre += f'ulong RB_[{(pmax[1] + 7) // 8}]; thread uchar* RB=(thread uchar*)RB_; thread uchar* RBp=RB;\n'
    else:
        pre += 'thread uchar* RBp=(thread uchar*)nullptr;\n'
    if fn.param_copy:
        pre += f'for(uint i_=0;i_<{fn.param_size}u;i_++) c.stk[(uint)fp+i_]=P[i_];\n'
        text = text.replace('PARAM_ADDR', 'fp')
    fn.body = pre + text


def collect(fn, tr, seen):
    if fn.name in seen:
        return
    seen[fn.name] = fn
    for c in fn.calls:
        collect(tr.funcs[c], tr, seen)


def stack_need(fn, tr, memo, active=()):
    if fn.name in active:
        raise SystemExit(f'ptx2msl: recursion through {fn.name}')
    if fn.name in memo:
        return memo[fn.name]
    frame = (fn.frame + 15) // 16 * 16
    m = 0
    for c in fn.calls:
        m = max(m, stack_need(tr.funcs[c], tr, memo, active + (fn.name,)))
    memo[fn.name] = frame + m
    return memo[fn.name]


def emit_kernel(k, tr, memo):
    seen = {}
    collect(k, tr, seen)
    order, done = [], set()

    def topo(f):
        if f.name in done:
            return
        done.add(f.name)
        for c in sorted(f.calls):
            topo(tr.funcs[c])
        order.append(f)

    topo(k)
    # shared layout
    used = set()
    needs = set()
    for f in seen.values():
        used |= f.shared_used
        needs |= f.needs
    off = 0
    defs = []
    dyn_align = 16
    dyn_syms = []
    for name in sorted(used):
        s = tr.syms[name]
        if s.dyn:
            dyn_syms.append(name)
            dyn_align = max(dyn_align, s.align)
            continue
        off = (off + s.align - 1) // s.align * s.align
        defs.append(f'#define SHOFF_{mangle(name)} {off}ul')
        off += s.size
    if 'barred' in needs:
        off = (off + 15) // 16 * 16
        defs.append(f'#define BARRED_OFF {off}u')
        off += 16
    static = off
    off = (off + dyn_align - 1) // dyn_align * dyn_align
    dyn_at = off
    for name in dyn_syms:
        defs.append(f'#define SHOFF_{mangle(name)} {off}ul')
    fp0 = 0
    pcopy = 0
    stack = stack_need(k, tr, memo)
    uses_paddr = 'PARAM_ADDR' in k.body
    if uses_paddr:
        pcopy = (k.param_size + 15) // 16 * 16
    total_stack = stack + pcopy
    src = []
    for n in sorted(needs):
        src.append(f'#define NEED_{n.upper()} 1')
    src += defs
    src.append('//PRELUDE')
    for f in order:
        if f is k:
            continue
        src.append(f'__attribute__((noinline)) static void F_{mangle(f.name)}(thread uchar* P, thread uchar* R, thread Ctx& c, ulong fp){{\n{f.body}\n}}')
    src.append(f'''kernel void K(constant uchar* kp [[buffer(0)]], constant LaunchInfo& li [[buffer(1)]],
 uint tidx [[thread_index_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
 SMEM_PARAM){{
 ulong stk_[{max((total_stack + 7) // 8, 1)}];
 Ctx c; ctx_init(c,li,tidx,gid,lane,(thread uchar*)stk_ SMEM_ARG);
 ulong fp={pcopy}ul;
''')
    if uses_paddr:
        src.append(f'for(uint i_=0;i_<{k.param_size}u;i_++) c.stk[i_]=kp[i_];\n#define PARAM_ADDR 0ul')
    src.append('thread uchar* P=(thread uchar*)nullptr; thread uchar* R=(thread uchar*)nullptr;')
    src.append(k.body)
    src.append('}')
    return '\n'.join(src), static, dyn_at


def main():
    path, outp = sys.argv[1], sys.argv[2]
    text = open(path).read()
    syms, funcs, relocs = parse_module(text)
    gl, off = {}, 0
    for s in syms.values():
        if s.space in ('global', 'const'):
            a = max(s.align, 16)
            off = (off + a - 1) // a * a
            gl[s.name] = off
            off += s.size
    gsize = off
    tr = Translator(syms, funcs, gl)
    for f in funcs.values():
        try:
            translate_func(f, tr)
        except KeyError as e:
            raise SystemExit(f'ptx2msl: symbol {e} in {f.name}')
    memo = {}
    entries = []
    blob = bytearray()
    deep = 0
    for f in funcs.values():
        deep = max(deep, f.depth)
    for f in funcs.values():
        if not f.is_kernel:
            continue
        srct, static, dyn_at = emit_kernel(f, tr, memo)
        b = srct.encode()
        entries.append((f.name, len(blob), len(b), static, dyn_at, f.maxntid, f.param_size,
                        ','.join(f'{o}:{s}' for _, o, s in f.params) or '-'))
        blob += b
    idx = [f'kernels {len(entries)}', f'globals {gsize}']
    gdata = bytearray(gsize)
    for s in syms.values():
        if s.name in gl:
            idx.append(f'sym {s.name} {gl[s.name]} {s.size}')
            if s.init:
                gdata[gl[s.name]:gl[s.name] + len(s.init)] = s.init
    for name, rel in relocs.items():
        for at, tgt, add in rel:
            if tgt in gl:
                idx.append(f'reloc {gl[name] + at} {gl[tgt] + add}')
    for e in entries:
        idx.append('kernel ' + ' '.join(str(x) for x in e))
    idx = ('\n'.join(idx) + '\n').encode()
    with open(outp, 'wb') as o:
        o.write(b'MTLPACK1')
        o.write(struct.pack('<QQQ', len(idx), len(gdata), len(blob)))
        o.write(idx)
        o.write(gdata)
        o.write(blob)
    print(f'ptx2msl: {len(entries)} kernels, {len(funcs) - len(entries)} functions, {gsize} bytes of globals, deepest nesting {deep}, {len(blob)} bytes of MSL')


if __name__ == '__main__':
    main()
