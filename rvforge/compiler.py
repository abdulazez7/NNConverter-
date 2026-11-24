"""Static graph compiler. ONNX is an internal interchange, not the user interface.

Each operator is lowered from its shapes and attributes. No architecture matching.
Compressed constants are read directly from packed storage; arithmetic stays FP32.
"""

from __future__ import annotations
import copy, hashlib, json, math, tempfile, shutil, warnings
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper, helper, shape_inference
from onnx.reference import ReferenceEvaluator


class ConversionError(ValueError):
    pass


def size(s):
    return math.prod(s)


def strides(s):
    return [size(s[i + 1 :]) for i in range(len(s))]


def coord(i, s, d):
    return f"(({i})/{strides(s)[d]}%{s[d]})"


def broadcast_index(i, out, inp):
    if len(inp) > len(out):
        raise ConversionError("Broadcast rank mismatch")
    terms = []
    for j, n in enumerate(inp):
        d = len(out) - len(inp) + j
        if n not in (1, out[d]):
            raise ConversionError(f"Cannot broadcast {inp} to {out}")
        if n != 1:
            terms.append(f"{coord(i,out,d)}*{strides(inp)[j]}")
    return "+".join(terms) or "0"


def f32(x):
    x = float(np.float32(x))
    if not math.isfinite(x):
        raise ConversionError("Nonfinite constant or reference output")
    return f"{x:.9e}f"


def arr(a):
    return ",".join(f32(x) for x in np.asarray(a).ravel())


def attr(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


MATH = r"""
/* Freestanding FP32 elementary functions; tested against reference outputs.
   No libc, allocation, or expanded compressed-weight buffers. */
typedef unsigned char rv_u8;
typedef signed char rv_i8;
typedef unsigned int rv_u32;
static float rv_abs(float x){return x<0?-x:x;}
static int rv_finite(float x){union{rv_u32 u;float f;}v;v.f=x;return (v.u&0x7fffffffu)<0x7f800000u;}
static float rv_exp(float x){
  if(x < -103.97208f) return 0.0f;
  if(x > 88.722839f){union{rv_u32 u;float f;}v;v.u=0x7f800000u;return v.f;}
  if(x!=x)return x;
  int n=(int)(x*1.4426950408889634f+(x>=0?0.5f:-0.5f));
  float r=(x-n*0.693359375f)-n*(-0.00021219444005469058f);
  float p=1.0f+r*(1.0f+r*(0.5f+r*(0.1666666667f+r*(0.0416666667f+r*(0.0083333333f+r*(0.0013888889f+r*0.0001984127f))))));
  union {rv_u32 u; float f;} s;
  if(n < -126){s.u=(rv_u32)(n+150)<<23;return (p*s.f)*0.00000011920928955078125f;}
  if(n > 127){s.u=(rv_u32)(n+126)<<23;return (p*s.f)*2;}
  s.u=(rv_u32)(n+127)<<23; return p*s.f;
}
static float rv_sqrt(float x){
  if(x==0)return x;
  if(x<0){union{rv_u32 u;float f;}v;v.u=0x7fc00000u;return v.f;}
  if(!rv_finite(x))return x;
  union {rv_u32 u;float f;} v; v.f=x; v.u=(v.u>>1)+0x1fc00000u;
  float y=v.f; for(int k=0;k<6;k++) y=0.5f*(y+x/y); return y;
}
static float rv_log(float x){
  if(x<=0){union{rv_u32 u;float f;}v;v.u=x==0?0xff800000u:0x7fc00000u;return v.f;}
  if(!rv_finite(x))return x;
  union {rv_u32 u;float f;} v; v.f=x;
  int shift=0;if(x<1.17549435e-38f){v.f=x*8388608.0f;shift=-23;}
  int e=(int)((v.u>>23)&255)-127+shift; v.u=(v.u&0x7fffffu)|0x3f800000u;
  float y=(v.f-1)/(v.f+1), yy=y*y, p=y, s=y;
  for(int k=3;k<=19;k+=2){p*=yy;s+=p/k;} return 2*s+e*0.6931471805599453f;
}
static float rv_tanh(float x){if(x>9)return 1;if(x< -9)return -1;float e=rv_exp(2*x);return (e-1)/(e+1);}
static float rv_erf(float x){float s=x<0?-1:1;x=rv_abs(x);float t=1/(1+0.3275911f*x);float p=(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;return s*(1-p*rv_exp(-x*x));}
"""


class Graph:
    def __init__(self, model, precision):
        self.model = shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        self.precision = precision
        self.constants = {
            i.name: numpy_helper.to_array(i).copy() for i in model.graph.initializer
        }
        self.shapes = {}
        self.types = {}
        self.nodes = []
        self.lines = []
        self.encoded = {}
        self.weight_names = set()
        for v in (
            list(self.model.graph.input)
            + list(self.model.graph.value_info)
            + list(self.model.graph.output)
        ):
            tt = v.type.tensor_type
            if all(d.HasField("dim_value") and d.dim_value > 0 for d in tt.shape.dim):
                self.shapes[v.name] = tuple(d.dim_value for d in tt.shape.dim)
                self.types[v.name] = tt.elem_type
        for name, a in self.constants.items():
            self.shapes[name] = a.shape
        for n in self.model.graph.node:
            if n.domain not in ("", "ai.onnx"):
                raise ConversionError(
                    f"Unsupported operator domain {n.domain} for {n.op_type}"
                )
            at = attr(n)
            if n.op_type == "Shape" and n.input[0] in self.shapes:
                a = np.array(self.shapes[n.input[0]], dtype=np.int64)[
                    at.get("start", 0) : at.get("end", None)
                ]
                self.constants[n.output[0]] = a
                self.shapes[n.output[0]] = a.shape
                continue
            if all(x in self.constants for x in n.input if x):
                try:
                    values = ReferenceEvaluator(n).run(
                        None, {x: self.constants[x] for x in n.input if x}
                    )
                    for key, value in zip(n.output, values):
                        if key:
                            self.constants[key] = np.asarray(value)
                            self.shapes[key] = np.asarray(value).shape
                    continue
                except Exception:
                    pass
            if (
                n.op_type == "Pad"
                and n.input[0] in self.shapes
                and n.input[1] in self.constants
            ):
                dims = self.shapes[n.input[0]]
                pads = self.constants[n.input[1]].ravel()
                if len(pads) == 2 * len(dims):
                    shape = tuple(
                        int(d + pads[j] + pads[j + len(dims)])
                        for j, d in enumerate(dims)
                    )
                    if min(shape) <= 0:
                        raise ConversionError("Pad produces an empty tensor")
                    self.shapes[n.output[0]] = shape
            for key in n.output:
                if key and key not in self.shapes:
                    raise ConversionError(
                        f"{n.op_type} {n.name}: output {key} has a dynamic or unknown shape; supply fixed example inputs"
                    )
            self.nodes.append(n)
            if n.op_type in ("Gemm", "MatMul", "Conv", "Gather"):
                candidates = (
                    n.input[:2]
                    if n.op_type == "MatMul"
                    else [n.input[0] if n.op_type == "Gather" else n.input[1]]
                )
                for key in candidates:
                    if (
                        key in self.constants
                        and self.constants[key].ndim >= 2
                        and self.constants[key].dtype.kind == "f"
                    ):
                        self.weight_names.add(key)
        names = []
        for v in model.graph.input:
            names.append(v.name)
        names += list(self.constants)
        for n in self.nodes:
            names += list(n.output)
        self.ids = {
            n: f"t{j}" for j, n in enumerate(dict.fromkeys(x for x in names if x))
        }
        for n, s in self.shapes.items():
            if len(s) > 8 or size(s) > 2**25:
                raise ConversionError(
                    f"Tensor {n} exceeds rank or element safety limit"
                )
        self.storage_bytes = 0
        self.float_weight_bytes = 0
        for key in self.weight_names:
            a = self.constants[key].astype(np.float32)
            self.float_weight_bytes += a.nbytes
            rows = a.reshape(-1, a.shape[-1])
            width = rows.shape[-1]
            if precision == "int8":
                scales = np.max(np.abs(rows), axis=1) / 127
                scales = np.where(scales == 0, 1, scales).astype(np.float32)
                q = np.clip(np.rint(rows / scales[:, None]), -127, 127).astype(np.int8)
                restored = (q * scales[:, None]).reshape(a.shape).astype(np.float32)
                self.encoded[key] = (q, scales, width, restored)
            elif precision == "binary":
                scales = np.abs(rows).mean(axis=1).astype(np.float32)
                bits = np.packbits((rows >= 0).ravel(), bitorder="little")
                restored = (
                    np.where(rows >= 0, scales[:, None], -scales[:, None])
                    .reshape(a.shape)
                    .astype(np.float32)
                )
                self.encoded[key] = (bits, scales, width, restored)

    def s(self, key):
        return self.shapes[key]

    def read(self, key, i="0"):
        return (
            f"{self.ids[key]}_get({i})"
            if key in self.encoded
            else f"{self.ids[key]}[{i}]"
        )

    def axes(self, n, at, default=None):
        if len(n.input) > 1 and n.input[1]:
            if n.input[1] not in self.constants:
                raise ConversionError("Axes must be constant")
            return [int(x) for x in self.constants[n.input[1]].ravel()]
        return list(at.get("axes", default if default is not None else []))

    def scalar(self, key):
        if key not in self.constants or self.constants[key].size != 1:
            raise ConversionError("Expected constant scalar")
        return float(self.constants[key].ravel()[0])

    def emit(self, n):
        op = n.op_type
        at = attr(n)
        ins = n.input
        out = n.output[0]
        os = self.s(out)
        N = size(os)
        y = self.ids[out]
        if self.types.get(out) in (6, 7) and op not in (
            "Identity",
            "Flatten",
            "Reshape",
            "Squeeze",
            "Unsqueeze",
            "Cast",
            "Transpose",
            "Expand",
            "Concat",
            "Split",
            "Slice",
            "Gather",
            "Pad",
        ):
            raise ConversionError("Runtime integer arithmetic is unsupported for " + op)
        if len([x for x in n.output if x]) > 1 and op not in ("Split",):
            raise ConversionError("Multiple outputs not implemented for " + op)
        x = ins[0] if ins else ""
        xs = self.s(x) if x else ()
        r = lambda k, i="i": self.read(ins[k], i)

        def loop(expr):
            return f"for(int i=0;i<{N};i++) {y}[i]={expr};"

        if op in (
            "Identity",
            "Flatten",
            "Reshape",
            "Squeeze",
            "Unsqueeze",
            "Cast",
            "Dropout",
        ):
            if size(xs) != N:
                raise ConversionError("Element count changes unexpectedly")
            if op == "Cast" and at.get("to") not in (1, 6, 7, 9):
                raise ConversionError("Cast target unsupported")
            if op == "Cast" and at.get("to") in (6, 7):
                return f"for(int i=0;i<{N};i++){{if({r(0)} < -16777216.0f || {r(0)} > 16777216.0f)return -3;{y}[i]=(float)(int)({r(0)});}}"
            if op == "Cast" and at.get("to") == 9:
                return loop(f"({r(0)}!=0)")
            if op == "Dropout" and len(ins) > 2 and self.scalar(ins[2]) != 0:
                raise ConversionError("Training Dropout unsupported")
            return loop(r(0))
        unary = {
            "Relu": lambda a: f"({a}>0?{a}:0)",
            "LeakyRelu": lambda a: f'({a}>0?{a}:{f32(at.get("alpha",.01))}*{a})',
            "Sigmoid": lambda a: f"(1/(1+rv_exp(-({a}))))",
            "Tanh": lambda a: f"rv_tanh({a})",
            "Abs": lambda a: f"rv_abs({a})",
            "Neg": lambda a: f"-({a})",
            "Exp": lambda a: f"rv_exp({a})",
            "Log": lambda a: f"rv_log({a})",
            "Sqrt": lambda a: f"rv_sqrt({a})",
            "Erf": lambda a: f"rv_erf({a})",
            "Softplus": lambda a: f"({a}>20?{a}:rv_log(1+rv_exp({a})))",
            "Elu": lambda a: f'({a}>=0?{a}:{f32(at.get("alpha",1))}*(rv_exp({a})-1))',
        }
        if op in unary:
            return loop(unary[op](r(0)))
        if op == "Clip":
            lo = (
                self.scalar(ins[1])
                if len(ins) > 1 and ins[1]
                else at.get("min", -3.4028234e38)
            )
            hi = (
                self.scalar(ins[2])
                if len(ins) > 2 and ins[2]
                else at.get("max", 3.4028234e38)
            )
            a = r(0)
            return loop(f"({a}<{f32(lo)}?{f32(lo)}:({a}>{f32(hi)}?{f32(hi)}:{a}))")
        binary = {
            "Add": "+",
            "Sub": "-",
            "Mul": "*",
            "Div": "/",
            "Equal": "==",
            "Greater": ">",
            "Less": "<",
            "GreaterOrEqual": ">=",
            "LessOrEqual": "<=",
            "And": "&&",
            "Or": "||",
        }
        if op in binary:
            a = r(0, broadcast_index("i", os, self.s(ins[0])))
            b = r(1, broadcast_index("i", os, self.s(ins[1])))
            return loop(f"({a}{binary[op]}{b})")
        if op in ("Max", "Min") and len(ins) == 2:
            a = r(0, broadcast_index("i", os, self.s(ins[0])))
            b = r(1, broadcast_index("i", os, self.s(ins[1])))
            return loop(f'({a}{">" if op=="Max" else "<"}{b}?{a}:{b})')
        if op == "Pow":
            p = self.scalar(ins[1])
            a = r(0, broadcast_index("i", os, xs))
            if p == 2:
                return loop(f"({a}*{a})")
            if p == 0.5:
                return loop(f"rv_sqrt({a})")
            if p == -1:
                return loop(f"(1/{a})")
            raise ConversionError("Pow supports constant exponents 2, 0.5 and -1")
        if op == "Where":
            a, b, c = [r(k, broadcast_index("i", os, self.s(ins[k]))) for k in range(3)]
            return loop(f"({a}?{b}:{c})")
        if op == "Expand":
            return loop(r(0, broadcast_index("i", os, xs)))
        if op == "Transpose":
            perm = list(at.get("perm", reversed(range(len(xs)))))
            ix = (
                "+".join(
                    f'{coord("i",os,k)}*{strides(xs)[d]}' for k, d in enumerate(perm)
                )
                or "0"
            )
            return loop(r(0, ix))
        if op in ("Gemm", "MatMul"):
            a, b = self.s(ins[0]), self.s(ins[1])
            if min(len(a), len(b)) < 2:
                raise ConversionError("MatMul requires both operands rank at least 2")
            ta = at.get("transA", 0) if op == "Gemm" else 0
            tb = at.get("transB", 0) if op == "Gemm" else 0
            if op == "Gemm" and (len(a) != 2 or len(b) != 2):
                raise ConversionError("Gemm operands must have rank 2")
            M, K = (a[-1], a[-2]) if ta else a[-2:]
            Kb, P = (b[-1], b[-2]) if tb else b[-2:]
            if K != Kb:
                raise ConversionError("MatMul inner dimension mismatch")
            batch = os[:-2]
            ai = broadcast_index(f"i/{M*P}", batch, a[:-2])
            bi = broadcast_index(f"i/{M*P}", batch, b[:-2])
            ai = f"({ai})*{size(a[-2:])}+" + (
                f"k*{M}+i/{P}%{M}" if ta else f"(i/{P}%{M})*{K}+k"
            )
            bi = f"({bi})*{size(b[-2:])}+" + (
                f"(i%{P})*{K}+k" if tb else f"k*{P}+i%{P}"
            )
            end = f's*{f32(at.get("alpha",1))}'
            if len(ins) > 2 and ins[2]:
                end += f'+{f32(at.get("beta",1))}*' + r(
                    2, broadcast_index("i", os, self.s(ins[2]))
                )
            return f"for(int i=0;i<{N};i++){{float s=0;for(int k=0;k<{K};k++)s+={r(0,ai)}*{r(1,bi)};{y}[i]={end};}}"
        if op in ("Softmax", "LogSoftmax"):
            axis = at.get("axis", -1) % len(xs)
            inner = size(xs[axis + 1 :])
            width = xs[axis]
            outer = size(xs[:axis])
            ix = f"(o*{width}+k)*{inner}+j"
            a = r(0, ix)
            val = f"rv_exp({a}-m)/s" if op == "Softmax" else f"({a}-m-rv_log(s))"
            return f"for(int o=0;o<{outer};o++)for(int j=0;j<{inner};j++){{float m=-3.4028234e38f,s=0;for(int k=0;k<{width};k++){{float v={a};if(v>m)m=v;}}for(int k=0;k<{width};k++)s+=rv_exp({a}-m);for(int k=0;k<{width};k++){y}[{ix}]={val};}}"
        if op in ("ReduceMean", "ReduceSum", "ReduceMax", "GlobalAveragePool"):
            axes = (
                list(range(2, len(xs)))
                if op == "GlobalAveragePool"
                else self.axes(n, at, list(range(len(xs))))
            )
            if not axes and at.get("noop_with_empty_axes", 0):
                return loop(r(0))
            axes = sorted({d % len(xs) for d in (axes or list(range(len(xs))))})
            keep = at.get("keepdims", 1)
            reduced = tuple(xs[d] for d in axes)
            terms = []
            oi = 0
            for d in range(len(xs)):
                c = (
                    coord("k", reduced, axes.index(d))
                    if d in axes
                    else coord("i", os, d if keep else oi)
                )
                if d not in axes:
                    oi += 1
                terms.append(f"{c}*{strides(xs)[d]}")
            a = r(0, "+".join(terms))
            init = "-3.4028234e38f" if op == "ReduceMax" else "0"
            update = f"if({a}>s)s={a};" if op == "ReduceMax" else f"s+={a};"
            value = (
                f"s/{size(reduced)}"
                if op in ("ReduceMean", "GlobalAveragePool")
                else "s"
            )
            return f"for(int i=0;i<{N};i++){{float s={init};for(int k=0;k<{size(reduced)};k++){{{update}}}{y}[i]={value};}}"
        if op == "LayerNormalization":
            axis = at.get("axis", -1) % len(xs)
            width = size(xs[axis:])
            outer = size(xs[:axis])
            eps = f32(at.get("epsilon", 1e-5))
            ix = f"o*{width}+k"
            scale = r(1, broadcast_index(ix, xs, self.s(ins[1])))
            bias = (
                r(2, broadcast_index(ix, xs, self.s(ins[2]))) if len(ins) > 2 else "0"
            )
            return f"for(int o=0;o<{outer};o++){{float m=0,v=0;for(int k=0;k<{width};k++)m+={r(0,ix)};m/={width};for(int k=0;k<{width};k++){{float d={r(0,ix)}-m;v+=d*d;}}float inv=1/rv_sqrt(v/{width}+{eps});for(int k=0;k<{width};k++){y}[{ix}]=({r(0,ix)}-m)*inv*{scale}+{bias};}}"
        if op == "BatchNormalization":
            if at.get("training_mode", 0):
                raise ConversionError("BatchNormalization must be in evaluation mode")
            ch = f"i/{size(xs[2:])}%{xs[1]}"
            eps = f32(at.get("epsilon", 1e-5))
            return loop(
                f"({r(0)}-{r(3,ch)})*{r(1,ch)}/rv_sqrt({r(4,ch)}+{eps})+{r(2,ch)}"
            )
        if op == "Gather":
            axis = at.get("axis", 0) % len(xs)
            ids = self.s(ins[1])
            before = size(xs[:axis])
            after = size(xs[axis + 1 :])
            count = size(ids)
            idx = r(1, f"i/{after}%{count}")
            return f'for(int i=0;i<{N};i++){{int k=(int){idx};if(k<0)k+={xs[axis]};if(k<0||k>={xs[axis]})return -2;{y}[i]={r(0,f"(i/{after*count}*{xs[axis]}+k)*{after}+i%{after}")};}}'
        if op == "Concat":
            axis = at["axis"] % len(os)
            inner = size(os[axis + 1 :])
            code = []
            offset = 0
            for key in ins:
                ns = self.s(key)
                count = size(ns)
                width = ns[axis]
                ix = f"(i/{width*inner}*{os[axis]}+{offset}+i/{inner}%{width})*{inner}+i%{inner}"
                code.append(
                    f'for(int i=0;i<{count};i++){y}[{ix}]={self.read(key,"i")};'
                )
                offset += width
            return "\n".join(code)
        if op == "Split":
            axis = at.get("axis", 0) % len(xs)
            inner = size(xs[axis + 1 :])
            offset = 0
            code = []
            for key in n.output:
                ns = self.s(key)
                width = ns[axis]
                ix = f"(i/{width*inner}*{xs[axis]}+{offset}+i/{inner}%{width})*{inner}+i%{inner}"
                code.append(
                    f"for(int i=0;i<{size(ns)};i++){self.ids[key]}[i]={r(0,ix)};"
                )
                offset += width
            return "\n".join(code)
        if op == "Slice":
            if any(k not in self.constants for k in ins[1:] if k):
                raise ConversionError("Slice parameters must be constant")
            starts = self.constants[ins[1]].ravel()
            ends = self.constants[ins[2]].ravel()
            axes = (
                self.constants[ins[3]].ravel()
                if len(ins) > 3 and ins[3]
                else np.arange(len(starts))
            )
            steps = (
                self.constants[ins[4]].ravel()
                if len(ins) > 4 and ins[4]
                else np.ones(len(starts), dtype=int)
            )
            mappings = {
                int(a)
                % len(xs): slice(int(s), int(e), int(t)).indices(xs[int(a) % len(xs)])
                for a, s, e, t in zip(axes, starts, ends, steps)
            }
            terms = []
            for d in range(len(xs)):
                start, _, step = mappings.get(d, (0, xs[d], 1))
                terms.append(f'({start}+{coord("i",os,d)}*{step})*{strides(xs)[d]}')
            return loop(r(0, "+".join(terms)))
        if op == "Pad":
            if at.get("mode", b"constant") != b"constant":
                raise ConversionError("Only constant Pad supported")
            pads = self.constants[ins[1]].ravel().astype(int)
            value = self.scalar(ins[2]) if len(ins) > 2 and ins[2] else 0
            if len(pads) != 2 * len(xs):
                raise ConversionError("Pad axes input is unsupported")
            coords = [f'({coord("i",os,d)}-({pads[d]}))' for d in range(len(xs))]
            valid = "&&".join(f"({c}>=0&&{c}<{xs[d]})" for d, c in enumerate(coords))
            ix = "+".join(f"{c}*{strides(xs)[d]}" for d, c in enumerate(coords))
            return loop(f"({valid}?{r(0,ix)}:{f32(value)})")
        if op in ("Conv", "MaxPool", "AveragePool"):
            if len(xs) not in (3, 4):
                raise ConversionError(f"{op} supports one and two spatial dimensions")
            spatial = len(xs) - 2
            kernel = list(
                at.get("kernel_shape", self.s(ins[1])[2:] if op == "Conv" else [])
            )
            st = list(at.get("strides", [1] * spatial))
            dil = list(at.get("dilations", [1] * spatial))
            pads = list(at.get("pads", [0] * (2 * spatial)))
            auto = at.get("auto_pad", b"NOTSET")
            if auto in (b"SAME_UPPER", b"SAME_LOWER"):
                for d in range(spatial):
                    total = max(
                        0,
                        (os[d + 2] - 1) * st[d]
                        + (kernel[d] - 1) * dil[d]
                        + 1
                        - xs[d + 2],
                    )
                    pads[d] = total // 2 if auto == b"SAME_UPPER" else (total + 1) // 2
            elif auto == b"VALID":
                pads = [0] * (2 * spatial)
            elif auto != b"NOTSET":
                raise ConversionError("Unsupported auto_pad")
            if at.get("ceil_mode", 0):
                raise ConversionError("Pooling ceil_mode is not supported")
            K = size(kernel)
            oxsize = size(os[2:])
            ixsize = size(xs[2:])
            oc = f"i/{oxsize}%{os[1]}"
            batch = f"i/{size(os[1:])}"
            cs = [
                f'({coord("i",os,d+2)}*{st[d]}-{pads[d]}+{coord("k",kernel,d)}*{dil[d]})'
                for d in range(spatial)
            ]
            valid = "&&".join(f"({c}>=0&&{c}<{xs[d+2]})" for d, c in enumerate(cs))
            pos = "+".join(f"{c}*{strides(xs[2:])[d]}" for d, c in enumerate(cs))
            if op == "Conv":
                group = at.get("group", 1)
                icg = xs[1] // group
                ocg = os[1] // group
                if xs[1] % group or os[1] % group:
                    raise ConversionError("Invalid Conv group")
                xi = f"({batch}*{xs[1]}+({oc})/{ocg}*{icg}+c)*{ixsize}+{pos}"
                wi = f"(({oc})*{icg}+c)*{K}+k"
                bias = r(2, oc) if len(ins) > 2 else "0"
                return f"for(int i=0;i<{N};i++){{float s={bias};for(int c=0;c<{icg};c++)for(int k=0;k<{K};k++)if({valid})s+={r(0,xi)}*{r(1,wi)};{y}[i]=s;}}"
            xi = f"(i/{oxsize})*{ixsize}+{pos}"
            a = r(0, xi)
            init = "-3.4028234e38f" if op == "MaxPool" else "0"
            update = f"if({a}>s)s={a};" if op == "MaxPool" else f"s+={a};"
            value = (
                "s"
                if op == "MaxPool"
                else (f"s/{K}" if at.get("count_include_pad", 0) else "s/count")
            )
            return f"for(int i=0;i<{N};i++){{float s={init};int count=0;for(int k=0;k<{K};k++)if({valid}){{{update}count++;}}{y}[i]={value};}}"
        raise ConversionError("Unsupported operator " + op)

    def generate(self):
        code = [
            "/* Generated by RVForge 0.1.0. Fixed shapes; stateful workspace; serialize calls. */",
            '#include "model.h"',
            MATH,
        ]
        live = set()
        for n in self.nodes:
            live.update(n.input)
            live.update(n.output)
        live.update(v.name for v in self.model.graph.output)
        for key, a in self.constants.items():
            if key not in live:
                continue
            ident = self.ids[key]
            if key in self.encoded:
                q, s, w, _ = self.encoded[key]
                self.storage_bytes += q.nbytes + s.nbytes
                typ = "rv_i8" if self.precision == "int8" else "rv_u8"
                code.append(
                    f"static const {typ} {ident}_q[{q.size}]={{"
                    + ",".join(str(int(x)) for x in q.ravel())
                    + "};"
                )
                code.append(
                    f"static const float {ident}_s[{s.size}]={{" + arr(s) + "};"
                )
                val = (
                    f"{ident}_q[i]"
                    if self.precision == "int8"
                    else f"(({ident}_q[i/8]>>(i%8)&1)?1.0f:-1.0f)"
                )
                code.append(
                    f"static float {ident}_get(int i){{return {val}*{ident}_s[i/{w}];}}"
                )
            else:
                if a.dtype.kind not in "fbiu":
                    raise ConversionError(f"Unsupported constant type {a.dtype}")
                # Shape constants need not be representable as runtime floats when only used structurally.
                structural = all(
                    (
                        key not in n.input
                        or n.op_type
                        in (
                            "Reshape",
                            "Slice",
                            "Squeeze",
                            "Unsqueeze",
                            "Expand",
                            "Pad",
                            "Split",
                        )
                        and list(n.input).index(key) > 0
                        or n.op_type.startswith("Reduce")
                        and list(n.input).index(key) > 0
                    )
                    for n in self.nodes
                )
                if structural and key not in [v.name for v in self.model.graph.output]:
                    continue
                if a.dtype.kind in "iu" and np.any(
                    np.abs(a.astype(np.float64)) > 16777216
                ):
                    raise ConversionError(
                        "Runtime integer constant exceeds exact FP32 integer range"
                    )
                self.storage_bytes += a.size * 4
                code.append(f"static const float {ident}[{a.size}]={{" + arr(a) + "};")
        input_names = [
            v.name for v in self.model.graph.input if v.name not in self.constants
        ]
        buffers = set(input_names)
        for n in self.nodes:
            buffers.update(x for x in n.output if x)
        self.workspace_bytes = sum(size(self.s(k)) * 4 for k in buffers)
        for key in sorted(buffers):
            code.append(f"static float {self.ids[key]}[{size(self.s(key))}];")
        params = [
            f'const {"int" if self.types.get(k) in (6,7) else "float"} *input{j}'
            for j, k in enumerate(input_names)
        ]
        params += [f"float *output{j}" for j, _ in enumerate(self.model.graph.output)]
        signature = "int rv_model_run(" + ",".join(params) + ")"
        code.append(signature + "{")
        for j, key in enumerate(input_names):
            if self.types.get(key) not in (1, 6, 7):
                raise ConversionError(
                    "Inputs must be float32, int32 or int64 token indices"
                )
            if self.types.get(key) in (6, 7):
                code.append(
                    f"for(int i=0;i<{size(self.s(key))};i++)if(input{j}[i]<-16777216||input{j}[i]>16777216)return -3;"
                )
            else:
                code.append(
                    f"for(int i=0;i<{size(self.s(key))};i++)if(!rv_finite(input{j}[i]))return -4;"
                )
            code.append(
                f"for(int i=0;i<{size(self.s(key))};i++){self.ids[key]}[i]=(float)input{j}[i];"
            )
        for n in self.nodes:
            try:
                code.append("/* " + n.op_type + " */\n" + self.emit(n))
                for key in n.output:
                    if key:
                        code.append(
                            f"for(int i=0;i<{size(self.s(key))};i++)if(!rv_finite({self.ids[key]}[i]))return -5;"
                        )
            except (KeyError, ValueError, IndexError, TypeError) as e:
                raise ConversionError(f"{n.op_type} ({n.name}): {e}") from e
        for j, v in enumerate(self.model.graph.output):
            code.append(
                f'for(int i=0;i<{size(self.s(v.name))};i++)output{j}[i]={self.read(v.name,"i")};'
            )
        code.append("return 0;}")
        header = (
            '#ifndef RVFORGE_MODEL_H\n#define RVFORGE_MODEL_H\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
            + signature
            + ";\n#ifdef __cplusplus\n}\n#endif\n#endif\n"
        )
        return "\n".join(code), header, input_names


def _numpy_inputs(inputs):
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    arrays = [
        x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        for x in inputs
    ]
    for a in arrays:
        if a.dtype not in (np.dtype("float32"), np.dtype("int32"), np.dtype("int64")):
            raise ConversionError(f"Input dtype {a.dtype} unsupported")
        if not np.all(np.isfinite(a)):
            raise ConversionError("Input values must be finite")
        if a.dtype.kind == "i" and np.any(np.abs(a.astype(np.float64)) > 16777216):
            raise ConversionError("Integer inputs must be within +/- 2^24")
    return arrays


def convert(
    model, example_inputs, output_dir, precision="fp32", validation_inputs=None
):
    """Export an eval-mode PyTorch module to C plus self-test fixtures.

    validation_inputs is an iterable of additional tuples of inputs of identical shapes.
    Graph structure must not depend on input values. Static shapes are specialized.
    """
    import torch, onnxruntime as ort

    if precision not in ("fp32", "int8", "binary"):
        raise ConversionError("precision must be fp32, int8 or binary")
    dest = Path(output_dir)
    if dest.exists() and any(dest.iterdir()):
        raise ConversionError(
            "Output directory must be empty; use a fresh build directory"
        )
    examples = _numpy_inputs(example_inputs)
    tensors = tuple(torch.from_numpy(a) for a in examples)
    fixtures = [examples] + [_numpy_inputs(x) for x in (validation_inputs or [])]
    for case in fixtures:
        if len(case) != len(examples) or any(
            a.shape != b.shape or a.dtype != b.dtype for a, b in zip(case, examples)
        ):
            raise ConversionError(
                "Validation shapes and dtypes must match example inputs"
            )
    with tempfile.TemporaryDirectory(prefix="rvforge-") as td:
        temp = Path(td)
        graphfile = temp / "graph.onnx"
        export_model = copy.deepcopy(model).cpu().eval()
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("error", torch.jit.TracerWarning)
            try:
                torch.onnx.export(
                    export_model,
                    tensors,
                    str(graphfile),
                    input_names=[f"input{j}" for j in range(len(tensors))],
                    opset_version=17,
                    dynamo=False,
                    do_constant_folding=True,
                )
            except Exception as e:
                raise ConversionError(f"Python graph export failed: {e}") from e
        original = onnx.load(graphfile)
        onnx.checker.check_model(original)
        g = Graph(original, precision)
        c, h, input_names = g.generate()
        if len(input_names) != len(examples):
            raise ConversionError(
                "Model export removed an unused input; remove it from the factory"
            )
        # Independent execution of the transformed weights in ONNX Runtime.
        transformed = copy.deepcopy(original)
        replacements = {k: v[3] for k, v in g.encoded.items()}
        for j, init in enumerate(transformed.graph.initializer):
            if init.name in replacements:
                transformed.graph.initializer[j].CopyFrom(
                    numpy_helper.from_array(replacements[init.name], init.name)
                )
        # Folded Constant-node weights can also be compressed; replace their nodes explicitly.
        if replacements.keys() - {i.name for i in original.graph.initializer}:
            raise ConversionError(
                "Compressed folded weights unsupported; export parameters as initializers"
            )
        onnx.save(transformed, temp / "reference.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        base = ort.InferenceSession(
            original.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        ref = ort.InferenceSession(
            transformed.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        ref_outputs = []
        errors = []
        fixture_data = {}
        for ci, case in enumerate(fixtures):
            feeds = dict(zip(input_names, case))
            before = base.run(None, feeds)
            after = ref.run(None, feeds)
            ref_outputs.append(after)
            with torch.inference_mode():
                eager = export_model(*(torch.from_numpy(a) for a in case))

            def flatten(value):
                if isinstance(value, (tuple, list)):
                    return [a for v in value for a in flatten(v)]
                if isinstance(value, dict):
                    return [a for v in value.values() for a in flatten(v)]
                return [value.detach().cpu().numpy()]

            eager = flatten(eager)
            if len(eager) != len(before) or any(
                not np.allclose(a, b, rtol=3e-4, atol=3e-4)
                for a, b in zip(eager, before)
            ):
                raise ConversionError(
                    f"Python export parity failed on fixture {ci}; graph may depend on input data"
                )
            if any(not np.all(np.isfinite(a)) for a in after + before):
                raise ConversionError("Nonfinite model output on verification fixture")
            for j, (a, b) in enumerate(zip(before, after)):
                delta = a.astype(np.float64) - b.astype(np.float64)
                errors.append(
                    dict(
                        case=ci,
                        output=j,
                        max_abs=float(np.max(np.abs(delta))),
                        rmse=float(np.sqrt(np.mean(delta**2))),
                    )
                )
                fixture_data[f"case{ci}_reference{j}"] = b
                fixture_data[f"case{ci}_original{j}"] = a
            for j, a in enumerate(case):
                fixture_data[f"case{ci}_input{j}"] = a
        (temp / "model.c").write_text(c)
        (temp / "model.h").write_text(h)
        from .verify import make_testbench

        (temp / "selftest.c").write_text(make_testbench(fixtures, ref_outputs))
        np.savez(temp / "fixtures.npz", **fixture_data)
        metadata = dict(
            version="0.1.0",
            precision=precision,
            arithmetic="FP32; weight storage only is compressed",
            inputs=[
                dict(
                    name=k,
                    shape=list(g.s(k)),
                    dtype=str(a.dtype),
                    c_dtype="int32" if a.dtype.kind == "i" else "float32",
                )
                for k, a in zip(input_names, examples)
            ],
            outputs=[
                dict(name=v.name, shape=list(g.s(v.name)))
                for v in original.graph.output
            ],
            operators=sorted({n.op_type for n in g.nodes}),
            operator_count=len(g.nodes),
            compressed_tensors=sorted(g.encoded),
            parameter_storage_bytes=g.storage_bytes,
            workspace_bytes=g.workspace_bytes,
            eligible_fp32_weight_bytes=g.float_weight_bytes,
            verification_cases=len(fixtures),
            compression_output_error=errors,
            limitations=[
                "Static shapes and data-independent graph only",
                "INT8 and binary weights are decoded during FP32 arithmetic",
                "No task-accuracy guarantee from numerical verification",
                "Generated workspace is static and not reentrant",
                "RV32 ELF targets QEMU virt, not arbitrary board memory maps",
            ],
        )
        metadata["source_graph_sha256"] = hashlib.sha256(
            graphfile.read_bytes()
        ).hexdigest()
        (temp / "manifest.json").write_text(json.dumps(metadata, indent=2))
        dest.mkdir(parents=True, exist_ok=True)
        for file in temp.iterdir():
            shutil.copy2(file, dest / file.name)
    return metadata
