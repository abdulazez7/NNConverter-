// Minimal RV32IMF functional emulator for the supplied QEMU-virt firmware.
"use strict";
const fs = require("node:fs");
const { performance } = require("node:perf_hooks");
const elf = fs.readFileSync(process.argv[2]);
const base = 0x80000000, bytes = new Uint8Array(16 << 20);
const ram = new DataView(bytes.buffer);
if (elf.toString("hex", 0, 6) !== "7f454c460101" || elf.readUInt16LE(18) !== 243)
    throw Error("Expected a little-endian RV32 ELF.");
const ph = elf.readUInt32LE(28), phSize = elf.readUInt16LE(42);
for (let i = 0; i < elf.readUInt16LE(44); i++) {
    const p = ph + i * phSize;
    if (elf.readUInt32LE(p) !== 1) continue;
    const offset = elf.readUInt32LE(p + 4), address = elf.readUInt32LE(p + 12);
    const length = elf.readUInt32LE(p + 16), capacity = elf.readUInt32LE(p + 20);
    if (address < base || address - base + capacity > bytes.length || length > capacity ||
        offset + length > elf.length) throw Error("ELF segment falls outside RAM or file.");
    bytes.set(elf.subarray(offset, offset + length), address - base);
}
const x = new Int32Array(32);
const fbits = new Uint32Array(32), f = new Float32Array(fbits.buffer);
const csr = new Uint32Array(4096);
const signed = (n, bits) => n << (32 - bits) >> (32 - bits);
let pc = elf.readUInt32LE(24);
let result = null, uart = "";
function load(address, code) {
    if (address === 0x10000005 && (code === 0 || code === 4)) return 0x20;
    const p = address - base;
    switch (code) {
        case 0: return ram.getInt8(p);
        case 1: return ram.getInt16(p, true);
        case 2: return ram.getInt32(p, true);
        case 4: return ram.getUint8(p);
        case 5: return ram.getUint16(p, true);
        default: throw Error("Illegal load.");
    }
}
function store(address, code, value) {
    if (address === 0x10000000 && code === 0) { uart += String.fromCharCode(value & 255); return; }
    if (address === 0x100000 && code === 2) {
        if ((value & 65535) === 0x5555) result = 0;
        else if ((value & 65535) === 0x3333) result = (value >>> 16) || 1;
        else throw Error("Unsupported guest exit code.");
        return;
    }
    const p = address - base;
    switch (code) {
        case 0: ram.setInt8(p, value); break;
        case 1: ram.setInt16(p, value, true); break;
        case 2: ram.setInt32(p, value, true); break;
        default: throw Error("Illegal store.");
    }
}
function reg(rd, value) { if (rd) x[rd] = value; }
function floatInstruction(inst, rd, a, b, mode, op) {
    switch (op) {
        case 0x00: f[rd] = f[a] + f[b]; return;
        case 0x04: f[rd] = f[a] - f[b]; return;
        case 0x08: f[rd] = f[a] * f[b]; return;
        case 0x0c: f[rd] = f[a] / f[b]; return;
        case 0x2c: f[rd] = Math.sqrt(f[a]); return;
        case 0x10: {
            const sign = mode === 2 ? fbits[a] ^ fbits[b] : mode === 1 ? ~fbits[b] : fbits[b];
            fbits[rd] = (fbits[a] & 0x7fffffff) | (sign & 0x80000000);
            return;
        }
        case 0x14: f[rd] = mode === 0 ? Math.min(f[a], f[b]) : Math.max(f[a], f[b]); return;
        case 0x50:
            reg(rd, mode === 2 ? f[a] === f[b] : mode === 1 ? f[a] < f[b] : f[a] <= f[b]);
            return;
        case 0x60: {
            if (mode !== 1 || b > 1) break;
            const lo = b ? 0 : -2147483648, hi = b ? 4294967295 : 2147483647;
            const value = Number.isNaN(f[a]) ? hi : Math.trunc(f[a]);
            reg(rd, Math.min(hi, Math.max(lo, value)));
            return;
        }
        case 0x68: if (b <= 1) { f[rd] = b ? x[a] >>> 0 : x[a]; return; } break;
        case 0x70: if (b === 0 && mode === 0) { reg(rd, fbits[a]); return; } break;
        case 0x78: if (b === 0 && mode === 0) { fbits[rd] = x[a]; return; } break;
    }
    throw Error("Unsupported FP instruction 0x" + inst.toString(16));
}
function integerOp(a, b, mode, fn) {
    const ua = a >>> 0, ub = b >>> 0;
    if (fn === 1) {
        switch (mode) {
            case 0: return Math.imul(a, b);
            case 1: case 2: case 3:
                return Number((BigInt(mode === 3 ? ua : a) * BigInt(mode === 1 ? b : ub)) >> 32n);
            case 4: return b === 0 ? -1 : a / b;
            case 5: return b === 0 ? -1 : ua / ub;
            case 6: return b === 0 ? a : a % b;
            case 7: return b === 0 ? a : ua % ub;
        }
    }
    switch (mode) {
        case 0: return fn === 0x20 ? a - b : a + b;
        case 1: return a << b;
        case 2: return a < b;
        case 3: return ua < ub;
        case 4: return a ^ b;
        case 5: return fn === 0x20 ? a >> b : a >>> b;
        case 6: return a | b;
        case 7: return a & b;
    }
}
const start = performance.now();
let instructions = 0;
while (result === null && instructions < 100000000) {
    const ip = pc;
    const inst = ram.getUint32(ip - base, true);
    const opcode = inst & 127, rd = inst >> 7 & 31, mode = inst >> 12 & 7;
    const rs1 = inst >> 15 & 31, rs2 = inst >> 20 & 31, fn = inst >>> 25;
    const a = x[rs1], b = x[rs2], imm = inst >> 20;
    const st = signed((inst >>> 25) << 5 | rd, 12);
    pc = (ip + 4) >>> 0;
    instructions++;
    switch (opcode) {
        case 0x03: reg(rd, load((a + imm) >>> 0, mode)); break;
        case 0x07: if(mode !== 2) throw Error("Expected FLW."); fbits[rd] = load((a + imm) >>> 0, 2); break;
        case 0x0f: break;
        case 0x13: reg(rd, integerOp(a, mode === 1 || mode === 5 ? rs2 : imm, mode, mode === 5 ? fn : 0)); break;
        case 0x17: reg(rd, ip + (inst & 0xfffff000)); break;
        case 0x23: store((a + st) >>> 0, mode, b); break;
        case 0x27: if(mode !== 2) throw Error("Expected FSW."); store((a + st) >>> 0, 2, fbits[rs2]); break;
        case 0x33: reg(rd, integerOp(a, b, mode, fn)); break;
        case 0x37: reg(rd, inst & 0xfffff000); break;
        case 0x53: floatInstruction(inst, rd, rs1, rs2, mode, fn); break;
        case 0x63: {
            const offset = signed((inst >>> 31) << 12 | ((inst >> 7) & 1) << 11 |
                ((inst >> 25) & 63) << 5 | (rd & 30), 13);
            const match = [a === b, a !== b, false, false, a < b, a >= b,
                (a >>> 0) < (b >>> 0), (a >>> 0) >= (b >>> 0)];
            if (match[mode]) pc = (ip + offset) >>> 0;
            break;
        }
        case 0x67: reg(rd, pc); pc = ((a + imm) & ~1) >>> 0; break;
        case 0x6f: {
            const offset = signed((inst >>> 31) << 20 | (inst & 0xff000) |
                ((inst >>> 20) & 1) << 11 | ((inst >>> 20) & 0x7fe), 21);
            reg(rd, pc); pc = (ip + offset) >>> 0;
            break;
        }
        case 0x73: {
            if (!mode || mode === 4) throw Error("Unsupported system instruction.");
            const index = inst >>> 20, old = csr[index], value = mode & 4 ? rs1 : a;
            if ((mode & 3) === 1) csr[index] = value;
            else if ((mode & 3) === 2) csr[index] = old | value;
            else csr[index] = old & ~value;
            reg(rd, old);
            break;
        }
        default: throw Error("Unsupported instruction 0x" + inst.toString(16) + " at 0x" + ip.toString(16));
    }
}
process.stdout.write(uart);
if (result === null) throw Error("Guest exceeded the instruction limit.");
console.log("RV32IMF functional emulation: " + instructions + " instructions; host elapsed " +
    (performance.now() - start).toFixed(1) + " ms");
process.exitCode = result;
