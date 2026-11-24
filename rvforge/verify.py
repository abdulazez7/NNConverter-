"""Compile and execute independent numerical parity fixtures."""

import json, shutil, subprocess
from pathlib import Path
import numpy as np
from .compiler import arr


def make_testbench(cases, outputs):
    code = [
        '#include "model.h"',
        r"""
#ifdef RV_BARE_METAL
static void put(char c){volatile unsigned char *u=(volatile unsigned char*)0x10000000;while(!(u[5]&32)){}u[0]=(unsigned char)c;}
static void text(const char *s){while(*s)put(*s++);}
static void finish(int fail){*(volatile unsigned int*)0x100000=fail?0x00013333:0x5555;for(;;){}}
#else
#include <stdio.h>
static void text(const char *s){fputs(s,stdout);}
#endif
static int close_enough(float a,float b){float d=a-b;if(d<0)d=-d;float m=b<0?-b:b;return a==a && d<=0.0003f+0.0003f*m;}
""",
    ]
    for ci, case in enumerate(cases):
        for j, a in enumerate(case):
            typ = "int" if a.dtype.kind == "i" else "float"
            values = (
                ",".join(str(int(x)) for x in a.ravel()) if typ == "int" else arr(a)
            )
            code.append(f"static const {typ} in_{ci}_{j}[{a.size}]={{" + values + "};")
        for j, a in enumerate(outputs[ci]):
            code.append(
                f"static const float expected_{ci}_{j}[{a.size}]={{" + arr(a) + "};"
            )
    for j, a in enumerate(outputs[0]):
        code.append(f"static float out_{j}[{a.size}];")
    code.append("int main(void){int fail=0;")
    for ci, case in enumerate(cases):
        args = [f"in_{ci}_{j}" for j in range(len(case))] + [
            f"out_{j}" for j in range(len(outputs[ci]))
        ]
        code.append("if(rv_model_run(" + ",".join(args) + ")!=0)fail++;")
        for j, a in enumerate(outputs[ci]):
            code.append(
                f"for(int i=0;i<{a.size};i++)if(!close_enough(out_{j}[i],expected_{ci}_{j}[i]))fail++;"
            )
    code.append(
        f'if(fail)text("FAIL numerical parity\\n");else text("PASS {len(cases)} fixtures numerical parity\\n");'
    )
    code.append("#ifdef RV_BARE_METAL\nfinish(fail);\n#endif\nreturn fail?1:0;}")
    return "\n".join(code)


def _run(cmd, cwd):
    proc = subprocess.run(
        [str(x) for x in cmd], cwd=cwd, capture_output=True, text=True, timeout=900
    )
    if proc.returncode:
        raise RuntimeError(
            "Command failed: "
            + " ".join(str(x) for x in cmd)
            + "\n"
            + proc.stdout[-8000:]
            + proc.stderr[-8000:]
        )
    return proc.stdout.strip()


def verify(folder, zig, riscv=False, node="node"):
    folder = Path(folder).resolve()
    zig = str(Path(zig).resolve())
    target = Path(__file__).parent / "target"
    result = {}
    host = folder / ("selftest.exe" if __import__("os").name == "nt" else "selftest")
    _run(
        [
            zig,
            "cc",
            "-w",
            "-O2",
            "-ffp-contract=off",
            "model.c",
            "selftest.c",
            "-o",
            host,
        ],
        folder,
    )
    result["host"] = _run([host], folder)
    if riscv:
        flags = [
            "-triple",
            "riscv32-unknown-unknown-elf",
            "-target-cpu",
            "generic-rv32",
            "-target-feature",
            "+m",
            "-target-feature",
            "+f",
            "-target-feature",
            "+zicsr",
            "-target-abi",
            "ilp32f",
        ]
        _run(
            [
                zig,
                "-cc1as",
                *flags,
                "-filetype",
                "obj",
                "-o",
                "start.o",
                target / "start.S",
            ],
            folder,
        )
        for src in ["model", "selftest"]:
            _run(
                [
                    zig,
                    "-cc1",
                    *flags,
                    "-std=c11",
                    "-O2",
                    "-ffreestanding",
                    "-fno-builtin",
                    "-ffp-contract=off",
                    "-emit-obj",
                    "-DRV_BARE_METAL",
                    "-o",
                    src + ".o",
                    src + ".c",
                ],
                folder,
            )
        _run(
            [
                zig,
                "ld.lld",
                "-m",
                "elf32lriscv",
                "--no-relax",
                "-T",
                target / "virt.ld",
                "start.o",
                "model.o",
                "selftest.o",
                "-o",
                "model-rv32.elf",
            ],
            folder,
        )
        result["rv32_functional_emulator"] = _run(
            [node, target / "emulate.js", folder / "model-rv32.elf"], folder
        )
    result["tolerance"] = (
        "abs <= 0.0003 + 0.0003 * abs(reference); all output elements checked"
    )
    result["reference"] = (
        "ONNX Runtime executing identical transformed weights; original-weight drift is in manifest.json"
    )
    (folder / "verification.json").write_text(json.dumps(result, indent=2))
    return result
