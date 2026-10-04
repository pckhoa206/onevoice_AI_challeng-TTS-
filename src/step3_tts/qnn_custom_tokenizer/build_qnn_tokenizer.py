#!/usr/bin/env python3
"""Build script for Qualcomm QNN Custom Supertonic Tokenizer Package."""

import os
import sys
import platform
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(BASE_DIR, "include")
SRC_DIR = os.path.join(BASE_DIR, "src")

IS_MAC = platform.system() == "Darwin"
LIB_NAME = "libQnnSupertonicTokenizer.dylib" if IS_MAC else "libQnnSupertonicTokenizer.so"
OUTPUT_PATH = os.path.join(BASE_DIR, LIB_NAME)


def find_compiler():
    # Prefer clang++ if available
    for c in ["clang++", "g++", "c++"]:
        path = shutil.which(c)
        if path:
            return path
    raise RuntimeError("No suitable C++ compiler (clang++, g++) found in PATH.")


def build():
    compiler = find_compiler()
    print(f"================================================================================")
    print(f" 🔨 COMPILING QUALCOMM QNN CUSTOM TOKENIZER PACKAGE")
    print(f" • Compiler : {compiler}")
    print(f" • Platform : {platform.system()} ({platform.machine()})")
    print(f" • Target   : {OUTPUT_PATH}")
    print(f"================================================================================")

    src_files = [
        os.path.join(SRC_DIR, "indexer_table.cpp"),
        os.path.join(SRC_DIR, "supertonic_tokenizer_kernel.cpp"),
        os.path.join(SRC_DIR, "QnnSupertonicTokenizerPackage.cpp"),
    ]

    for f in src_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing source file: {f}")

    cmd = [
        compiler,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-shared",
        f"-I{INCLUDE_DIR}",
        f"-I{SRC_DIR}",
    ]

    if IS_MAC:
        cmd.extend(["-dynamiclib", "-Wl,-undefined,dynamic_lookup"])

    cmd.extend(src_files)
    cmd.extend(["-o", OUTPUT_PATH])

    print("Running command:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if res.returncode != 0:
        print("❌ Compilation Failed!")
        print(res.stderr)
        sys.exit(1)

    print(f"✅ Compilation Succeeded! Binary generated: {OUTPUT_PATH}")
    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024.0
    print(f" • Binary Size: {file_size_kb:.1f} KB")
    print("================================================================================")


if __name__ == "__main__":
    build()
