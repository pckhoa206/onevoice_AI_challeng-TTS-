"""Build Static Character-to-Jamo Decomposition Table for 100% Pure NPU Tokenization.

Pre-computes exact token mappings for:
  - English / ASCII (Codepoints 0 to 127) -> [token_id, 0, 0]
  - Korean Hangul Syllables (Codepoints 0xAC00 to 0xD7A3, 11,172 characters) -> [token_cho, token_jung, token_jong]

Table Shape: [11300, 3] int64 (~271 KB) or int32 (~135.6 KB).
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine

OUTPUT_DIR = os.path.join(ROOT, "outputs", "embedding_tables")


def build_decomposition_table():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    engine = SupertonicPureNPUV2Engine()
    indexer = engine._helper_tts.model.text_processor.indexer

    # Table size: 128 (ASCII) + 11172 (Hangul) = 11300 rows
    table = np.zeros((11300, 3), dtype=np.int64)

    # 1. Populate ASCII (0 to 127)
    for cp in range(128):
        tok_id = indexer[cp] if cp < len(indexer) and indexer[cp] != -1 else 0
        table[cp, 0] = tok_id
        table[cp, 1] = 0
        table[cp, 2] = 0

    # 2. Populate Hangul syllables (0xAC00 to 0xD7A3)
    # Row index: 128 + (cp - 0xAC00)
    for cp in range(0xAC00, 0xD7A4):
        s_idx = cp - 0xAC00
        row_idx = 128 + s_idx

        l_idx = s_idx // 588
        v_idx = (s_idx % 588) // 28
        t_idx = s_idx % 28

        tok_l = indexer[0x1100 + l_idx] if (0x1100 + l_idx) < len(indexer) else 0
        tok_v = indexer[0x1161 + v_idx] if (0x1161 + v_idx) < len(indexer) else 0
        tok_t = indexer[0x11A7 + t_idx] if t_idx > 0 and (0x11A7 + t_idx) < len(indexer) else 0

        table[row_idx, 0] = max(0, tok_l)
        table[row_idx, 1] = max(0, tok_v)
        table[row_idx, 2] = max(0, tok_t)

    table_path = os.path.join(OUTPUT_DIR, "static_char_decomp_table.npy")
    np.save(table_path, table)
    print("=" * 80)
    print(" ✅ STATIC CHAR DECOMPOSITION TABLE BUILT SUCCESSFULLY!")
    print(f" • Shape : {table.shape} ({table.dtype})")
    print(f" • Size  : {table.nbytes / 1024:.2f} KB")
    print(f" • Saved : {os.path.relpath(table_path, ROOT)}")
    print("=" * 80)
    return table


if __name__ == "__main__":
    build_decomposition_table()
