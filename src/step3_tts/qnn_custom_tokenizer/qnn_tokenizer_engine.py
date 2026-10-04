"""Qualcomm QNN Custom Tokenizer Engine Wrapper.

Loads the compiled QNN Custom Op Package (C++ Hexagon/Host kernel)
and provides high-speed tokenization with direct NumPy output buffers.
"""

import os
import sys
import platform
import ctypes
import numpy as np
from typing import Tuple, Optional


class QnnTokenizerEngine:
    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            is_mac = platform.system() == "Darwin"
            lib_name = "libQnnSupertonicTokenizer.dylib" if is_mac else "libQnnSupertonicTokenizer.so"
            lib_path = os.path.join(base_dir, lib_name)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"QNN Custom Tokenizer binary not found at '{lib_path}'. "
                f"Please run `python src/step3_tts/qnn_custom_tokenizer/build_qnn_tokenizer.py` first."
            )

        self.lib = ctypes.CDLL(lib_path)

        # Function signature:
        # int supertonic_qnn_tokenize(const char* utf8_text, const char* lang, int64_t* out_ids, float* out_mask, int max_len);
        self.tokenize_fn = self.lib.supertonic_qnn_tokenize
        self.tokenize_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.tokenize_fn.restype = ctypes.c_int

    def tokenize(
        self, text: str, lang: str = "en", max_len: int = 64
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenizes text using the compiled QNN C++ Custom Kernel.

        Args:
            text: Input string (English or Korean).
            lang: "en" or "ko".
            max_len: Fixed static sequence length for Qualcomm NPU (default 64).

        Returns:
            Tuple of:
              - text_ids: Shape (1, max_len) of dtype int64.
              - text_mask: Shape (1, 1, max_len) of dtype float32.
        """
        out_ids = np.zeros((1, max_len), dtype=np.int64)
        out_mask = np.zeros((1, 1, max_len), dtype=np.float32)

        utf8_bytes = text.encode("utf-8")
        lang_bytes = lang.encode("utf-8")

        ptr_ids = out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        ptr_mask = out_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        valid_len = self.tokenize_fn(utf8_bytes, lang_bytes, ptr_ids, ptr_mask, max_len)

        return out_ids, out_mask


# Global singleton instance for high-performance reuse
_DEFAULT_ENGINE: Optional[QnnTokenizerEngine] = None


def get_qnn_tokenizer() -> QnnTokenizerEngine:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = QnnTokenizerEngine()
    return _DEFAULT_ENGINE
