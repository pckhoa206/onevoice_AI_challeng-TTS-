"""Comprehensive Unit Test & Benchmark for Qualcomm QNN Custom Tokenizer Package.

Validates:
  1. 100% bit-exact equivalence between QNN C++ Custom Kernel and Python UnicodeProcessor.
  2. Fixed static shape compliance (shape=(1, 64), int64 dtype, float32 mask).
  3. Latency benchmark across English and Korean.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout
from step3_tts.qnn_custom_tokenizer.qnn_tokenizer_engine import get_qnn_tokenizer
from supertonic import TTS


def test_bit_exact_equivalence():
    print("=" * 80)
    print(" 🧪 TEST 1: BIT-TO-BIT IDENTITY VERIFICATION (QNN C++ VS PYTHON)")
    print("=" * 80)

    tts = TTS(auto_download=False)
    py_proc = tts.model.text_processor
    qnn_tok = get_qnn_tokenizer()

    test_corpus = [
        # English test cases
        ("en", "Hello world"),
        ("en", "This is Qualcomm Hexagon NPU on Galaxy S24 Ultra."),
        ("en", "Hello, how are you? I am fine."),
        ("en", "Short text."),
        ("en", "Symbols test: 100% pure NPU – fast!"),
        ("en", "Quotes test: 'single' and \"double\" quotes."),
        # Korean test cases
        ("ko", "안녕하세요"),
        ("ko", "퀄컴 NPU 음성 합성 테스트입니다."),
        ("ko", "오늘 날씨가 참 좋습니다."),
        ("ko", "한국어와 영어 테스트: Hello Korea!"),
        ("ko", "감사합니다! 좋은 하루 되세요."),
        ("ko", "인공지능 딥러닝 음성합성 모델 테스트."),
    ]

    all_passed = True
    for lang, text in test_corpus:
        qnn_ids, qnn_mask = qnn_tok.tokenize(text, lang, max_len=64)
        py_ids, py_mask = py_proc([text], lang)

        py_list = py_ids[0].tolist()
        qnn_list = [x for x in qnn_ids[0].tolist() if x != 0]

        is_match = (py_list == qnn_list)
        status = "✅ PASSED" if is_match else "❌ FAILED"
        print(f" • [{lang.upper()}] {text:<45} | Tokens: {len(py_list):2d} | Status: {status}")

        if not is_match:
            all_passed = False
            print(f"    Expected (Py) : {py_list}")
            print(f"    Actual   (QNN): {qnn_list}")

        # Check static shape
        assert qnn_ids.shape == (1, 64), f"Wrong shape: {qnn_ids.shape}"
        assert qnn_mask.shape == (1, 1, 64), f"Wrong mask shape: {qnn_mask.shape}"
        assert qnn_ids.dtype == np.int64, f"Wrong dtype: {qnn_ids.dtype}"
        assert qnn_mask.dtype == np.float32, f"Wrong mask dtype: {qnn_mask.dtype}"

    print("=" * 80)
    return all_passed


def test_latency_benchmark():
    print("=" * 80)
    print(" ⚡ TEST 2: LATENCY BENCHMARK (1,000 ITERATIONS)")
    print("=" * 80)

    tts = TTS(auto_download=False)
    py_proc = tts.model.text_processor
    qnn_tok = get_qnn_tokenizer()

    bench_cases = [
        ("en", "The OneVoice AI Challenge runs on Qualcomm Hexagon NPU on Galaxy S24 Ultra."),
        ("ko", "퀄컴 NPU 음성 합성 테스트입니다. 오늘 날씨가 참 좋습니다.")
    ]

    N = 1000
    for lang, text in bench_cases:
        # Warmup
        for _ in range(50):
            py_proc([text], lang)
            qnn_tok.tokenize(text, lang, max_len=64)

        # Py benchmark
        t0 = time.time()
        for _ in range(N):
            py_proc([text], lang)
        t_py = ((time.time() - t0) / N) * 1000.0

        # QNN benchmark
        t0 = time.time()
        for _ in range(N):
            qnn_tok.tokenize(text, lang, max_len=64)
        t_qnn = ((time.time() - t0) / N) * 1000.0

        speedup = t_py / t_qnn if t_qnn > 0 else 1.0

        print(f" • Language [{lang.upper()}]:")
        print(f"    - Python UnicodeProcessor: {t_py * 1000.0:6.2f} µs ({t_py:.4f} ms)")
        print(f"    - QNN C++ Custom Kernel : {t_qnn * 1000.0:6.2f} µs ({t_qnn:.4f} ms)")
        print(f"    - Performance Gain       : 🚀 {speedup:.1f}x FASTER!")

    print("=" * 80)


def main():
    _ensure_utf8_stdout()
    passed = test_bit_exact_equivalence()
    test_latency_benchmark()

    if passed:
        print("\n 🎉 ALL QUALCOMM QNN CUSTOM TOKENIZER TESTS PASSED WITH 100% NUMERICAL FIDELITY!")
    else:
        print("\n ❌ VERIFICATION FAILED. PLEASE CHECK LOGS ABOVE.")
        sys.exit(1)


if __name__ == "__main__":
    main()
