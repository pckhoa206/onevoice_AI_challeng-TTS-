#!/usr/bin/env python3
"""Verification test for post-patch synthesis quality."""

import os
import sys
import soundfile as sf
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine

def run_test():
    engine = SupertonicPureNPUV2Engine()
    out_dir = os.path.join(ROOT, "outputs", "test_post_patch")
    os.makedirs(out_dir, exist_ok=True)

    samples = [
        ("The OneVoice AI engine is running smoothly on Qualcomm Snapdragon NPU.", "en"),
        ("Chào mừng bạn đến với hệ thống tổng hợp giọng nói NPU thuần khiết.", "vi"),
        ("중동의 따뜻한 기후에서는 집이 그다지 중요하지 않았습니다.", "ko")
    ]

    for text, lang in samples:
        print(f"\n[Testing] Language: {lang} | Text: '{text}'")
        wav, stats = engine.synthesize(text, language=lang)
        out_path = os.path.join(out_dir, f"test_patched_{lang}.wav")
        sf.write(out_path, wav, stats["sample_rate"])
        dur = stats["duration_sec"]
        lat = stats["total_latency_ms"]
        rtf = stats["rtf"]
        print(f"  • Generated: {dur}s audio in {lat} ms (RTF: {rtf}) -> {out_path}")

    print("\n✅ All 3 test languages synthesized successfully!")

if __name__ == "__main__":
    run_test()
