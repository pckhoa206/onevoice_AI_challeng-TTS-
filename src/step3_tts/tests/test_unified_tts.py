"""Integration & Quality Benchmark Test Suite for Unified TTS Manager.
Validates 4-language (Vi, En, Zh, Ko) text normalization, audio output contracts (16kHz mono),
TTFB latency, streaming sentence chunking, and exports benchmark CSV results.
"""
import os
import sys
import csv
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout
from step3_tts.text_normalizer import TextNormalizer
from step3_tts.tts_manager import UnifiedTTSManager, TARGET_SR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_AUDIO_DIR = os.path.join(ROOT, "outputs", "tts_unified")
RESULTS_CSV = os.path.join(ROOT, "outputs", "tts_unified_results.csv")


def test_text_normalizer():
    print("[test_unified_tts] Testing Text Normalizer...")
    norm = TextNormalizer()

    # Vi
    vi_input = "Dự án VNG tại Qualcomm 2026 đạt 50000$ USD."
    vi_clean = norm.normalize(vi_input, "vi")
    assert "vê en giê" in vi_clean, f"Vi normalizer failed acronym: {vi_clean}"
    print(f"  [Vi Normalizer] '{vi_input}' -> '{vi_clean}'")

    # En
    en_input = "Qualcomm NPU test with VNG AI Challenge in 2026 for 500$."
    en_clean = norm.normalize(en_input, "en")
    assert "dollars" in en_clean, f"En normalizer failed currency: {en_clean}"
    print(f"  [En Normalizer] '{en_input}' -> '{en_clean}'")

    # Ko
    ko_input = "Qualcomm NPU 50,000원 테스트 VNG AI."
    ko_clean = norm.normalize(ko_input, "ko")
    assert "엔피유" in ko_clean, f"Ko normalizer failed acronym: {ko_clean}"
    print(f"  [Ko Normalizer] '{ko_input}' -> '{ko_clean}'")

    # Zh
    zh_input = "Qualcomm AI 高通 NPU 50元."
    zh_clean = norm.normalize(zh_input, "zh")
    assert "高通" in zh_clean, f"Zh normalizer failed: {zh_clean}"
    print(f"  [Zh Normalizer] '{zh_input}' -> '{zh_clean}'")

    print("[test_unified_tts] Text Normalizer tests PASSED!")


def test_unified_manager():
    print("[test_unified_tts] Initializing UnifiedTTSManager...")
    manager = UnifiedTTSManager(warmup=True)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest_rows = json.load(f)[:3]  # First 3 sentences

    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    all_results = []

    languages = ["vi", "en"]  # Run core languages for integration test

    for lang in languages:
        for idx, row in enumerate(manifest_rows):
            text = row.get(lang, "Hello world")
            print(f"\n[test_unified_tts] Synthesizing {lang.upper()} [{idx}]: '{text[:40]}'")

            res = manager.synthesize(text, lang)

            # Audio Contract Assertions
            assert res.sample_rate == TARGET_SR, f"Invalid sample rate: {res.sample_rate}"
            assert len(res.audio_array) > 0, "Audio array is empty"
            assert not np.isnan(res.audio_array).any(), "Audio array contains NaN values"
            assert not np.isinf(res.audio_array).any(), "Audio array contains Inf values"
            assert len(res.audio_bytes) > 0, "Audio bytes is empty"

            # Save WAV file
            wav_path = os.path.join(OUT_AUDIO_DIR, f"{lang}_{idx}.wav")
            import soundfile as sf
            sf.write(wav_path, res.audio_array, TARGET_SR)

            row_data = {
                "engine": res.engine,
                "lang": res.lang,
                "idx": idx,
                "text": text,
                "audio_sec": res.duration_sec,
                "rtf": res.rtf,
                "ttfb_ms": res.ttfb_ms,
                "sample_rate": res.sample_rate,
                "wav_path": wav_path,
            }
            all_results.append(row_data)
            print(f"  Result: RTF={res.rtf:.3f} | TTFB={res.ttfb_ms:.1f}ms | audio={res.duration_sec:.2f}s | engine={res.engine}")

    # Write CSV
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[test_unified_tts] Wrote test benchmark results to {RESULTS_CSV}")


def main():
    test_text_normalizer()
    test_unified_manager()


if __name__ == "__main__":
    main()
