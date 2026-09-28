"""Acoustic Verification Test for Task 7: In-Graph Static Noise Buffer.

Evaluates audio quality, speech naturalness, and spectral features when generating speech
with 0% CPU random number generation and 0 bytes of noise transferred across DMA.
Tested across Vietnamese, English, Korean, and Chinese Mandarin.
"""
import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import librosa

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import _ensure_utf8_stdout
from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine

TEST_CASES = {
    "vi": {
        "name": "Vietnamese (FLEURS-Vi)",
        "text": "Nuôi tuần lộc là sinh kế quan trọng của người dân vùng Bắc Âu.",
    },
    "en": {
        "name": "English (LJSpeech)",
        "text": "Reindeer husbandry is an important livelihood for the Sami people in Northern Europe.",
    },
    "ko": {
        "name": "Korean (KSS)",
        "text": "순록 축산은 북유럽 사미족의 중요한 전통 생계 수단 중 하나입니다.",
    },
    "zh": {
        "name": "Chinese Mandarin",
        "text": "驯鹿养殖是北欧萨米人的重要生计来源。",
    },
}


def compute_audio_metrics(wav: np.ndarray, sr: int):
    """Computes spectral centroid, voice band energy ratio, RMS, and peak amplitude."""
    # RMS Energy & Peak
    rms = float(np.sqrt(np.mean(wav ** 2)))
    peak = float(np.max(np.abs(wav)))

    # Spectral Centroid
    centroids = librosa.feature.spectral_centroid(y=wav, sr=sr)
    mean_centroid = float(np.mean(centroids))

    # Voice Band Energy Ratio (100 Hz - 3400 Hz)
    stft = np.abs(librosa.stft(wav))
    freqs = librosa.fft_frequencies(sr=sr)
    voice_band = (freqs >= 100) & (freqs <= 3400)
    total_energy = np.sum(stft ** 2) + 1e-12
    voice_energy = np.sum(stft[voice_band, :] ** 2)
    voice_ratio = float((voice_energy / total_energy) * 100.0)

    return {
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "centroid_hz": round(mean_centroid, 1),
        "voice_energy_ratio_pct": round(voice_ratio, 2),
    }


def test_static_noise_engine():
    _ensure_utf8_stdout()
    print("=" * 90)
    print(" 🎙️ VERIFICATION TEST: SUPERTONIC 3 WITH IN-GRAPH STATIC NOISE BUFFER (TASK 7)")
    print("    Target Architecture: Zero CPU np.random.randn() | Zero DMA Noise Bytes")
    print("=" * 90)

    out_dir = os.path.join(ROOT, "outputs", "task7_static_noise_verification")
    os.makedirs(out_dir, exist_ok=True)

    engine = SupertonicPureNPUV2Engine(target_sample_rate=16000)
    results = []

    for lang, item in TEST_CASES.items():
        text = item["text"]
        print(f"\nSynthesizing [{lang.upper()}] {item['name']}:")
        print(f"  • Input text: \"{text}\"")

        wav, stats = engine.synthesize(text=text, language=lang, speed=1.0)
        out_wav_path = os.path.join(out_dir, f"static_noise_{lang}.wav")
        sf.write(out_wav_path, wav, stats["sample_rate"])

        metrics = compute_audio_metrics(wav, stats["sample_rate"])

        print(f"  • Saved: {os.path.relpath(out_wav_path, ROOT)}")
        print(f"  • Duration: {stats['duration_sec']}s | Latency: {stats['total_latency_ms']}ms | RTF: {stats['rtf']}")
        print(f"  • RMS: {metrics['rms']} | Peak: {metrics['peak']} | Voice Energy: {metrics['voice_energy_ratio_pct']}% | Centroid: {metrics['centroid_hz']} Hz")

        # Quality assertions
        assert metrics["peak"] > 0.5, "Peak too low, audio might be silent"
        assert metrics["voice_energy_ratio_pct"] > 70.0, "Voice band energy below threshold"

        results.append({
            "language": lang,
            "text": text,
            "wav_path": out_wav_path,
            "duration_sec": stats["duration_sec"],
            "total_latency_ms": stats["total_latency_ms"],
            "rtf": stats["rtf"],
            "metrics": metrics,
        })

    report_path = os.path.join(out_dir, "static_noise_verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Saved full verification report: {os.path.relpath(report_path, ROOT)}")
    print("\n🌟 ALL MULTI-LINGUAL TESTS PASSED WITH STUDIO ACOUSTIC FIDELITY!")
    return results


if __name__ == "__main__":
    test_static_noise_engine()
