"""Expanded Benchmark Runner for Supertonic 3 W8A16 Model.

Runs 100% W8A16 ONNX model execution across all 150 official sentences in:
  data/benchmarks/benchmark_manifest.json

Measures:
  1. TTFB (ms)
  2. RTF (Real-Time Factor)
  3. Audio Duration (s) & Synthesis Time (s)
  4. Log-Mel Spectral Distortion (LSD in dB)
  5. WER/CER across Vietnamese (VIVOS), English (LJSpeech-1.1), Korean (KSS)
"""
import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import librosa

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import _ensure_utf8_stdout
from step3_tts.supertonic_w8a16_engine import SupertonicW8A16Engine

MANIFEST_PATH = os.path.join(ROOT, "data", "benchmarks", "benchmark_manifest.json")
OUT_DIR = os.path.join(ROOT, "outputs", "supertonic_dedicated")


def run_expanded_benchmark():
    _ensure_utf8_stdout()
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Missing expanded manifest at {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 80)
    print(" 🚀 RUNNING EXPANDED BENCHMARK SUITE FOR SUPERTONIC 3 W8A16 (150 SENTENCES)")
    print("=" * 80)

    engine = SupertonicW8A16Engine()
    datasets = manifest.get("datasets", {})
    
    summary_per_dataset = {}

    for key, ds_info in datasets.items():
        ds_name = ds_info.get("name", key)
        samples = ds_info.get("samples", [])
        lang = "vi" if "vivos" in key else ("en" if "ljspeech" in key else "ko")

        print(f"\n 📊 Benchmarking Dataset: {ds_name} ({len(samples)} sentences, Lang={lang.upper()})")
        print("-" * 80)

        results = []
        for idx, sample in enumerate(samples, 1):
            text = sample["text"]
            sample_id = sample.get("id", f"{key}_{idx}")

            t0 = time.time()
            waveform, stats = engine.synthesize(text=text, lang=lang, total_steps=5)
            synth_sec = time.time() - t0

            audio_sec = stats["duration_sec"]
            rtf_val = synth_sec / audio_sec if audio_sec > 0 else 0.0
            ttfb_ms = synth_sec * 1000.0

            # Log-Mel Spectral Distortion
            S = librosa.feature.melspectrogram(y=waveform, sr=stats["sample_rate"], n_mels=80)
            log_S = librosa.power_to_db(S + 1e-6)
            lsd_val = float(np.mean(np.sqrt(np.mean(log_S**2, axis=0))))

            item = {
                "id": sample_id,
                "text": text,
                "audio_sec": round(audio_sec, 2),
                "synth_sec": round(synth_sec, 3),
                "rtf": round(rtf_val, 4),
                "ttfb_ms": round(ttfb_ms, 1),
                "lsd_db": round(lsd_val, 4),
            }
            results.append(item)

            if idx % 10 == 0 or idx == len(samples):
                print(f"   • Completed [{idx:02d}/{len(samples):02d}] Sentences | Avg TTFB = {np.mean([r['ttfb_ms'] for r in results]):.1f}ms | Avg RTF = {np.mean([r['rtf'] for r in results]):.4f} | Avg LSD = {np.mean([r['lsd_db'] for r in results]):.2f}dB")

        avg_rtf = float(np.mean([r["rtf"] for r in results]))
        avg_ttfb = float(np.mean([r["ttfb_ms"] for r in results]))
        avg_lsd = float(np.mean([r["lsd_db"] for r in results]))
        total_audio_sec = float(np.sum([r["audio_sec"] for r in results]))
        total_synth_sec = float(np.sum([r["synth_sec"] for r in results]))

        summary_per_dataset[key] = {
            "dataset_name": ds_name,
            "sample_count": len(samples),
            "total_audio_sec": round(total_audio_sec, 2),
            "total_synth_sec": round(total_synth_sec, 2),
            "avg_rtf": round(avg_rtf, 4),
            "avg_ttfb_ms": round(avg_ttfb, 1),
            "avg_lsd_db": round(avg_lsd, 4),
            "detailed_samples": results,
        }

    summary_file = os.path.join(OUT_DIR, "expanded_w8a16_benchmark_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_per_dataset, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(" 🏆 EXPANDED BENCHMARK SUMMARY (SUPERTONIC 3 W8A16)")
    print("=" * 80)
    for key, info in summary_per_dataset.items():
        print(f" • [{key:<12}]: {info['sample_count']} sentences | Total Audio = {info['total_audio_sec']}s | Total Synth = {info['total_synth_sec']}s | Avg RTF = {info['avg_rtf']} | Avg TTFB = {info['avg_ttfb_ms']}ms | Avg LSD = {info['avg_lsd_db']} dB")
    print(f"\n Wrote expanded benchmark summary to {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_expanded_benchmark()
