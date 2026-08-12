"""Batch Synthesis Script for Step 3 TTS.
Synthesizes 10 numbered sentences (1 to 10) for each of the 4 languages (Vi, En, Zh, Ko)
from data/mt/manifest.json and outputs standardized 16kHz mono WAV files to outputs/.
"""
import os
import sys
import json
import time
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout
from step3_tts.tts_manager import UnifiedTTSManager, TARGET_SR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_DIR = os.path.join(ROOT, "outputs")


def main():
    print("=" * 70)
    print("  ONEVOICE STEP 3 TTS — BATCH 10-SENTENCE SYNTHESIS FOR 4 LANGUAGES")
    print("=" * 70)

    # 1. Clean existing .wav files in outputs/
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        if f.endswith(".wav"):
            os.remove(os.path.join(OUT_DIR, f))
    print(f"[batch_synth] Cleaned previous .wav files in '{OUT_DIR}'")

    # 2. Load dataset
    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)[:10]  # Take first 10 sentences

    # 3. Initialize UnifiedTTSManager
    manager = UnifiedTTSManager(warmup=True)
    languages = ["vi", "en", "zh", "ko"]
    summary = []

    print("\n[batch_synth] Starting batch synthesis for 10 sentences x 4 languages (40 total WAVs)...")

    for lang in languages:
        print(f"\n--- SYNTHESIZING {lang.upper()} (Sentences 1 to 10) ---")
        for i, row in enumerate(rows, start=1):
            text = row[lang]
            t0 = time.perf_counter()
            res = manager.synthesize(text, lang)
            elapsed = time.perf_counter() - t0

            wav_filename = f"{lang}_{i}.wav"
            wav_path = os.path.join(OUT_DIR, wav_filename)
            sf.write(wav_path, res.audio_array, TARGET_SR)

            summary.append({
                "lang": lang,
                "num": i,
                "filename": wav_filename,
                "text": text[:35] + "...",
                "duration_sec": res.duration_sec,
                "rtf": res.rtf,
                "engine": res.engine,
            })
            print(f"  [{lang.upper()} {i:2d}/10] -> {wav_filename:<10s} | {res.duration_sec:4.2f}s | RTF={res.rtf:.4f} | '{text[:35]}...'")

    print("\n" + "=" * 70)
    print("  BATCH SYNTHESIS COMPLETE: 40/40 WAV FILES GENERATED SUCCESSFULLY ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
