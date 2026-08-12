"""Batch Optimal Synthesis & Candidate Refinement for Step 3 TTS.
Runs the Iterative Candidate Search Engine (IterativeTTSRefiner) across ALL 10 sentences
for ALL 4 languages (Vi, En, Zh, Ko).
Evaluates audio waveform contracts & in-flight round-trip WER/CER scores to select
and save the OPTIMAL CHAMPION candidate WAV for every single sentence.
"""
import os
import sys
import json
import time
import csv
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout
from step3_tts.iterative_refiner import IterativeTTSRefiner
from step3_tts.tts_manager import TARGET_SR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
OUT_DIR = os.path.join(ROOT, "outputs")
AUDIT_CSV = os.path.join(OUT_DIR, "optimal_candidates_audit.csv")


def main():
    print("=" * 75)
    print("  ONEVOICE STEP 3 TTS — OPTIMAL CANDIDATE SEARCH BATCH (40 SENTENCES)")
    print("=" * 75)

    # 1. Load dataset (first 10 sentences)
    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)[:10]

    os.makedirs(OUT_DIR, exist_ok=True)
    refiner = IterativeTTSRefiner(max_trials=3)
    languages = ["vi", "en", "zh", "ko"]
    audit_log = []

    print("\n[batch_optimal_synth] Evaluating candidate search loop for 10 sentences x 4 languages...")

    for lang in languages:
        print(f"\n==================================================")
        print(f"  OPTIMIZING {lang.upper()} SENTENCES (1 to 10)")
        print(f"==================================================")
        for i, row in enumerate(rows, start=1):
            raw_text = row[lang]

            # Run in-flight trial search & candidate selection
            champion = refiner.synthesize_optimal(raw_text, lang)

            # Save optimal WAV file
            wav_filename = f"{lang}_{i}.wav"
            wav_path = os.path.join(OUT_DIR, wav_filename)
            sf.write(wav_path, champion.audio_array, TARGET_SR)

            audit_entry = {
                "lang": lang,
                "sentence_idx": i,
                "filename": wav_filename,
                "champion_trial": champion.trial_idx,
                "wer_score": champion.wer_score,
                "peak_amplitude": champion.peak,
                "rms_energy": champion.rms,
                "duration_sec": champion.duration_sec,
                "rtf": champion.rtf,
                "ttfb_ms": champion.ttfb_ms,
                "raw_text": raw_text[:40] + "...",
                "normalized_variant": champion.text_variant[:40] + "...",
                "wav_path": wav_path,
            }
            audit_log.append(audit_entry)

            print(f"  🏆 [{lang.upper()} {i:2d}/10] Champion: Trial #{champion.trial_idx} | WER={champion.wer_score:.2%} | Peak={champion.peak} | Saved -> {wav_filename}")

    # Write Audit CSV
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_log[0].keys()))
        writer.writeheader()
        writer.writerows(audit_log)

    print("\n" + "=" * 75)
    print(f"  OPTIMAL CANDIDATE SEARCH BATCH COMPLETE ✅")
    print(f"  • Total Sentences Evaluated : 40/40")
    print(f"  • Champion WAVs Overwritten : outputs/<lang>_<1-10>.wav")
    print(f"  • Complete Audit Report     : {AUDIT_CSV}")
    print("=" * 75)


if __name__ == "__main__":
    main()
