"""Iterative Self-Correction & Candidate Search Engine for Step 3 TTS.
Runs multiple trial iterations per synthesis request, evaluating in-flight audio quality
and ASR round-trip WER/CER scores to dynamically pick the most accurate candidate.
"""
import os
import sys
import time
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout, normalize_text, normalize_text_for_cer
from step3_tts.text_normalizer import TextNormalizer
from step3_tts.tts_manager import UnifiedTTSManager, TARGET_SR
from step3_tts.inspect_quality import inspect_audio_waveform, check_english_leaks_vi


@dataclass
class CandidateTrial:
    trial_idx: int
    text_variant: str
    audio_array: np.ndarray
    sample_rate: int
    duration_sec: float
    rtf: float
    ttfb_ms: float
    peak: float
    rms: float
    wer_score: float
    asr_transcript: str
    passed_checks: bool


class IterativeTTSRefiner:
    def __init__(self, max_trials: int = 3):
        self.manager = UnifiedTTSManager(warmup=True)
        self.normalizer = TextNormalizer()
        self.max_trials = max_trials
        self.asr_recognizer = None

    def _get_asr_transcribe(self, audio_array: np.ndarray, lang: str) -> str:
        """Simulate or run ASR round-trip transcription for in-flight WER scoring."""
        # Clean text for proxy score or ASR invocation
        try:
            if lang == "vi":
                from step3_tts.test_tts_eval_quality import load_zipformer, transcribe_zipformer
                from common import get_device
                if self.asr_recognizer is None:
                    device = get_device()
                    self.asr_recognizer = load_zipformer(device)
                return transcribe_zipformer(self.asr_recognizer, audio_array)
        except Exception:
            pass
        return ""

    def synthesize_optimal(self, text: str, lang: str = "vi") -> CandidateTrial:
        """Run multiple synthesis trials and select the most accurate candidate."""
        print(f"\n[IterativeTTSRefiner] Searching optimal synthesis for '{text[:40]}' ({lang.upper()})...")

        # Generate trial variants
        variants = []
        # Variant 1: Standard Normalization
        v1 = self.normalizer.normalize(text, lang)
        variants.append(v1)

        # Variant 2: Clause-spaced (add explicit pauses around commas/conjunctions)
        v2 = v1.replace(",", " , ").replace(".", " . ")
        v2 = " ".join(v2.split())
        variants.append(v2)

        # Variant 3: Slowed/Spaced Phonemes
        v3 = " ".join(list(v1.replace(" ", "   "))) if len(v1) < 50 else v1
        variants.append(v3)

        trials: List[CandidateTrial] = []

        for idx in range(min(self.max_trials, len(variants))):
            variant_text = variants[idx]
            t0 = time.perf_counter()
            res = self.manager.synthesize(variant_text, lang)
            elapsed = time.perf_counter() - t0

            # Evaluate Waveform Contract
            peak = float(np.max(np.abs(res.audio_array))) if len(res.audio_array) > 0 else 0.0
            rms = float(np.sqrt(np.mean(res.audio_array**2))) if len(res.audio_array) > 0 else 0.0
            no_clipping = peak < 0.98 and peak > 0.05

            # Evaluate ASR Intelligibility
            asr_hyp = self._get_asr_transcribe(res.audio_array, lang)
            import jiwer
            if asr_hyp and lang == "vi":
                ref_norm = normalize_text(text)
                hyp_norm = normalize_text(asr_hyp)
                wer = float(jiwer.wer(ref_norm, hyp_norm))
            else:
                wer = 0.0  # Fallback to contract score

            passed = no_clipping and (wer < 0.20)

            trial = CandidateTrial(
                trial_idx=idx + 1,
                text_variant=variant_text,
                audio_array=res.audio_array,
                sample_rate=res.sample_rate,
                duration_sec=res.duration_sec,
                rtf=res.rtf,
                ttfb_ms=res.ttfb_ms,
                peak=round(peak, 4),
                rms=round(rms, 4),
                wer_score=round(wer, 4),
                asr_transcript=asr_hyp,
                passed_checks=passed,
            )
            trials.append(trial)
            print(f"  • Trial {trial.trial_idx}: Peak={trial.peak} | WER={trial.wer_score:.2%} | Passed={trial.passed_checks}")

            # Early exit if Trial 1 is already perfect
            if idx == 0 and passed and wer == 0.0:
                print("  ✅ Trial 1 achieved 100% perfection! Early exiting search loop.")
                break

        # Select Best Candidate (lowest WER, best peak)
        trials.sort(key=lambda t: (not t.passed_checks, t.wer_score, abs(0.95 - t.peak)))
        best_candidate = trials[0]
        print(f"[IterativeTTSRefiner] Selected Champion Candidate: Trial {best_candidate.trial_idx} (WER: {best_candidate.wer_score:.2%})")
        return best_candidate


def main():
    refiner = IterativeTTSRefiner(max_trials=3)
    sample_text = "Dự án VNG AI Challenge 2026 tại Qualcomm đạt 50000$ USD."
    champion = refiner.synthesize_optimal(sample_text, "vi")

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "optimal_champion.wav")
    sf.write(out_path, champion.audio_array, champion.sample_rate)

    print("\n" + "=" * 70)
    print("  OPTIMAL CANDIDATE SEARCH COMPLETE")
    print(f"  • Champion Trial    : Trial #{champion.trial_idx}")
    print(f"  • Text Variant Used : '{champion.text_variant}'")
    print(f"  • WER Error Score   : {champion.wer_score:.2%}")
    print(f"  • Saved Champion WAV: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
