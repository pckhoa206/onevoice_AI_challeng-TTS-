"""CLI Synthesis Tool for Step 3 TTS.
Allows instant testing of any custom text string in Vietnamese (vi), English (en), Chinese (zh), or Korean (ko).
"""
import os
import sys
import argparse
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout
from step3_tts.text_normalizer import TextNormalizer
from step3_tts.tts_manager import UnifiedTTSManager, TARGET_SR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT_WAV = os.path.join(ROOT, "outputs", "custom_test.wav")


def main():
    parser = argparse.ArgumentParser(description="Step 3 TTS Custom Text Test CLI")
    parser.add_argument("--text", "-t", type=str, required=True, help="Custom text to synthesize")
    parser.add_argument("--lang", "-l", type=str, default="vi", choices=["vi", "en", "zh", "ko"], help="Target language code")
    parser.add_argument("--out", "-o", type=str, default=DEFAULT_OUT_WAV, help="Output WAV file path")
    args = parser.parse_args()

    print("=" * 60)
    print("  ONEVOICE STEP 3 TTS — CUSTOM TEXT SYNTHESIS TEST")
    print("=" * 60)
    print(f" Input Text : '{args.text}'")
    print(f" Language   : {args.lang.upper()}")

    # 1. Text Normalization
    norm = TextNormalizer()
    clean_text = norm.normalize(args.text, args.lang)
    print(f" Normalized : '{clean_text}'")

    # 2. Synthesis via UnifiedTTSManager
    print("\n[CLI] Loading UnifiedTTSManager...")
    manager = UnifiedTTSManager(warmup=True)

    print(f"[CLI] Synthesizing audio for {args.lang.upper()}...")
    res = manager.synthesize(args.text, args.lang)

    # 3. Save WAV file
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, res.audio_array, TARGET_SR)

    print("-" * 60)
    print(f" ✅ Synthesis Complete!")
    print(f" Engine Used   : {res.engine}")
    print(f" Sample Rate   : {res.sample_rate} Hz (Mono 16-bit PCM)")
    print(f" Audio Length  : {res.duration_sec:.2f} seconds")
    print(f" Time-to-First  : {res.ttfb_ms:.1f} ms")
    print(f" RTF (Speed)   : {res.rtf:.4f} ({'Faster' if res.rtf < 1 else 'Slower'} than real-time)")
    print(f" Saved WAV To  : {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
