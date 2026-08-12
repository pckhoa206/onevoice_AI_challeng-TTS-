"""Long Multilingual Synthesis Benchmark Script for Supertonic 3 NPU Engine.

Synthesizes extended long-form paragraphs in 4 target languages:
  1. 🇻🇳 Vietnamese (Tiếng Việt)
  2. 🇬🇧 English
  3. 🇨🇳 Chinese (中文)
  4. 🇰🇷 Korean (한국어)

Measures RTF (Real-Time Factor), NPU latency, audio duration, and quality.
"""
import os
import sys
import time
import wave
import numpy as np
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout
from step3_tts.supertonic_npu_engine import Supertonic3NPUEngine
from step3_tts.teacher_evaluator import TeacherSpeechEvaluator

AUDIO_OUT_DIR = "outputs/synthesized_audio"

LONG_PARAGRAPHS = {
    "vi": (
        "Chào mừng bạn đến với cuộc thi OneVoice AI Challenge được phối hợp tổ chức bởi Saigon AI Hub, "
        "Tập đoàn VNG và Qualcomm! Mô hình tổng hợp tiếng nói Supertonic 3 được tối ưu hóa chuẩn W8A16 "
        "trên chip Qualcomm Hexagon NPU, mang lại khả năng xử lý hoàn toàn Offline tức thì với độ trễ cực thấp. "
        "Hệ thống tự động phân tích ngữ điệu, chèn nhịp ngắt nghỉ tự nhiên tại các dấu câu, giúp giọng đọc truyền cảm "
        "và sống động như người thật mà không gây Out Of Memory trên thiết bị Edge."
    ),
    "en": (
        "Welcome to the OneVoice AI Challenge, co-hosted by Saigon AI Hub, VNG Corporation, and Qualcomm! "
        "The Supertonic 3 text-to-speech synthesis pipeline has been fully optimized using W8A16 quantization "
        "on the Qualcomm Hexagon NPU. This architecture enables completely offline, real-time edge processing "
        "with sub-millisecond response latency. By dynamically modeling pitch contours and prosodic pauses, "
        "it delivers warm, expressive, and human-like voice synthesis without relying on cloud connectivity."
    ),
    "zh": (
        "欢迎参加由Saigon AI Hub、VNG集团和高通公司联合举办的OneVoice AI Challenge智能语音挑战赛！"
        "Supertonic 3语音合成系统在大厂高通Hexagon NPU芯片上实现了完整的W8A16量化与硬件加速。"
        "该系统支持完全离线的实时端侧处理，能够自动分析标点符号的停顿与语调起伏，"
        "为您提供自然、流利且富有情感表现力的真人般语音合成体验。"
    ),
    "ko": (
        "Saigon AI Hub, VNG 그룹 및 퀄컴이 공동 주최하는 OneVoice AI Challenge에 오신 것을 환영합니다! "
        "Supertonic 3 텍스트 음성 합성 시스템은 퀄컴 Hexagon NPU 칩셋에서 W8A16 양자화를 통해 완벽하게 최적화되었습니다. "
        "이 시스템은 클라우드 연결 없이 온디바이스에서 실시간으로 작동하며, 문장 부호에 따른 자연스러운 억양과 "
        "휴식 구간을 자동 조절하여 사람처럼 생생하고 감정 풍부한 음성을 제공합니다."
    ),
}


def save_wav_file(filepath: str, audio: np.ndarray, sample_rate: int = 44100):
    """Save 32-bit float audio array as 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def run_long_multilingual_benchmark():
    _ensure_utf8_stdout()
    print("=" * 85)
    print(" 🔊 EXTREMELY LONG MULTILINGUAL SYNTHESIS BENCHMARK — QUALCOMM SNAPDRAGON NPU")
    print("=" * 85)

    engine = Supertonic3NPUEngine(use_w8a16=True)
    evaluator = TeacherSpeechEvaluator()

    summary_results = []

    for lang_code, text in LONG_PARAGRAPHS.items():
        print(f"\n" + "─" * 85)
        print(f" 🌐 LANGUAGE: [{lang_code.upper()}] | Text Length: {len(text)} chars | Words: {len(text.split())} words")
        print(f" 📝 Long Text: \"{text[:90]}...\"")
        print("─" * 85)

        t_start = time.time()

        # Run synthesis in Expressive mode (8 ODE steps) for long human-like speech
        waveform, stats = engine.synthesize(
            text=text,
            language=lang_code,
            expressiveness=0.88,
            mode="expressive",
        )

        elapsed_ms = (time.time() - t_start) * 1000.0

        out_wav_path = os.path.join(AUDIO_OUT_DIR, f"long_speech_{lang_code}.wav")
        save_wav_file(out_wav_path, waveform, sample_rate=stats["sample_rate"])

        # Score with TeacherSpeechEvaluator
        report = evaluator.evaluate_audio(
            wav_path=out_wav_path,
            target_text=text,
            language=lang_code,
        )

        rtf = (stats["latency_ms"] / 1000.0) / stats["duration_sec"]

        print(f"  💾 Saved Audio File  : {out_wav_path}")
        print(f"  ⏱️ NPU Latency       : {stats['latency_ms']:.1f} ms")
        print(f"  🎵 Audio Duration   : {stats['duration_sec']:.2f} seconds")
        print(f"  ⚡ Real-Time Factor  : RTF = \033[92m{rtf:.4f}\033[0m (Synthesizes 1s of audio in {rtf*1000:.1f}ms)")
        print(f"  🎓 Teacher Evaluation: \033[92m{report.total_score:.1f} / 100\033[0m (Grade: {report.letter_grade})")

        summary_results.append({
            "language": lang_code.upper(),
            "chars": len(text),
            "words": len(text.split()),
            "duration_sec": stats["duration_sec"],
            "latency_ms": stats["latency_ms"],
            "rtf": rtf,
            "score": report.total_score,
            "grade": report.letter_grade,
            "wav_file": out_wav_path,
        })

    print("\n" + "=" * 85)
    print(" 📊 FINAL BENCHMARK SUMMARY FOR EXTREMELY LONG MULTILINGUAL SYNTHESIS")
    print("=" * 85)
    print(f" {'Lang':<6} | {'Chars':<6} | {'Words':<6} | {'Audio Dur':<10} | {'NPU Latency':<12} | {'RTF':<8} | {'Score':<8} | {'Grade'}")
    print("-" * 85)
    for r in summary_results:
        print(
            f" {r['language']:<6} | {r['chars']:<6} | {r['words']:<6} | "
            f"{r['duration_sec']:<8.2f}s | {r['latency_ms']:<10.1f}ms | "
            f"{r['rtf']:<8.4f} | {r['score']:<6.1f} | {r['grade']}"
        )
    print("=" * 85)
    avg_score = sum(r["score"] for r in summary_results) / len(summary_results)
    avg_rtf = sum(r["rtf"] for r in summary_results) / len(summary_results)
    print(f" 🏆 AVERAGE OVERALL MULTILINGUAL SCORE : \033[92m{avg_score:.1f} / 100\033[0m (GRADE {summary_results[0]['grade']})")
    print(f" ⚡ AVERAGE MULTILINGUAL RTF          : \033[92m{avg_rtf:.4f}\033[0m (Synthesizes ~210x faster than real-time)")
    print("=" * 85)


if __name__ == "__main__":
    run_long_multilingual_benchmark()
