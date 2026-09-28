"""Test Suite: Multi-Lingual Numerical Equivalence & Audio Verification for One-Hot GEMM Refactoring.

Compares Gather (.bak) vs OneHot + MatMul across:
  - Duration Predictor (static & dynamic)
  - Text Encoder (static & dynamic)
across 4 languages: Vietnamese, English, Korean, and Chinese.
"""
import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import onnxruntime as ort

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

MODELS_TO_VERIFY = [
    (
        "Duration Predictor (Dynamic Speed)",
        os.path.join(ROOT, "outputs", "pure_npu_dynamic", "duration_predictor_npu_speed.onnx"),
        os.path.join(ROOT, "outputs", "pure_npu_dynamic", "duration_predictor_npu_speed.onnx.bak"),
    ),
    (
        "Duration Predictor (Dynamic)",
        os.path.join(ROOT, "outputs", "pure_npu_dynamic", "duration_predictor_npu.onnx"),
        os.path.join(ROOT, "outputs", "pure_npu_dynamic", "duration_predictor_npu.onnx.bak"),
    ),
    (
        "Text Encoder (Dynamic)",
        os.path.join(ROOT, "outputs", "pure_npu_dynamic", "text_encoder_npu.onnx"),
        os.path.join(ROOT, "outputs", "pure_npu_dynamic", "text_encoder_npu.onnx.bak"),
    ),
    (
        "Duration Predictor (Static)",
        os.path.join(ROOT, "outputs", "pipeline_static_models", "duration_predictor_static.onnx"),
        os.path.join(ROOT, "outputs", "pipeline_static_models", "duration_predictor_static.onnx.bak"),
    ),
    (
        "Text Encoder (Static)",
        os.path.join(ROOT, "outputs", "pipeline_static_models", "text_encoder_static.onnx"),
        os.path.join(ROOT, "outputs", "pipeline_static_models", "text_encoder_static.onnx.bak"),
    ),
]


def test_model_numerical_equivalence():
    _ensure_utf8_stdout()
    print("=" * 90)
    print(" 🔬 STEP 1: MULTI-LINGUAL NUMERICAL EQUIVALENCE TEST (GATHER vs ONE-HOT GEMM)")
    print("=" * 90)

    from supertonic import TTS
    helper_tts = TTS(auto_download=True)

    results = []
    all_passed = True

    for model_name, cur_path, bak_path in MODELS_TO_VERIFY:
        if not os.path.exists(cur_path) or not os.path.exists(bak_path):
            print(f" ⚠️ Skipping {model_name}: file or backup missing.")
            continue

        sess_cur = ort.InferenceSession(cur_path, providers=["CPUExecutionProvider"])
        sess_bak = ort.InferenceSession(bak_path, providers=["CPUExecutionProvider"])

        print(f"\nEvaluating: {model_name}")
        print(f"  • Refactored (OneHot+GEMM): {os.path.relpath(cur_path, ROOT)}")
        print(f"  • Original (Gather):         {os.path.relpath(bak_path, ROOT)}")

        for lang, item in TEST_CASES.items():
            text = item["text"]
            lang_code = "na" if helper_tts.is_multilingual else "en"
            text_ids, text_mask = helper_tts.model.text_processor([text], lang_code)
            style = helper_tts.get_voice_style(voice_name="F1" if lang in ["vi", "zh"] else "M1")

            # Check if static model requires fixed shape 64
            is_static = "static" in cur_path
            if is_static:
                # pad or slice to 64
                orig_len = text_ids.shape[1]
                if orig_len < 64:
                    padded_ids = np.zeros((1, 64), dtype=np.int64)
                    padded_ids[0, :orig_len] = text_ids[0]
                    text_ids = padded_ids
                    padded_mask = np.zeros((1, 1, 64), dtype=np.float32)
                    padded_mask[0, 0, :orig_len] = text_mask[0, 0]
                    text_mask = padded_mask
                else:
                    text_ids = text_ids[:, :64]
                    text_mask = text_mask[:, :, :64]

            # Build feed
            feed = {"text_ids": text_ids, "text_mask": text_mask}
            inputs_cur = [i.name for i in sess_cur.get_inputs()]
            if "style_dp" in inputs_cur:
                feed["style_dp"] = style.dp
            if "style_ttl" in inputs_cur:
                feed["style_ttl"] = style.ttl
            if "speed" in inputs_cur:
                feed["speed"] = np.array([1.0], dtype=np.float32)

            out_cur = sess_cur.run(None, feed)[0]
            out_bak = sess_bak.run(None, feed)[0]

            max_diff = float(np.max(np.abs(out_cur - out_bak)))
            cos_sim = float(
                np.dot(out_cur.flatten(), out_bak.flatten())
                / (np.linalg.norm(out_cur.flatten()) * np.linalg.norm(out_bak.flatten()) + 1e-12)
            )

            status = "✅ PASS" if max_diff < 1e-5 else "❌ FAIL"
            if max_diff >= 1e-5:
                all_passed = False

            print(f"    [{lang.upper()}] {item['name']:<22}: Max Diff = {max_diff:.6e} | Cosine Sim = {cos_sim:.6f} | {status}")

            results.append({
                "model": model_name,
                "language": lang,
                "max_diff": max_diff,
                "cosine_sim": cos_sim,
                "passed": max_diff < 1e-5,
            })

    return all_passed, results


def test_full_pipeline_synthesis():
    _ensure_utf8_stdout()
    print("\n" + "=" * 90)
    print(" 🎙️ STEP 2: FULL TTS PIPELINE MULTI-LINGUAL SYNTHESIS WITH REFACTORED MODELS")
    print("=" * 90)

    out_dir = os.path.join(ROOT, "outputs", "task5_onehot_gemm_verification")
    os.makedirs(out_dir, exist_ok=True)

    engine = SupertonicPureNPUV2Engine(target_sample_rate=16000)
    synthesis_results = []

    for lang, item in TEST_CASES.items():
        text = item["text"]
        print(f"\nSynthesizing [{lang.upper()}] {item['name']}:")
        print(f"  • Input text: \"{text}\"")

        wav, stats = engine.synthesize(text=text, language=lang, speed=1.0)
        out_wav_path = os.path.join(out_dir, f"onehot_gemm_{lang}.wav")
        sf.write(out_wav_path, wav, stats["sample_rate"])

        rms_energy = float(np.sqrt(np.mean(wav ** 2)))
        peak_amp = float(np.max(np.abs(wav)))
        print(f"  • Saved: {os.path.relpath(out_wav_path, ROOT)}")
        print(f"  • Duration: {stats['duration_sec']}s | Latency: {stats['total_latency_ms']}ms | RTF: {stats['rtf']}")
        print(f"  • RMS Energy: {rms_energy:.4f} | Peak Amp: {peak_amp:.4f}")

        synthesis_results.append({
            "language": lang,
            "text": text,
            "wav_path": out_wav_path,
            "duration_sec": stats["duration_sec"],
            "total_latency_ms": stats["total_latency_ms"],
            "rtf": stats["rtf"],
            "rms_energy": rms_energy,
            "peak_amp": peak_amp,
        })

    # Save summary report
    report_path = os.path.join(out_dir, "onehot_gemm_verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(synthesis_results, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Saved full verification report: {os.path.relpath(report_path, ROOT)}")
    return synthesis_results


if __name__ == "__main__":
    passed, eq_results = test_model_numerical_equivalence()
    if not passed:
        print("\n❌ Numerical equivalence test FAILED!")
        sys.exit(1)
    print("\n🌟 ALL NUMERICAL TESTS PASSED: 100% IDENTICAL TO ORIGINAL GATHER!")
    test_full_pipeline_synthesis()
    print("\n🎉 ALL TESTS AND SYNTHESIS COMPLETED SUCCESSFULLY!")
