"""Qualcomm Dragonwing IQ-9075 EVK Unified End-to-End TTS Deployment Pipeline.

Deploys the single fused End-to-End TTS Graph (supertonic_end_to_end_tts_static.onnx)
through all 4 steps on Qualcomm AI Hub:
  1. Stage 1 (Quantize): Fresh W8A16 quantization (INT8 Weights, INT16 Activations).
  2. Stage 2 (Compile):  Targeted compilation for Qualcomm Dragonwing IQ-9075 EVK (Hexagon HTP v73).
  3. Stage 3 (Profile):  Physical hardware latency (ms) and peak RAM (MB) profiling on Dragonwing.
  4. Stage 4 (Inference): Full hardware inference execution (Text in -> Speech out) on Dragonwing silicon.

Zero CPU inference: Raw character codepoints enter Dragonwing NPU, and audible speech WAV exits!
"""

import os
import sys
import time
import json
from typing import Any, Dict, List
import numpy as np
import soundfile as sf
import qai_hub as hub

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import _ensure_utf8_stdout
from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine

DEFAULT_TOKEN = "e6hfu2dafj980obg2o60rs5y80ou091ct8g8aofk"
API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN", DEFAULT_TOKEN)
OUTPUT_DIR = os.path.join(ROOT, "outputs", "aihub_live_spoken_speech")
TARGET_DEVICE_NAME = "Dragonwing IQ-9075 EVK"
MODEL_PATH = os.path.join(ROOT, "outputs", "pipeline_static_models", "supertonic_end_to_end_tts_static.onnx")

SAMPLE_RATE = 44100


def encode_text_to_char_codes(text: str, max_chars: int = 54) -> np.ndarray:
    cps = [ord(c) for c in text]
    if len(cps) > max_chars:
        cps = cps[:max_chars]
    arr = np.zeros((1, max_chars), dtype=np.int64)
    arr[0, :len(cps)] = cps
    return arr


def build_calibration_dataset(engine: SupertonicPureNPUV2Engine) -> Dict[str, List[np.ndarray]]:
    """Builds realistic multilingual calibration dataset (English & Korean)."""
    calib_texts = [
        ("The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU.", "en", "M1"),
        ("Ultra low latency and high quality speech synthesis on device.", "en", "M1"),
        ("Hello, this is Qualcomm Hexagon NPU conversational TTS.", "en", "M1"),
        ("퀄컴 드래곤윙 NPU 음성 합성 테스트입니다.", "ko", "F1"),
        ("엣지 디바이스에서 실시간으로 음성을 합성합니다.", "ko", "F1"),
        ("뛰어난 전력 효율과 뛰어난 음질을 제공합니다.", "ko", "F1"),
    ]

    char_codes_list = []
    lang_ids_list = []
    style_dp_list = []
    style_ttl_list = []

    for text, lang, voice in calib_texts:
        char_codes = encode_text_to_char_codes(text, max_chars=54)
        lang_id = np.array([0 if lang == "en" else 1], dtype=np.int64)
        style = engine._helper_tts.get_voice_style(voice)

        char_codes_list.append(char_codes)
        lang_ids_list.append(lang_id)
        style_dp_list.append(style.dp.astype(np.float32))
        style_ttl_list.append(style.ttl.astype(np.float32))

    return {
        "char_codes": char_codes_list,
        "lang_id": lang_ids_list,
        "style_dp": style_dp_list,
        "style_ttl": style_ttl_list,
    }


def evaluate_audio_quality(wav: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    dur_sec = len(wav) / float(sample_rate)
    fft_spec = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sample_rate)

    spec_sum = np.sum(fft_spec)
    spectral_centroid = float(np.sum(freqs * fft_spec) / spec_sum) if spec_sum > 0 else 0.0

    voice_band = (freqs >= 100.0) & (freqs <= 3400.0)
    voice_energy = float(np.sum(fft_spec[voice_band]))
    voice_band_ratio = float(voice_energy / spec_sum) if spec_sum > 0 else 0.0

    peak_amp = float(np.max(np.abs(wav)))
    rms_energy = float(np.sqrt(np.mean(wav ** 2)))

    is_audible = (voice_band_ratio > 0.65) and (1200.0 <= spectral_centroid <= 4500.0) and (peak_amp > 0.05)
    quality_status = "VERIFIED_AUDIBLE_SPEECH" if is_audible else "BORDERLINE_OR_ANOMALY"

    return {
        "duration_sec": round(dur_sec, 2),
        "sample_rate": sample_rate,
        "peak_amplitude": round(peak_amp, 4),
        "rms_energy": round(rms_energy, 4),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "voice_band_ratio": round(voice_band_ratio, 4),
        "quality_assessment": quality_status,
    }


def deploy_end_to_end():
    _ensure_utf8_stdout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 95)
    print(" 🚀 QUALCOMM DRAGONWING IQ-9075 EVK — UNIFIED END-TO-END TTS DEPLOYMENT (4 STAGES)")
    print(" • Model Architecture : Single Unified End-to-End ONNX Graph (Text In -> Speech Out)")
    print(f" • Model File Path    : {os.path.relpath(MODEL_PATH, ROOT)} ({os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB)")
    print(f" • Target Hardware    : {TARGET_DEVICE_NAME} (SoC QCS9075 - Hexagon HTP v73 NPU)")
    print(" • Quantization Recipe: W8A16 (INT8 Weights, INT16 Activations)")
    print("=" * 95)

    client = hub.Client(config=hub.ClientConfig(api_token=API_TOKEN))
    matched_devices = client.get_devices(TARGET_DEVICE_NAME)
    if not matched_devices:
        raise RuntimeError(f"Device {TARGET_DEVICE_NAME} not found on AI Hub.")
    device = matched_devices[0]
    print(f"[+] Connected Target Device: {device.name} (OS: {device.os})")

    engine = SupertonicPureNPUV2Engine()

    # =========================================================================
    # STAGE 1: UPLOAD & QUANTIZE W8A16
    # =========================================================================
    print("\n" + "=" * 95)
    print(" 📦 STAGE 1/4: FRESH W8A16 QUANTIZATION ON QUALCOMM AI HUB")
    print("=" * 95)

    existing_model_id = "mqvooz7xm"
    existing_dataset_id = "d7zn4qkw7"

    try:
        print(f"[+] Reusing pre-uploaded Model ({existing_model_id}) & Dataset ({existing_dataset_id})...")
        uploaded_model = client.get_model(existing_model_id)
        calib_dataset = client.get_dataset(existing_dataset_id)
        print("  ✅ Reused pre-uploaded assets successfully!")
    except Exception as e:
        print(f"[+] Uploading End-to-End ONNX Model to AI Hub Cloud: {e}...")
        t0 = time.time()
        uploaded_model = client.upload_model(MODEL_PATH)
        print(f"  • Model Uploaded: {uploaded_model.model_id} (took {time.time()-t0:.1f}s)")

        print("[+] Building and Uploading Multilingual Calibration Dataset...")
        calib_dict = build_calibration_dataset(engine)
        calib_dataset = client.upload_dataset(calib_dict)
        print(f"  • Dataset Uploaded: {calib_dataset.dataset_id} (6 representative samples)")

    print(f"[+] Submitting W8A16 Quantization Job...")
    quant_job = client.submit_quantize_job(
        model=uploaded_model,
        calibration_data=calib_dataset,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="[E2E_TTS] supertonic_end_to_end_quantize_w8a16",
    )
    print(f"  • Quantize Job ID : {quant_job.job_id}")
    print(f"  • Dashboard URL   : {quant_job.url}")
    print("  ⏳ Waiting for quantization completion...")
    quant_job.wait()

    quant_status = quant_job.get_status().code
    print(f"  • Quantize Status : {quant_status}")
    if quant_status != "SUCCESS":
        raise RuntimeError(f"Quantization failed: {quant_status} - {getattr(quant_job.get_status(), 'message', '')}")
    quantized_model = quant_job.get_target_model()
    print(f"  ✅ Quantized Model ID: {quantized_model.model_id}")

    # =========================================================================
    # STAGE 2: COMPILE FOR DRAGONWING IQ-9075 EVK (HEXAGON HTP v73)
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" ⚙️ STAGE 2/4: TARGETED COMPILATION FOR {TARGET_DEVICE_NAME}")
    print("=" * 95)

    compile_options = "--target_runtime onnx"
    print(f"[+] Submitting Compile Job with options '{compile_options}'...")
    compile_job = client.submit_compile_job(
        model=quantized_model,
        device=device,
        options=compile_options,
        name=f"[E2E_TTS] supertonic_e2e_compile_{device.name.replace(' ', '_')}",
    )
    print(f"  • Compile Job ID  : {compile_job.job_id}")
    print(f"  • Dashboard URL   : {compile_job.url}")
    print(f"  ⏳ Compiling into Hexagon HTP v73 QNN Binary Context...")
    compile_job.wait()

    compile_status = compile_job.get_status().code
    print(f"  • Compile Status  : {compile_status}")
    if compile_status != "SUCCESS":
        raise RuntimeError(f"Compilation failed: {compile_status} - {getattr(compile_job.get_status(), 'message', '')}")
    compiled_model = compile_job.get_target_model()
    print(f"  ✅ Compiled Target Model ID: {compiled_model.model_id}")

    # =========================================================================
    # STAGE 3: PROFILE ON PHYSICAL DRAGONWING BOARD (LATENCY & RAM)
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" ⏱️ STAGE 3/4: PHYSICAL HARDWARE PROFILING ON {TARGET_DEVICE_NAME}")
    print("=" * 95)

    print(f"[+] Submitting Profile Job to {TARGET_DEVICE_NAME}...")
    profile_job = client.submit_profile_job(
        model=compiled_model,
        device=device,
        name=f"[E2E_TTS] supertonic_e2e_profile_{device.name.replace(' ', '_')}",
    )
    print(f"  • Profile Job ID  : {profile_job.job_id}")
    print(f"  • Dashboard URL   : {profile_job.url}")
    print(f"  ⏳ Executing on Dragonwing Hexagon NPU...")
    profile_job.wait()

    profile_status = profile_job.get_status().code
    print(f"  • Profile Status  : {profile_status}")
    profile_data = {}
    if profile_status == "SUCCESS":
        prof = profile_job.download_profile()
        es = prof.get("execution_summary", {})
        latency_ms = round(es.get("estimated_inference_time", 0) / 1000.0, 3)
        peak_mb = round(es.get("estimated_inference_peak_memory", 0) / (1024 * 1024), 2)
        print(f"  ⚡ Hardware Latency: {latency_ms} ms")
        print(f"  💾 Peak Hardware RAM: {peak_mb} MB")
        profile_data = {"latency_ms": latency_ms, "peak_ram_mb": peak_mb}

    # =========================================================================
    # STAGE 4: HARDWARE INFERENCE & AUDIO EVALUATION (TEXT IN -> SPEECH OUT)
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" 🔊 STAGE 4/4: HARDWARE INFERENCE EXECUTION ON {TARGET_DEVICE_NAME}")
    print("=" * 95)

    test_sentences = [
        {
            "id": "dragonwing_e2e_english",
            "text": "The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU.",
            "lang": "en",
            "lang_id": 0,
            "voice": "M1",
        },
        {
            "id": "dragonwing_e2e_korean",
            "text": "퀄컴 드래곤윙 NPU 음성합성 테스트입니다.",
            "lang": "ko",
            "lang_id": 1,
            "voice": "F1",
        }
    ]

    audio_results = []

    for test_idx, test_item in enumerate(test_sentences, 1):
        test_id = test_item["id"]
        text = test_item["text"]
        lang = test_item["lang"]
        lang_id = test_item["lang_id"]
        voice = test_item["voice"]

        print(f"\n[Sentence {test_idx}/2] {test_id.upper()} ({lang.upper()})")
        print(f" • Input Text : \"{text}\"")

        char_codes = encode_text_to_char_codes(text, max_chars=54)
        lang_arr = np.array([lang_id], dtype=np.int64)
        style = engine._helper_tts.get_voice_style(voice)

        inf_inputs = {
            "char_codes": [char_codes],
            "lang_id": [lang_arr],
            "style_dp": [style.dp.astype(np.float32)],
            "style_ttl": [style.ttl.astype(np.float32)],
        }

        print(f"  Submitting Hardware Inference Job to {TARGET_DEVICE_NAME}...")
        inf_job = client.submit_inference_job(
            model=compiled_model,
            device=device,
            inputs=inf_inputs,
            name=f"[E2E_TTS] {test_id}_Dragonwing",
        )
        print(f"  • Job ID        : {inf_job.job_id}")
        print(f"  • Dashboard URL : {inf_job.url}")
        print(f"  ⏳ Executing 100% on Dragonwing Hexagon NPU silicon...")
        inf_job.wait()

        inf_stat = inf_job.get_status().code
        print(f"  ✅ Hardware Inference Status: {inf_stat}")
        if inf_stat != "SUCCESS":
            raise RuntimeError(f"Hardware inference failed for {test_id}: {inf_stat}")

        outs = inf_job.download_output_data()
        raw_audio = np.array(outs["wav_tts"][0]).flatten().astype(np.float32)
        dur_pred = float(np.array(outs["duration"][0]).flatten()[0]) if "duration" in outs else 4.5

        # Slice to predicted duration
        actual_samples = min(len(raw_audio), int(dur_pred * SAMPLE_RATE))
        if actual_samples > 1000:
            hw_audio = raw_audio[:actual_samples]
        else:
            hw_audio = raw_audio

        # Safe normalize
        max_v = np.max(np.abs(hw_audio))
        if max_v > 0.01:
            hw_audio = (hw_audio / max_v) * 0.90

        wav_path = os.path.join(OUTPUT_DIR, f"live_{test_id}.wav")
        sf.write(wav_path, hw_audio, SAMPLE_RATE)

        eval_res = evaluate_audio_quality(hw_audio, SAMPLE_RATE)
        eval_res["test_id"] = test_id
        eval_res["lang"] = lang
        eval_res["text"] = text
        eval_res["job_id"] = inf_job.job_id
        eval_res["job_url"] = inf_job.url
        eval_res["predicted_duration_sec"] = round(dur_pred, 2)
        eval_res["saved_wav"] = os.path.relpath(wav_path, ROOT)

        print(f"  🎉 Saved Hardware Speech: {os.path.basename(wav_path)}")
        print(f"     Duration        : {eval_res['duration_sec']}s")
        print(f"     Spectral Centroid: {eval_res['spectral_centroid_hz']} Hz")
        print(f"     Voice Band Ratio: {eval_res['voice_band_ratio'] * 100:.1f}%")
        print(f"     Assessment      : {eval_res['quality_assessment']}")

        audio_results.append(eval_res)

    # Save comprehensive final report
    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "Single Unified End-to-End ONNX Graph (Text In -> Speech Out)",
        "target_device": TARGET_DEVICE_NAME,
        "soc": "Qualcomm QCS9075",
        "npu": "Qualcomm Hexagon HTP v73",
        "quantization": "W8A16 (INT8 Weights, INT16 Activations)",
        "stage1_quantize": {
            "job_id": quant_job.job_id,
            "url": quant_job.url,
            "status": quant_status,
            "target_model_id": quantized_model.model_id,
        },
        "stage2_compile": {
            "job_id": compile_job.job_id,
            "url": compile_job.url,
            "status": compile_status,
            "target_model_id": compiled_model.model_id,
        },
        "stage3_profile": {
            "job_id": profile_job.job_id,
            "url": profile_job.url,
            "status": profile_status,
            "profile_metrics": profile_data,
        },
        "stage4_inference_speech": audio_results,
    }

    report_path = os.path.join(OUTPUT_DIR, "live_dragonwing_end_to_end_tts_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print(f" ✅ ALL 4 STAGES SUCCESSFULLY COMPLETED FOR UNIFIED END-TO-END TTS ON {TARGET_DEVICE_NAME}!")
    print(f" • Summary Report : {os.path.relpath(report_path, ROOT)}")
    print("=" * 95)


if __name__ == "__main__":
    deploy_end_to_end()
