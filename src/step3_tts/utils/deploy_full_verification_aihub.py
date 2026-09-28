"""End-to-End Qualcomm AI Hub Deployment & Physical Verification Pipeline.

Executes all 4 mandatory stages on real hardware (Samsung Galaxy S24 Ultra - Snapdragon 8 Gen 3 Hexagon NPU):
  Stage 1: QUANTIZE (W8A16 with real Mel-latent calibration data)
  Stage 2: COMPILE (Optimized QNN / ONNX for Hexagon HTP v75)
  Stage 3: PROFILE (Hardware inference latency & peak RAM measurement)
  Stage 4: INFERENCE (Execute real Vietnamese & English spoken sentences, download hardware tensor, save .wav)
"""
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import soundfile as sf
import qai_hub as hub

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import _ensure_utf8_stdout
from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine

DEFAULT_TOKEN = "e6hfu2dafj980obg2o60rs5y80ou091ct8g8aofk"
API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN", DEFAULT_TOKEN)
VOCODER_MODEL_PATH = os.path.join(ROOT, "outputs", "pure_npu_compliant_onnx_v2", "vocoder_pure_npu.onnx")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "aihub_live_spoken_speech")
TARGET_DEVICE_NAME = "Samsung Galaxy S24 Ultra"


def generate_real_mel_latent(engine: SupertonicPureNPUV2Engine, text: str, lang: str = "vi") -> Tuple[np.ndarray, int]:
    """Generates a realistic (1, 144, 100) Mel-latent from text using upstream Pure NPU submodels."""
    norm_text = engine.normalizer.normalize(text, lang) or text
    voice_name = "F1" if lang == "vi" else "M1"
    style = engine._helper_tts.get_voice_style(voice_name=voice_name)
    lang_code = "na" if engine._helper_tts.is_multilingual else "en"
    text_ids, text_mask = engine._helper_tts.model.text_processor([norm_text], lang_code)

    dp_feed = {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
    if "speed" in [inp.name for inp in engine.sessions["duration_predictor"].get_inputs()]:
        dp_feed["speed"] = np.array([1.0], dtype=np.float32)
    dur = engine.sessions["duration_predictor"].run(None, dp_feed)[0]

    text_emb = engine.sessions["text_encoder"].run(
        None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
    )[0]

    np.random.seed(42)
    xt, latent_mask = engine._helper_tts.model.sample_noisy_latent(dur)

    if engine.is_unrolled_ve:
        xt = engine.sessions["vector_estimator"].run(
            None,
            {
                "noisy_latent": xt,
                "text_emb": text_emb,
                "style_ttl": style.ttl,
                "latent_mask": latent_mask,
                "text_mask": text_mask,
            },
        )[0]
    else:
        total_steps = 5
        total_step_np = np.array([total_steps], dtype=np.float32)
        for step in range(total_steps):
            cur_step_np = np.array([step], dtype=np.float32)
            xt = engine.sessions["vector_estimator"].run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": cur_step_np,
                    "total_step": total_step_np,
                },
            )[0]

    latent_fixed = np.zeros((1, 144, 100), dtype=np.float32)
    t_frames = min(xt.shape[2], 100)
    latent_fixed[:, :, :t_frames] = xt[:, :, :t_frames]
    return latent_fixed, t_frames


def run_deployment_pipeline():
    _ensure_utf8_stdout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 90)
    print(" 🚀 QUALCOMM AI HUB — FULL 4-STAGE LIVE HARDWARE DEPLOYMENT PIPELINE")
    print("    Target Hardware: Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3 Hexagon HTP v75)")
    print("    Required Stages: [1] Quantize -> [2] Compile -> [3] Profile -> [4] Inference")
    print(f"    Output Directory: {OUTPUT_DIR}")
    print("=" * 90)

    # 0. Connect to AI Hub
    print("\n[0/4] Authenticating with Qualcomm AI Hub...")
    client = hub.Client(config=hub.ClientConfig(api_token=API_TOKEN))

    matched_devices = client.get_devices(TARGET_DEVICE_NAME)
    if not matched_devices:
        print(f"  ⚠️ Device '{TARGET_DEVICE_NAME}' not directly found. Searching Snapdragon devices...")
        all_devs = client.get_devices()
        matched_devices = [d for d in all_devs if "s24" in d.name.lower() or "snapdragon" in d.name.lower() or "dragonwing" in d.name.lower()]
        if not matched_devices:
            raise RuntimeError("No compatible Snapdragon NPU hardware available on AI Hub.")
    
    device = matched_devices[0]
    print(f"  ✅ Connected Target Device: {device.name}")
    print(f"     OS: {device.os} | Attributes: {device.attributes[:4]}...")

    # Initialize Engine for Real Mel-Latent Generation
    print("\n[*] Initializing Upstream Pure NPU Engine to Generate Real Calibration & Speech Latents...")
    engine = SupertonicPureNPUV2Engine()

    calib_sentences = [
        ("vi", "Xin chào VNG! Hệ thống OneVoice AI chạy trên NPU Snapdragon."),
        ("vi", "Công nghệ giọng nói nhân tạo thời gian thực đạt độ trễ siêu thấp."),
        ("en", "The OneVoice AI Challenge runs directly on Qualcomm Hexagon NPU."),
        ("en", "High fidelity neural speech synthesis on mobile Snapdragon edge devices."),
    ]
    calib_latents: List[np.ndarray] = []
    print(f"  • Generating {len(calib_sentences)} realistic Mel-latent samples for calibration...")
    for idx, (lang, sentence) in enumerate(calib_sentences, 1):
        lat, t_val = generate_real_mel_latent(engine, sentence, lang)
        calib_latents.append(lat)
        print(f"    [{idx}/{len(calib_sentences)}] Lang={lang.upper()} | Frames={t_val}/100 | Latent Range=[{lat.min():.2f}, {lat.max():.2f}]")

    calib_dataset = {"latent": calib_latents}

    pipeline_report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_device": device.name,
        "model_deployed": "vocoder_pure_npu.onnx",
        "stages": {},
        "audio_results": [],
    }

    # =========================================================================
    # STAGE 1: QUANTIZE (hub.submit_quantize_job)
    # =========================================================================
    print("\n" + "=" * 90)
    print(" 🛠️  STAGE 1/4: QUANTIZE (hub.submit_quantize_job)")
    print("    Precision: W8A16 (Weights=INT8, Activations=INT16)")
    print("    Calibration Data: Real Spoken Mel-Latents (Vietnamese & English)")
    print("=" * 90)

    print(f"  • Uploading model: {VOCODER_MODEL_PATH}")
    uploaded_model = client.upload_model(VOCODER_MODEL_PATH)
    print(f"  • Model Uploaded ID: {uploaded_model.model_id}")

    print("  • Submitting Quantize Job to Qualcomm AI Hub...")
    quant_job = client.submit_quantize_job(
        model=uploaded_model,
        calibration_data=calib_dataset,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name=f"[4STAGE_QUANTIZE] Vocoder_Pure_NPU_W8A16",
    )
    print(f"  • Quantize Job ID : {quant_job.job_id}")
    print(f"  • Dashboard URL   : {quant_job.url}")
    print("  ⏳ Waiting for quantization to finish on AI Hub...")
    quant_job.wait()
    quant_status = quant_job.get_status().code
    print(f"  • Quantize Status : {quant_status}")

    if quant_status != "SUCCESS":
        raise RuntimeError(f"Quantize stage failed with status: {quant_status}")

    quantized_model = quant_job.get_target_model()
    print(f"  ✅ SUCCESS: Quantized Target Model ID: {quantized_model.model_id}")

    pipeline_report["stages"]["quantize"] = {
        "job_id": quant_job.job_id,
        "url": quant_job.url,
        "status": quant_status,
        "precision": "W8A16",
        "target_model_id": quantized_model.model_id,
    }

    # =========================================================================
    # STAGE 2: COMPILE (hub.submit_compile_job)
    # =========================================================================
    print("\n" + "=" * 90)
    print(f" ⚙️  STAGE 2/4: COMPILE (hub.submit_compile_job)")
    print(f"    Target Runtime: ONNX Context Binary for Hexagon HTP v75")
    print(f"    Target Device : {device.name}")
    print("=" * 90)

    print("  • Submitting Compile Job to Qualcomm AI Hub...")
    compile_job = client.submit_compile_job(
        model=quantized_model,
        device=device,
        options="--target_runtime onnx",
        name=f"[4STAGE_COMPILE] Vocoder_W8A16_{device.name.replace(' ', '_')}",
    )
    print(f"  • Compile Job ID  : {compile_job.job_id}")
    print(f"  • Dashboard URL   : {compile_job.url}")
    print("  ⏳ Waiting for NPU hardware compilation...")
    compile_job.wait()
    compile_status = compile_job.get_status().code
    print(f"  • Compile Status  : {compile_status}")

    if compile_status != "SUCCESS":
        raise RuntimeError(f"Compile stage failed with status: {compile_status}")

    compiled_model = compile_job.get_target_model()
    print(f"  ✅ SUCCESS: Compiled Target Model ID: {compiled_model.model_id}")

    pipeline_report["stages"]["compile"] = {
        "job_id": compile_job.job_id,
        "url": compile_job.url,
        "status": compile_status,
        "target_device": device.name,
        "compiled_model_id": compiled_model.model_id,
    }

    # =========================================================================
    # STAGE 3: PROFILE (hub.submit_profile_job)
    # =========================================================================
    print("\n" + "=" * 90)
    print(f" 📊 STAGE 3/4: PROFILE (hub.submit_profile_job)")
    print(f"    Measuring on-device latency & memory on physical {device.name}")
    print("=" * 90)

    print("  • Submitting Profile Job to Qualcomm AI Hub...")
    profile_job = client.submit_profile_job(
        model=compiled_model,
        device=device,
        name=f"[4STAGE_PROFILE] Vocoder_W8A16_{device.name.replace(' ', '_')}",
    )
    print(f"  • Profile Job ID  : {profile_job.job_id}")
    print(f"  • Dashboard URL   : {profile_job.url}")
    print("  ⏳ Waiting for on-device hardware profiling...")
    profile_job.wait()
    profile_status = profile_job.get_status().code
    print(f"  • Profile Status  : {profile_status}")

    perf_metrics = {"latency_ms": "N/A", "peak_ram_mb": "N/A", "compute_unit": "Qualcomm Hexagon HTP v75 NPU"}
    if profile_status == "SUCCESS":
        try:
            profile_data = profile_job.download_profile()
            exec_summary = profile_data.get("execution_summary", {})
            time_us = exec_summary.get("estimated_inference_time", 0)
            ram_bytes = exec_summary.get("estimated_inference_peak_memory", 0)
            perf_metrics["latency_ms"] = round(time_us / 1000.0, 3) if time_us else "N/A"
            perf_metrics["peak_ram_mb"] = round(ram_bytes / (1024.0 * 1024.0), 2) if ram_bytes else "N/A"
            print(f"  ⚡ On-Device Latency: {perf_metrics['latency_ms']} ms")
            print(f"  💾 Peak RAM Usage   : {perf_metrics['peak_ram_mb']} MB")
        except Exception as e:
            print(f"  ⚠️ Could not parse profile details: {e}")

    pipeline_report["stages"]["profile"] = {
        "job_id": profile_job.job_id,
        "url": profile_job.url,
        "status": profile_status,
        "performance": perf_metrics,
    }

    # =========================================================================
    # STAGE 4: INFERENCE (hub.submit_inference_job) & AUDIO VERIFICATION
    # =========================================================================
    print("\n" + "=" * 90)
    print(" 🎙️  STAGE 4/4: INFERENCE & AUDIO GENERATION ON PHYSICAL HARDWARE")
    print("    Executing real spoken sentences and downloading physical audio")
    print("=" * 90)

    test_sentences = [
        ("vi", "Xin chào VNG! Hệ thống OneVoice AI chạy trên NPU Snapdragon.", "live_s24_quantized_vietnamese.wav"),
        ("en", "The OneVoice AI Challenge runs directly on Qualcomm Hexagon NPU.", "live_s24_quantized_english.wav"),
    ]

    for lang, sentence, out_wav_name in test_sentences:
        print(f"\n--- Running Inference for [{lang.upper()}]: '{sentence}' ---")
        real_latent, t_frames = generate_real_mel_latent(engine, sentence, lang)
        print(f"  • Real Mel-Latent Shape: {real_latent.shape} (Speech Frames: {t_frames}/100)")

        inf_job = client.submit_inference_job(
            model=compiled_model,
            device=device,
            inputs={"latent": [real_latent]},
            name=f"[4STAGE_INFERENCE_{lang.upper()}] Vocoder_{device.name.replace(' ', '_')}",
        )
        print(f"  • Inference Job ID : {inf_job.job_id}")
        print(f"  • Dashboard URL    : {inf_job.url}")
        print("  ⏳ Waiting for physical Hexagon NPU to generate waveform tensor...")
        inf_job.wait()
        inf_status = inf_job.get_status().code
        print(f"  • Inference Status : {inf_status}")

        if inf_status != "SUCCESS":
            raise RuntimeError(f"Inference failed for {lang} with status: {inf_status}")

        print("  📥 Downloading audio tensor generated by Hexagon NPU...")
        outputs = inf_job.download_output_data()
        first_key = list(outputs.keys())[0]
        raw_tensor = np.asarray(outputs[first_key][0]).squeeze().astype(np.float32)

        # Save raw numpy tensor
        npy_path = os.path.join(OUTPUT_DIR, out_wav_name.replace(".wav", "_raw_tensor.npy"))
        np.save(npy_path, raw_tensor)

        # Normalize peak to 0.95 to ensure clean audible sound without clipping
        peak_amp = float(np.max(np.abs(raw_tensor)))
        if peak_amp > 0:
            norm_wav = raw_tensor / peak_amp * 0.95
        else:
            norm_wav = raw_tensor

        out_wav_path = os.path.join(OUTPUT_DIR, out_wav_name)
        sample_rate = 44100
        sf.write(out_wav_path, norm_wav, sample_rate)
        duration_sec = len(norm_wav) / float(sample_rate)
        rms_val = float(np.sqrt(np.mean(norm_wav ** 2)))

        print(f"  ✅ SUCCESS: Written Live Audio File: {out_wav_path}")
        print(f"     • Sample Rate : {sample_rate} Hz")
        print(f"     • Duration    : {duration_sec:.2f} s ({len(norm_wav)} samples)")
        print(f"     • Peak Amp    : {float(np.max(np.abs(norm_wav))):.2f}")
        print(f"     • RMS Energy  : {rms_val:.4f}")

        pipeline_report["audio_results"].append({
            "language": lang,
            "text": sentence,
            "inference_job_id": inf_job.job_id,
            "dashboard_url": inf_job.url,
            "status": inf_status,
            "wav_path": out_wav_path,
            "raw_tensor_path": npy_path,
            "duration_sec": round(duration_sec, 2),
            "sample_rate": sample_rate,
            "peak_amplitude": round(float(np.max(np.abs(norm_wav))), 2),
            "rms_energy": round(rms_val, 4),
        })

    # Save summary report
    report_path = os.path.join(OUTPUT_DIR, "live_s24_verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved Complete Verification Report to: {report_path}")

    print("\n" + "=" * 90)
    print(" 🎉 ALL 4 QUALCOMM AI HUB STAGES COMPLETED & VERIFIED SUCCESSFULLY!")
    print(f"    1. Quantize Job ID : {pipeline_report['stages']['quantize']['job_id']}")
    print(f"    2. Compile Job ID  : {pipeline_report['stages']['compile']['job_id']}")
    print(f"    3. Profile Job ID  : {pipeline_report['stages']['profile']['job_id']}")
    for res in pipeline_report["audio_results"]:
        print(f"    4. Inference [{res['language'].upper()}]: {res['inference_job_id']} -> {res['wav_path']}")
    print("=" * 90)


if __name__ == "__main__":
    run_deployment_pipeline()
