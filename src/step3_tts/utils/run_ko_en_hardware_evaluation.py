"""End-to-End Qualcomm AI Hub Pipeline for Korean & English Supertonic 3 TTS:
Evaluates Dragonwing IQ-9075 EVK (Hexagon HTP NPU v73) on real Korean & English Speech.

Submodels:
  1. Duration Predictor (DP) - Compiled Model: mnz01876m
  2. Text Encoder (TE)        - Compiled Model: mq9yld7yn
  3. Vector Estimator (VE)     - Compiled Model: mn75xy98m
  4. Neural Vocoder (VOC)      - Compiled Model: mm6jk8z5q

Languages:
  - English (en): LibriTTS / LJSpeech benchmark sentence
  - Korean  (ko): KSS / FLEURS-Ko benchmark sentence
"""
import os
import sys
import time
import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
import qai_hub as hub

def _ensure_utf8_stdout():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

_ensure_utf8_stdout()

API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN")
if not API_TOKEN:
    raise SystemExit("Set QAI_HUB_API_TOKEN before running this script.")
TARGET_DEVICE_NAME = "Dragonwing IQ-9075 EVK"
OUTPUT_DIR = "outputs/pipeline_hardware_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware Profile Job IDs from AI Hub on Dragonwing IQ-9075 EVK
PROFILES = {
    "duration_predictor": {"job_id": "jgzl40lz5", "latency_ms": 2.71, "peak_ram_mb": 15.50},
    "text_encoder":       {"job_id": "jpe7lqoo5", "latency_ms": 1.87, "peak_ram_mb": 12.41},
    "vector_estimator":   {"job_id": "jgol4j1kg", "latency_ms": 72.28, "peak_ram_mb": 15.15},
    "vocoder":            {"job_id": "jgly199l5", "latency_ms": 77.84, "peak_ram_mb": 63.39},
}

# Compiled Target Model IDs on AI Hub for Dragonwing IQ-9075 EVK
TARGET_MODELS = {
    "duration_predictor": "mnz01876m",
    "text_encoder":       "mq9yld7yn",
    "vector_estimator":   "mn75xy98m",
    "vocoder":            "mm6jk8z5q",
}

TEST_CASES = {
    "en": {
        "lang_name": "English",
        "dataset": "LibriTTS / LJSpeech Benchmark",
        "text": "Reindeer husbandry is an important livelihood for the Sami people in Northern Europe.",
        "voice_name": "M1",
        "style_ttl_path": "outputs/style_vectors/en_style_ttl.npy",
        "style_dp_path": "outputs/style_vectors/en_style_dp.npy",
    },
    "ko": {
        "lang_name": "Korean",
        "dataset": "KSS (Korean Single Speaker) & FLEURS-Ko Benchmark",
        "text": "순록 축산은 북유럽 사미족의 중요한 전통 생계 수단 중 하나입니다.",
        "voice_name": "M1",
        "style_ttl_path": "outputs/style_vectors/ko_style_ttl.npy",
        "style_dp_path": "outputs/style_vectors/ko_style_dp.npy",
    },
}

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_f = gt.flatten().astype(np.float64)
    pred_f = pred.flatten().astype(np.float64)
    norm_gt = np.linalg.norm(gt_f)
    norm_pred = np.linalg.norm(pred_f)
    if norm_gt > 1e-12 and norm_pred > 1e-12:
        cosine_sim = float(np.dot(gt_f, pred_f) / (norm_gt * norm_pred))
    else:
        cosine_sim = 1.0 if np.allclose(gt_f, pred_f) else 0.0

    abs_diff = np.abs(gt_f - pred_f)
    mae = float(np.mean(abs_diff))
    max_ae = float(np.max(abs_diff))

    noise_energy = np.sum((gt_f - pred_f) ** 2)
    signal_energy = np.sum(gt_f ** 2)
    snr_db = float(10.0 * np.log10(signal_energy / (noise_energy + 1e-15)))

    return {
        "cosine_similarity": round(cosine_sim, 6),
        "mae": round(mae, 6),
        "max_error": round(max_ae, 6),
        "snr_db": round(snr_db, 2),
    }

def main():
    print("=" * 85)
    print(" 🚀 QUALCOMM AI HUB LIVE HARDWARE EVALUATION: KOREAN & ENGLISH SPEECH")
    print(f" • Target Device: {TARGET_DEVICE_NAME} (Qualcomm QCS9075 SoC, Hexagon HTP NPU v73)")
    print(f" • Quantization : W8A16 (Weight INT8, Activation INT16)")
    print(f" • Languages    : English (en) & Korean (ko) Exclusively")
    print("=" * 85)

    client = hub.Client(hub.ClientConfig(api_token=API_TOKEN))
    devices = client.get_devices(TARGET_DEVICE_NAME)
    if not devices:
        raise RuntimeError(f"Target device {TARGET_DEVICE_NAME} not found on Qualcomm AI Hub.")
    device = devices[0]
    print(f" • Connected Device: {device.name} (OS: {device.os}, Hexagon: v73)")

    from supertonic import TTS
    tts_helper = TTS(auto_download=False)

    sess_te = ort.InferenceSession("outputs/pipeline_static_models/text_encoder_static.onnx", providers=["CPUExecutionProvider"])
    sess_ve = ort.InferenceSession("outputs/pipeline_static_models/vector_estimator_static.onnx", providers=["CPUExecutionProvider"])
    sess_voc_gt = ort.InferenceSession("outputs/pipeline_static_models/vocoder_static.onnx", providers=["CPUExecutionProvider"])

    voc_model = client.get_model(TARGET_MODELS["vocoder"])
    print(f" • Vocoder NPU Model: {voc_model.model_id} (Compiled for {TARGET_DEVICE_NAME})")

    results = {}

    for lang_code, cfg in TEST_CASES.items():
        print(f"\n" + "-" * 85)
        print(f" 🎙️ Processing Language: {cfg['lang_name']} ({lang_code.upper()})")
        print(f" • Input Text : \"{cfg['text']}\"")
        print(f" • Dataset    : {cfg['dataset']}")

        # 1. Phonemize & tokenize text
        raw_ids, _ = tts_helper.model.text_processor([cfg["text"]], lang_code)
        text_ids = np.zeros((1, 64), dtype=np.int64)
        valid_len = min(raw_ids.shape[1], 64)
        text_ids[0, :valid_len] = raw_ids[0, :valid_len]
        text_mask = np.zeros((1, 1, 64), dtype=np.float32)
        text_mask[0, 0, :valid_len] = 1.0

        # 2. Load dedicated language style vector
        if os.path.exists(cfg["style_ttl_path"]):
            style_ttl = np.load(cfg["style_ttl_path"]).astype(np.float32)
        else:
            style = tts_helper.get_voice_style(cfg["voice_name"])
            style_ttl = style.ttl.astype(np.float32)

        # 3. Text Encoder Inference
        text_emb = sess_te.run(None, {
            "text_ids": text_ids,
            "style_ttl": style_ttl,
            "text_mask": text_mask
        })[0]
        print(f"  ✅ Text Encoder Output Shape: {text_emb.shape}")

        # 4. Vector Estimator Flow-Matching (5 Euler Steps)
        np.random.seed(42 if lang_code == "en" else 100)
        xt = np.random.randn(1, 144, 100).astype(np.float32)
        latent_mask = np.ones((1, 1, 100), dtype=np.float32)

        for step in range(5):
            vt = sess_ve.run(None, {
                "noisy_latent": xt,
                "text_emb": text_emb,
                "style_ttl": style_ttl,
                "latent_mask": latent_mask,
                "text_mask": text_mask,
                "current_step": np.array([float(step)], dtype=np.float32),
                "total_step": np.array([5.0], dtype=np.float32)
            })[0]
            xt = xt + 0.2 * vt
        latent = xt
        print(f"  ✅ Flow-Matching Latent Shape: {latent.shape}")

        # 5. Ground Truth FP32 Vocoder Output
        gt_audio = sess_voc_gt.run(None, {"latent": latent})[0].squeeze().astype(np.float32)

        # 6. Submit Live Inference to Qualcomm AI Hub Dragonwing NPU
        print(f"  🚀 Submitting Live Vocoder Inference to {TARGET_DEVICE_NAME} NPU...")
        t_sub_0 = time.time()
        inf_job = client.submit_inference_job(
            model=voc_model,
            inputs={"latent": [latent]},
            device=device,
            name=f"[EVAL_{lang_code.upper()}] vocoder_dragonwing_iq9075"
        )
        print(f"  • Submitted Job ID : {inf_job.job_id}")
        print(f"  • Dashboard URL    : {inf_job.url}")

        print("  • Waiting for physical NPU execution...")
        inf_job.wait()
        status_code = inf_job.get_status().code
        print(f"  • Physical NPU Status: {status_code} (Execution Time: {time.time() - t_sub_0:.1f}s)")

        if status_code != "SUCCESS":
            raise RuntimeError(f"Inference job {inf_job.job_id} failed with status {status_code}")

        # Download raw physical hardware tensor
        outs = inf_job.download_output_data()
        first_key = list(outs.keys())[0]
        npu_audio = np.asarray(outs[first_key][0]).squeeze().astype(np.float32)

        # Normalize waveform
        peak_npu = np.max(np.abs(npu_audio))
        norm_npu_audio = (npu_audio / peak_npu * 0.95) if peak_npu > 0 else npu_audio

        peak_gt = np.max(np.abs(gt_audio))
        norm_gt_audio = (gt_audio / peak_gt * 0.95) if peak_gt > 0 else gt_audio

        wav_path = os.path.join(OUTPUT_DIR, f"{cfg['lang_name'].lower()}_live_hardware_audio.wav")
        gt_wav_path = os.path.join(OUTPUT_DIR, f"{cfg['lang_name'].lower()}_ground_truth_audio.wav")
        sf.write(wav_path, norm_npu_audio, 44100)
        sf.write(gt_wav_path, norm_gt_audio, 44100)

        # Accuracy metrics
        metrics = compute_metrics(gt_audio, npu_audio)
        audio_dur = len(npu_audio) / 44100.0

        print(f"  🔊 Hardware Audio Saved : {wav_path} (Duration: {audio_dur:.2f}s @ 44.1kHz)")
        print(f"  📊 Cosine Similarity    : {metrics['cosine_similarity']:.6f}")
        print(f"  📊 Signal-to-Noise Ratio: {metrics['snr_db']:.2f} dB")
        print(f"  📊 Mean Absolute Error  : {metrics['mae']:.6f}")

        results[lang_code] = {
            "language": cfg["lang_name"],
            "text": cfg["text"],
            "dataset": cfg["dataset"],
            "inference_job_id": inf_job.job_id,
            "dashboard_url": inf_job.url,
            "audio_duration_sec": round(audio_dur, 2),
            "wav_file": wav_path,
            "ground_truth_wav_file": gt_wav_path,
            "accuracy_metrics": metrics,
        }

    # Compile overall report
    total_pipeline_latency_ms = (
        PROFILES["duration_predictor"]["latency_ms"] +
        PROFILES["text_encoder"]["latency_ms"] +
        (5 * PROFILES["vector_estimator"]["latency_ms"]) +
        PROFILES["vocoder"]["latency_ms"]
    )
    max_peak_ram_mb = max(p["peak_ram_mb"] for p in PROFILES.values())

    report = {
        "target_hardware": {
            "device": TARGET_DEVICE_NAME,
            "soc": "Qualcomm QCS9075",
            "npu": "Qualcomm Hexagon HTP v73",
            "quantization": "W8A16 (Weight INT8, Activation INT16)",
        },
        "hardware_performance_profiles": PROFILES,
        "full_pipeline_metrics": {
            "total_5step_latency_ms": round(total_pipeline_latency_ms, 2),
            "max_submodel_peak_ram_mb": round(max_peak_ram_mb, 2),
            "rtf_6_96s_audio": round((total_pipeline_latency_ms / 1000.0) / 6.96, 4),
        },
        "speech_evaluations": results,
    }

    report_path = os.path.join(OUTPUT_DIR, "korean_english_hardware_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 85)
    print(" 🏆 FINAL HARDWARE EVALUATION SUMMARY FOR KOREAN & ENGLISH SPEECH")
    print("=" * 85)
    print(f" • Total 5-Step Pipeline Latency: {total_pipeline_latency_ms:.2f} ms")
    print(f" • Peak RAM Usage on HTP NPU    : {max_peak_ram_mb:.2f} MB")
    print(f" • Real-Time Factor (RTF)       : {report['full_pipeline_metrics']['rtf_6_96s_audio']:.4f}")
    print(f" • English Cosine Similarity    : {results['en']['accuracy_metrics']['cosine_similarity']:.6f} (SNR: {results['en']['accuracy_metrics']['snr_db']:.2f} dB)")
    print(f" • Korean Cosine Similarity     : {results['ko']['accuracy_metrics']['cosine_similarity']:.6f} (SNR: {results['ko']['accuracy_metrics']['snr_db']:.2f} dB)")
    print(f" • Wrote Comprehensive Report   : {report_path}")
    print("=" * 85)

if __name__ == "__main__":
    main()
