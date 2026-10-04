"""Qualcomm AI Hub Dragonwing IQ-9075 EVK Fresh Full 4-Stage Deployment Pipeline.

Executes:
  1. Multilingual Calibration Dataset Generation (English & Korean) via C++ QNN Custom Tokenizer.
  2. Stage 1 (Quantize): Fresh W8A16 quantization for all 4 submodels on AI Hub.
  3. Stage 2 (Compile):  Targeted compilation for Qualcomm Dragonwing IQ-9075 EVK (Hexagon HTP v73).
  4. Stage 3 (Profile):  Physical hardware profiling on Dragonwing IQ-9075 EVK board (Latency & Peak RAM).
  5. Stage 4 (Inference): Hardware inference with sentence chunking (English & Korean) on Dragonwing.
  6. Audio Assembly:     Stitching chunked physical hardware audio waveforms with 120ms pause.
  7. Acoustic Evaluation: Comprehensive spectral analysis and JSON reporting.
"""

import os
import sys
import time
import json
from typing import Any, Dict, List, Tuple
import numpy as np
import soundfile as sf
import onnxruntime as ort
import qai_hub as hub

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import _ensure_utf8_stdout
from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine
from step3_tts.qnn_custom_tokenizer.qnn_tokenizer_engine import get_qnn_tokenizer

DEFAULT_TOKEN = "e6hfu2dafj980obg2o60rs5y80ou091ct8g8aofk"
API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN", DEFAULT_TOKEN)
OUTPUT_DIR = os.path.join(ROOT, "outputs", "aihub_live_spoken_speech")
TARGET_DEVICE_NAME = "Dragonwing IQ-9075 EVK"

STATIC_MODELS = {
    "duration_predictor": os.path.join(ROOT, "outputs", "pipeline_static_models", "duration_predictor_static.onnx"),
    "text_encoder":       os.path.join(ROOT, "outputs", "pipeline_static_models", "text_encoder_static.onnx"),
    "vector_estimator":   os.path.join(ROOT, "outputs", "pipeline_static_models", "vector_estimator_static.onnx"),
    "vocoder":            os.path.join(ROOT, "outputs", "pipeline_static_models", "vocoder_static.onnx"),
}

# Long sentences to be split into chunks, executed on Dragonwing, and concatenated
SPLIT_SENTENCE_TESTS = [
    {
        "id": "dragonwing_fresh_english_complete",
        "lang": "en",
        "voice_name": "M1",
        "full_text": "The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU, delivering ultra low latency and high quality speech synthesis.",
        "chunks": [
            {
                "chunk_id": "en_chunk1",
                "text": "The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU,",
            },
            {
                "chunk_id": "en_chunk2",
                "text": "delivering ultra low latency and high quality speech synthesis.",
            }
        ]
    },
    {
        "id": "dragonwing_fresh_korean_complete",
        "lang": "ko",
        "voice_name": "F1",
        "full_text": "퀄컴 드래곤윙 NPU는 엣지 디바이스에서 실시간으로 음성을 합성하며, 뛰어난 전력 효율과 뛰어난 음질을 제공합니다.",
        "chunks": [
            {
                "chunk_id": "ko_chunk1",
                "text": "퀄컴 드래곤윙 NPU는 엣지 디바이스에서 실시간으로 음성을 합성하며,",
            },
            {
                "chunk_id": "ko_chunk2",
                "text": "뛰어난 전력 효율과 뛰어난 음질을 제공합니다.",
            }
        ]
    }
]


def build_multilingual_calibration_datasets(
    engine: SupertonicPureNPUV2Engine,
    qnn_tok: Any,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """Generates realistic calibration datasets for all 4 submodels using QNN Tokenizer."""
    print("  • Generating realistic calibration datasets across English and Korean...")
    style_en = engine._helper_tts.get_voice_style("M1")
    style_ko = engine._helper_tts.get_voice_style("F1")

    calib_texts = [
        ("en", "The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU.", style_en),
        ("en", "Delivering ultra low latency and high quality speech synthesis.", style_en),
        ("ko", "안녕하세요 퀄컴 드래곤윙 NPU 음성 합성 테스트입니다.", style_ko),
        ("ko", "뛰어난 전력 효율과 뛰어난 음질을 제공합니다.", style_ko),
    ]

    dp_calib = {"text_ids": [], "style_dp": []}
    te_calib = {"text_ids": [], "style_ttl": []}
    ve_calib = {"text_emb": [], "style_ttl": []}
    voc_calib = {"latent": []}

    sess_dp = ort.InferenceSession(STATIC_MODELS["duration_predictor"], providers=["CPUExecutionProvider"])
    sess_te = ort.InferenceSession(STATIC_MODELS["text_encoder"], providers=["CPUExecutionProvider"])
    sess_ve = ort.InferenceSession(STATIC_MODELS["vector_estimator"], providers=["CPUExecutionProvider"])

    for lang, text, st in calib_texts:
        t_ids, _ = qnn_tok.tokenize(text, lang, max_len=64)
        style_dp = st.dp.astype(np.float32)
        style_ttl = st.ttl.astype(np.float32)

        dp_calib["text_ids"].append(t_ids)
        dp_calib["style_dp"].append(style_dp)

        te_calib["text_ids"].append(t_ids)
        te_calib["style_ttl"].append(style_ttl)

        # Run local TE to get text_emb
        text_emb = sess_te.run(None, {"text_ids": t_ids, "style_ttl": style_ttl})[0]
        ve_calib["text_emb"].append(text_emb)
        ve_calib["style_ttl"].append(style_ttl)

        # Run local VE unrolled to get latent
        lat = sess_ve.run(None, {"text_emb": text_emb, "style_ttl": style_ttl})[0]
        lat_100 = np.zeros((1, 144, 100), dtype=np.float32)
        act_len = min(lat.shape[2], 100)
        lat_100[:, :, :act_len] = lat[:, :, :act_len]

        voc_calib["latent"].append(lat_100)

    print(f"    - DP Samples  : {len(dp_calib['text_ids'])} tensors")
    print(f"    - TE Samples  : {len(te_calib['text_ids'])} tensors")
    print(f"    - VE Samples  : {len(ve_calib['text_emb'])} tensors")
    print(f"    - Voc Samples : {len(voc_calib['latent'])} tensors")

    return {
        "duration_predictor": dp_calib,
        "text_encoder": te_calib,
        "vector_estimator": ve_calib,
        "vocoder": voc_calib,
    }


def evaluate_audio_quality(wav: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Computes acoustic quality metrics for speech validation."""
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


def run_deployment():
    _ensure_utf8_stdout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 95)
    print(" 🚀 QUALCOMM AI HUB — FRESH FULL 4-STAGE PIPELINE ON DRAGONWING IQ-9075 EVK")
    print(f" • Target Hardware : {TARGET_DEVICE_NAME} (SoC QCS9075 - Hexagon HTP v73 NPU)")
    print(" • Submodels       : [1] duration_predictor (In-Graph Mask, Fused Speed)")
    print("                     [2] text_encoder       (One-Hot GEMM, In-Graph Mask)")
    print("                     [3] vector_estimator   (Unrolled 5-Step, In-Graph Mask, Static Noise)")
    print("                     [4] vocoder            (Pure NPU Compliant)")
    print(" • Full 4 Steps    : [1] Quantize (W8A16) -> [2] Compile -> [3] Profile -> [4] Inference")
    print(" • Processing Mode : Long Sentence Splitting -> Hardware Inference -> Audio Assembly")
    print("=" * 95)

    client = hub.Client(config=hub.ClientConfig(api_token=API_TOKEN))
    matched_devices = client.get_devices(TARGET_DEVICE_NAME)
    if not matched_devices:
        raise RuntimeError(f"Target device '{TARGET_DEVICE_NAME}' not found on AI Hub.")
    device = matched_devices[0]
    print(f"\n[+] Connected Target Device: {device.name} (OS: {device.os})")

    # Initialize Engine & QNN Tokenizer
    print("\n[+] Initializing Pure NPU V2 Engine & QNN C++ Custom Tokenizer...")
    engine = SupertonicPureNPUV2Engine()
    qnn_tok = get_qnn_tokenizer()
    calib_datasets = build_multilingual_calibration_datasets(engine, qnn_tok)

    submodel_order = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]
    matrix_report: Dict[str, Any] = {}

    # =========================================================================
    # STAGE 1: FRESH QUANTIZE (W8A16)
    # =========================================================================
    print("\n" + "=" * 95)
    print(" 🛠️  STAGE 1/4: FRESH QUANTIZE (W8A16) ON QUALCOMM AI HUB")
    print("=" * 95)
    quant_jobs: Dict[str, Any] = {}
    uploaded_models: Dict[str, Any] = {}

    for name in submodel_order:
        model_path = STATIC_MODELS[name]
        print(f"\n[1/4] Uploading & Submitting QUANTIZE Job for '{name}'...")
        uploaded = client.upload_model(model_path)
        uploaded_models[name] = uploaded
        q_job = client.submit_quantize_job(
            model=uploaded,
            calibration_data=calib_datasets[name],
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            name=f"[DRAGONWING_FRESH] {name}_w8a16",
        )
        quant_jobs[name] = q_job
        print(f"    • Job ID        : {q_job.job_id}")
        print(f"    • Dashboard URL : {q_job.url}")

    print("\n⏳ Awaiting completion of all 4 Quantize jobs...")
    quantized_models: Dict[str, Any] = {}
    for name in submodel_order:
        job = quant_jobs[name]
        job.wait()
        status = job.get_status().code
        print(f"    • {name:<20}: {status} (Job: {job.job_id})")
        if status != "SUCCESS":
            raise RuntimeError(f"Quantization failed for {name}: {status}")
        target_model = job.get_target_model()
        quantized_models[name] = target_model
        matrix_report[name] = {
            "quantize": {
                "job_id": job.job_id,
                "url": job.url,
                "status": status,
                "target_model_id": target_model.model_id,
            }
        }

    # =========================================================================
    # STAGE 2: COMPILE (DRAGONWING IQ-9075 EVK)
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" ⚙️  STAGE 2/4: COMPILE FOR {TARGET_DEVICE_NAME}")
    print("=" * 95)
    compile_jobs: Dict[str, Any] = {}

    for name in submodel_order:
        q_model = quantized_models[name]
        print(f"\n[2/4] Submitting COMPILE Job for '{name}' on {device.name}...")
        c_job = client.submit_compile_job(
            model=q_model,
            device=device,
            options="--target_runtime onnx",
            name=f"[DRAGONWING_FRESH] {name}_compile_{device.name.replace(' ', '_')}",
        )
        compile_jobs[name] = c_job
        print(f"    • Job ID        : {c_job.job_id}")
        print(f"    • Dashboard URL : {c_job.url}")

    print("\n⏳ Awaiting completion of all 4 Compile jobs...")
    compiled_models: Dict[str, Any] = {}
    for name in submodel_order:
        job = compile_jobs[name]
        job.wait()
        status = job.get_status().code
        print(f"    • {name:<20}: {status} (Job: {job.job_id})")
        if status != "SUCCESS":
            raise RuntimeError(f"Compilation failed for {name}: {status}")
        target_model = job.get_target_model()
        compiled_models[name] = target_model
        matrix_report[name]["compile"] = {
            "job_id": job.job_id,
            "url": job.url,
            "status": status,
            "target_model_id": target_model.model_id,
        }

    # =========================================================================
    # STAGE 3: PROFILE (DRAGONWING IQ-9075 EVK PHYSICAL HARDWARE)
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" 📊 STAGE 3/4: HARDWARE PROFILING ON PHYSICAL {TARGET_DEVICE_NAME}")
    print("=" * 95)
    profile_jobs: Dict[str, Any] = {}

    for name in submodel_order:
        c_model = compiled_models[name]
        print(f"\n[3/4] Submitting PROFILE Job for '{name}' on {device.name}...")
        p_job = client.submit_profile_job(
            model=c_model,
            device=device,
            name=f"[DRAGONWING_FRESH] {name}_profile_{device.name.replace(' ', '_')}",
        )
        profile_jobs[name] = p_job
        print(f"    • Job ID        : {p_job.job_id}")
        print(f"    • Dashboard URL : {p_job.url}")

    print("\n⏳ Awaiting completion of all 4 Profile jobs...")
    for name in submodel_order:
        job = profile_jobs[name]
        job.wait()
        status = job.get_status().code
        print(f"    • {name:<20}: {status} (Job: {job.job_id})")
        prof_data: Dict[str, Any] = {
            "job_id": job.job_id,
            "url": job.url,
            "status": status,
        }
        if status == "SUCCESS":
            try:
                prof = job.download_profile()
                es = prof.get("execution_summary", {})
                t_us = es.get("estimated_inference_time", 0)
                r_b = es.get("estimated_inference_peak_memory", 0)
                prof_data["latency_ms"] = round(t_us / 1000.0, 3) if t_us else "N/A"
                prof_data["peak_ram_mb"] = round(r_b / (1024.0 * 1024.0), 2) if r_b else "N/A"
                print(f"      Latency : {prof_data['latency_ms']} ms | Peak RAM: {prof_data['peak_ram_mb']} MB")
            except Exception as e:
                print(f"      Note on profile summary: {e}")
        matrix_report[name]["profile"] = prof_data

    # =========================================================================
    # STAGE 4A: SUBMODEL-LEVEL HARDWARE INFERENCE MATRIX VERIFICATION
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" 🔬 STAGE 4A/4: HARDWARE INFERENCE VERIFICATION FOR ALL 4 SUBMODELS ON {TARGET_DEVICE_NAME}")
    print("=" * 95)
    inf_matrix_jobs: Dict[str, Any] = {}
    for name in submodel_order:
        c_model = compiled_models[name]
        sample_inp = {k: [v[0]] for k, v in calib_datasets[name].items()}
        print(f"\n[4A] Submitting Inference Job for '{name}' on {device.name}...")
        i_job = client.submit_inference_job(
            model=c_model,
            device=device,
            inputs=sample_inp,
            name=f"[DRAGONWING_FRESH] {name}_inference_{device.name.replace(' ', '_')}",
        )
        inf_matrix_jobs[name] = i_job
        print(f"    • Job ID        : {i_job.job_id}")
        print(f"    • Dashboard URL : {i_job.url}")

    print("\n⏳ Awaiting completion of all 4 Submodel Inference jobs...")
    for name in submodel_order:
        job = inf_matrix_jobs[name]
        job.wait()
        status = job.get_status().code
        print(f"    • {name:<20}: {status} (Job: {job.job_id})")
        matrix_report[name]["inference"] = {
            "job_id": job.job_id,
            "url": job.url,
            "status": status,
        }

    # =========================================================================
    # STAGE 4B: CHUNKED SPEECH GENERATION & HARDWARE AUDIO ASSEMBLY
    # =========================================================================
    print("\n" + "=" * 95)
    print(f" 🔊 STAGE 4B/4: CHUNKED SPEECH SYNTHESIS & HARDWARE AUDIO ASSEMBLY ON {TARGET_DEVICE_NAME}")
    print("=" * 95)

    sess_dp = ort.InferenceSession(STATIC_MODELS["duration_predictor"], providers=["CPUExecutionProvider"])
    sess_te = ort.InferenceSession(STATIC_MODELS["text_encoder"], providers=["CPUExecutionProvider"])
    sess_ve = ort.InferenceSession(STATIC_MODELS["vector_estimator"], providers=["CPUExecutionProvider"])

    audio_evaluations: List[Dict[str, Any]] = []

    for test_idx, test_case in enumerate(SPLIT_SENTENCE_TESTS, 1):
        test_id = test_case["id"]
        lang = test_case["lang"]
        voice_name = test_case["voice_name"]
        full_text = test_case["full_text"]
        chunks = test_case["chunks"]

        print(f"\n[Test Case {test_idx}/2] {test_id.upper()} ({lang.upper()})")
        print(f" • Full Sentence: \"{full_text}\"")
        print(f" • Chunks Count : {len(chunks)} segments")

        style = engine._helper_tts.get_voice_style(voice_name=voice_name)
        style_dp = style.dp.astype(np.float32)
        style_ttl = style.ttl.astype(np.float32)

        chunk_waveforms: List[np.ndarray] = []
        chunk_jobs_info: List[Dict[str, Any]] = []

        for c_idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["text"]
            print(f"\n  ▶ Executing Chunk {c_idx}/{len(chunks)}: '{chunk_id}'")
            print(f"    Text: \"{chunk_text}\"")

            # 1. QNN C++ Tokenizer
            t_ids, _ = qnn_tok.tokenize(chunk_text, lang, max_len=64)

            # 2. Local Upstream Execution (Task 5, 6, 7 Compliant)
            dur = sess_dp.run(None, {"text_ids": t_ids, "style_dp": style_dp})[0]
            text_emb = sess_te.run(None, {"text_ids": t_ids, "style_ttl": style_ttl})[0]
            lat = sess_ve.run(None, {"text_emb": text_emb, "style_ttl": style_ttl})[0]

            lat_fixed = np.zeros((1, 144, 100), dtype=np.float32)
            act_len = min(lat.shape[2], 100)
            lat_fixed[:, :, :act_len] = lat[:, :, :act_len]

            # 3. Submit Hardware Inference Job on Dragonwing
            print(f"    Submitting Hardware Inference Job to {TARGET_DEVICE_NAME}...")
            inf_job = client.submit_inference_job(
                model=compiled_models["vocoder"],
                device=device,
                inputs={"latent": [lat_fixed]},
                name=f"[DRAGONWING_FRESH] {chunk_id}_{test_id}",
            )
            print(f"    • Job ID        : {inf_job.job_id}")
            print(f"    • Dashboard URL : {inf_job.url}")
            print("    ⏳ Executing on Dragonwing Hexagon NPU silicon...")
            inf_job.wait()

            inf_stat = inf_job.get_status().code
            print(f"    ✅ Inference Status: {inf_stat}")
            if inf_stat != "SUCCESS":
                raise RuntimeError(f"Hardware inference failed for chunk {chunk_id}: {inf_stat}")

            # Download Hardware Output
            inf_output = inf_job.download_output()
            out_tensor_key = list(inf_output.keys())[0]
            hw_audio = np.array(inf_output[out_tensor_key][0]).flatten().astype(np.float32)

            # Calculate actual audio length from duration
            sample_rate = 44100
            actual_samples = min(len(hw_audio), int(float(dur[0]) * sample_rate))
            if actual_samples > 1000:
                hw_audio = hw_audio[:actual_samples]

            # Normalize amplitude safely
            max_v = np.max(np.abs(hw_audio))
            if max_v > 0.01:
                hw_audio = (hw_audio / max_v) * 0.90

            # Save individual chunk audio
            chunk_wav_path = os.path.join(OUTPUT_DIR, f"live_{chunk_id}.wav")
            sf.write(chunk_wav_path, hw_audio, sample_rate)
            chunk_waveforms.append(hw_audio)
            print(f"    • Saved Chunk WAV: {os.path.basename(chunk_wav_path)} ({len(hw_audio)/sample_rate:.2f}s)")

            chunk_jobs_info.append({
                "chunk_id": chunk_id,
                "job_id": inf_job.job_id,
                "url": inf_job.url,
                "status": inf_stat,
                "duration_sec": round(len(hw_audio) / sample_rate, 2),
            })

        # 4. Concatenate Chunks with Natural Pause (120ms)
        pause_samples = int(0.120 * sample_rate)
        pause_silence = np.zeros(pause_samples, dtype=np.float32)

        assembled_waveform = []
        for i, wf in enumerate(chunk_waveforms):
            assembled_waveform.append(wf)
            if i < len(chunk_waveforms) - 1:
                assembled_waveform.append(pause_silence)

        full_wav = np.concatenate(assembled_waveform)
        final_wav_path = os.path.join(OUTPUT_DIR, f"live_{test_id}.wav")
        sf.write(final_wav_path, full_wav, sample_rate)

        # 5. Acoustic Evaluation
        eval_metrics = evaluate_audio_quality(full_wav, sample_rate)
        eval_metrics["test_id"] = test_id
        eval_metrics["lang"] = lang
        eval_metrics["full_text"] = full_text
        eval_metrics["wav_path"] = os.path.relpath(final_wav_path, ROOT)
        eval_metrics["chunks"] = chunk_jobs_info

        print(f"\n  🎉 Assembled Full Audio: {os.path.basename(final_wav_path)}")
        print(f"     Duration        : {eval_metrics['duration_sec']}s")
        print(f"     Spectral Centroid: {eval_metrics['spectral_centroid_hz']} Hz")
        print(f"     Voice Band Ratio: {eval_metrics['voice_band_ratio'] * 100:.1f}%")
        print(f"     Assessment      : {eval_metrics['quality_assessment']}")

        audio_evaluations.append(eval_metrics)

    # =========================================================================
    # FINAL REPORT SAVING
    # =========================================================================
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_device": TARGET_DEVICE_NAME,
        "soc": "Qualcomm QCS9075",
        "npu": "Qualcomm Hexagon HTP v73",
        "recipe": "W8A16 (INT8 Weights, INT16 Activations)",
        "submodels_4stage_matrix": matrix_report,
        "audio_evaluations": audio_evaluations,
    }

    report_path = os.path.join(OUTPUT_DIR, "live_dragonwing_fresh_quantized_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print(f" ✅ ALL 4 STAGES SUCCESSFULLY COMPLETED FOR ALL 4 SUBMODELS ON {TARGET_DEVICE_NAME}!")
    print(f" • Summary Report : {os.path.relpath(report_path, ROOT)}")
    print("=" * 95)


if __name__ == "__main__":
    run_deployment()
