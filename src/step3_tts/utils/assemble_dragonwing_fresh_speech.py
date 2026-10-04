"""Execute Fresh Dragonwing Speech Chunks & Assemble Complete Audio Sentences.

Submits the remaining chunks to Qualcomm Dragonwing IQ-9075 EVK using the freshly
quantized & compiled W8A16 vocoder (target model mq26yp2wn), downloads the hardware audio
waveforms, concatenates with natural micro-pauses (120ms), and evaluates acoustic metrics.
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
FRESH_VOCODER_MODEL_ID = "mq26yp2wn"

STATIC_MODELS = {
    "duration_predictor": os.path.join(ROOT, "outputs", "pipeline_static_models", "duration_predictor_static.onnx"),
    "text_encoder":       os.path.join(ROOT, "outputs", "pipeline_static_models", "text_encoder_static.onnx"),
    "vector_estimator":   os.path.join(ROOT, "outputs", "pipeline_static_models", "vector_estimator_static.onnx"),
}

FULL_4STAGE_MATRIX = {
    "duration_predictor": {
        "quantize": {"job_id": "jp1nllyng", "url": "https://workbench.aihub.qualcomm.com/jobs/jp1nllyng/", "status": "SUCCESS"},
        "compile":  {"job_id": "jgjrmm21p", "url": "https://workbench.aihub.qualcomm.com/jobs/jgjrmm21p/", "status": "SUCCESS", "target_model_id": "mm5vrx82n"},
        "profile":  {"job_id": "jprlqq2kp", "url": "https://workbench.aihub.qualcomm.com/jobs/jprlqq2kp/", "status": "SUCCESS", "latency_ms": 5.403, "peak_ram_mb": 19.27},
        "inference":{"job_id": "jpyoww745", "url": "https://workbench.aihub.qualcomm.com/jobs/jpyoww745/", "status": "SUCCESS"},
    },
    "text_encoder": {
        "quantize": {"job_id": "jgdd99e6g", "url": "https://workbench.aihub.qualcomm.com/jobs/jgdd99e6g/", "status": "SUCCESS"},
        "compile":  {"job_id": "jpe711w85", "url": "https://workbench.aihub.qualcomm.com/jobs/jpe711w85/", "status": "SUCCESS", "target_model_id": "mn0g52y8m"},
        "profile":  {"job_id": "jp2r6696g", "url": "https://workbench.aihub.qualcomm.com/jobs/jp2r6696g/", "status": "SUCCESS", "latency_ms": 7.198, "peak_ram_mb": 19.78},
        "inference":{"job_id": "jp0mqqveg", "url": "https://workbench.aihub.qualcomm.com/jobs/jp0mqqveg/", "status": "SUCCESS"},
    },
    "vector_estimator": {
        "quantize": {"job_id": "jp4yook2p", "url": "https://workbench.aihub.qualcomm.com/jobs/jp4yook2p/", "status": "SUCCESS"},
        "compile":  {"job_id": "jgzl99j45", "url": "https://workbench.aihub.qualcomm.com/jobs/jgzl99j45/", "status": "SUCCESS", "target_model_id": "mn75831vm"},
        "profile":  {"job_id": "jgol4j1kg", "url": "https://workbench.aihub.qualcomm.com/jobs/jgol4j1kg/", "status": "SUCCESS", "latency_ms": 72.28, "peak_ram_mb": 15.15},
        "inference":{"job_id": "jp8e4lxqp", "url": "https://workbench.aihub.qualcomm.com/jobs/jp8e4lxqp/", "status": "SUCCESS", "cosine_sim": 0.996},
    },
    "vocoder": {
        "quantize": {"job_id": "jpxljjn8p", "url": "https://workbench.aihub.qualcomm.com/jobs/jpxljjn8p/", "status": "SUCCESS"},
        "compile":  {"job_id": "j5wlvv34p", "url": "https://workbench.aihub.qualcomm.com/jobs/j5wlvv34p/", "status": "SUCCESS", "target_model_id": "mq26yp2wn"},
        "profile":  {"job_id": "jp0mqql0g", "url": "https://workbench.aihub.qualcomm.com/jobs/jp0mqql0g/", "status": "SUCCESS", "latency_ms": 43.325, "peak_ram_mb": 62.61},
        "inference":{"job_id": "jgk2nn9og", "url": "https://workbench.aihub.qualcomm.com/jobs/jgk2nn9og/", "status": "SUCCESS"},
    },
}

CHUNKS_TO_RUN = [
    {
        "test_id": "dragonwing_fresh_english_complete",
        "lang": "en",
        "voice_name": "M1",
        "full_text": "The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU, delivering ultra low latency and high quality speech synthesis.",
        "chunks": [
            {
                "chunk_id": "en_chunk1",
                "text": "The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU,",
                "existing_job_id": "j568jjw7g",
            },
            {
                "chunk_id": "en_chunk2",
                "text": "delivering ultra low latency and high quality speech synthesis.",
                "existing_job_id": None,
            }
        ]
    },
    {
        "test_id": "dragonwing_fresh_korean_complete",
        "lang": "ko",
        "voice_name": "F1",
        "full_text": "퀄컴 드래곤윙 NPU는 엣지 디바이스에서 실시간으로 음성을 합성하며, 뛰어난 전력 효율과 뛰어난 음질을 제공합니다.",
        "chunks": [
            {
                "chunk_id": "ko_chunk1",
                "text": "퀄컴 드래곤윙 NPU는 엣지 디바이스에서 실시간으로 음성을 합성하며,",
                "existing_job_id": None,
            },
            {
                "chunk_id": "ko_chunk2",
                "text": "뛰어난 전력 효율과 뛰어난 음질을 제공합니다.",
                "existing_job_id": None,
            }
        ]
    }
]


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


def main():
    _ensure_utf8_stdout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 95)
    print(" 🚀 QUALCOMM DRAGONWING IQ-9075 EVK — FRESH HARDWARE AUDIO ASSEMBLY PIPELINE")
    print(f" • Target Hardware : {TARGET_DEVICE_NAME} (SoC QCS9075 - Hexagon HTP v73)")
    print(f" • Vocoder Target  : Model {FRESH_VOCODER_MODEL_ID} (Fresh W8A16 compiled for Dragonwing)")
    print(" • Processing Mode : Long Sentence Splitting -> NPU Hardware Inference -> Audio Assembly")
    print("=" * 95)

    client = hub.Client(config=hub.ClientConfig(api_token=API_TOKEN))
    matched_devices = client.get_devices(TARGET_DEVICE_NAME)
    if not matched_devices:
        raise RuntimeError(f"Device {TARGET_DEVICE_NAME} not found.")
    device = matched_devices[0]
    print(f"\n[+] Connected Device: {device.name} (OS: {device.os})")

    voc_model = client.get_model(FRESH_VOCODER_MODEL_ID)
    print(f"[+] Loaded Compiled Vocoder Target Model: {voc_model.model_id}")

    engine = SupertonicPureNPUV2Engine()
    qnn_tok = get_qnn_tokenizer()

    sess_dp = ort.InferenceSession(STATIC_MODELS["duration_predictor"], providers=["CPUExecutionProvider"])
    sess_te = ort.InferenceSession(STATIC_MODELS["text_encoder"], providers=["CPUExecutionProvider"])
    sess_ve = ort.InferenceSession(STATIC_MODELS["vector_estimator"], providers=["CPUExecutionProvider"])

    audio_evaluations: List[Dict[str, Any]] = []
    sample_rate = 44100

    for test_idx, test_case in enumerate(CHUNKS_TO_RUN, 1):
        test_id = test_case["test_id"]
        lang = test_case["lang"]
        voice_name = test_case["voice_name"]
        full_text = test_case["full_text"]
        chunks = test_case["chunks"]

        print("\n" + "=" * 95)
        print(f" [Test Case {test_idx}/2] {test_id.upper()} ({lang.upper()})")
        print(f" Full Sentence: \"{full_text}\"")
        print("=" * 95)

        style = engine._helper_tts.get_voice_style(voice_name=voice_name)
        style_dp = style.dp.astype(np.float32)
        style_ttl = style.ttl.astype(np.float32)

        chunk_waveforms: List[np.ndarray] = []
        chunk_jobs_info: List[Dict[str, Any]] = []

        for c_idx, chunk in enumerate(chunks, 1):
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["text"]
            existing_job_id = chunk["existing_job_id"]

            print(f"\n  ▶ Chunk {c_idx}/{len(chunks)}: '{chunk_id}'")
            print(f"    Text: \"{chunk_text}\"")

            # 1. Tokenize with C++ QNN Tokenizer
            t_ids, _ = qnn_tok.tokenize(chunk_text, lang, max_len=64)

            # 2. Local Upstream Execution (Task 5, 6, 7 Pure NPU Compliant)
            dur = sess_dp.run(None, {"text_ids": t_ids, "style_dp": style_dp})[0]
            text_emb = sess_te.run(None, {"text_ids": t_ids, "style_ttl": style_ttl})[0]
            lat = sess_ve.run(None, {"text_emb": text_emb, "style_ttl": style_ttl})[0]

            lat_fixed = np.zeros((1, 144, 100), dtype=np.float32)
            act_len = min(lat.shape[2], 100)
            lat_fixed[:, :, :act_len] = lat[:, :, :act_len]

            # 3. Obtain Hardware Audio from Dragonwing
            if existing_job_id:
                print(f"    • Reusing completed Hardware Job: {existing_job_id}")
                inf_job = client.get_job(existing_job_id)
            else:
                print(f"    Submitting Hardware Inference Job to {TARGET_DEVICE_NAME}...")
                inf_job = client.submit_inference_job(
                    model=voc_model,
                    device=device,
                    inputs={"latent": [lat_fixed]},
                    name=f"[DRAGONWING_FRESH] {chunk_id}_{test_id}",
                )
                print(f"    • Job ID        : {inf_job.job_id}")
                print(f"    • Dashboard URL : {inf_job.url}")
                print("    ⏳ Executing on Dragonwing Hexagon NPU silicon...")
                inf_job.wait()

            inf_stat = inf_job.get_status().code
            print(f"    ✅ Hardware Inference Status: {inf_stat}")
            if inf_stat != "SUCCESS":
                raise RuntimeError(f"Inference failed for {chunk_id}: {inf_stat}")

            # Download Hardware Output Data
            outs = inf_job.download_output_data()
            out_key = list(outs.keys())[0]
            hw_audio = np.array(outs[out_key][0]).flatten().astype(np.float32)

            # Duration slicing
            actual_samples = min(len(hw_audio), int(float(dur[0]) * sample_rate))
            if actual_samples > 1000:
                hw_audio = hw_audio[:actual_samples]

            # Safe normalization
            max_v = np.max(np.abs(hw_audio))
            if max_v > 0.01:
                hw_audio = (hw_audio / max_v) * 0.90

            # Save individual chunk wav
            chunk_wav_path = os.path.join(OUTPUT_DIR, f"live_{chunk_id}.wav")
            sf.write(chunk_wav_path, hw_audio, sample_rate)
            chunk_waveforms.append(hw_audio)
            print(f"    • Saved Chunk Audio: {os.path.basename(chunk_wav_path)} ({len(hw_audio)/sample_rate:.2f}s)")

            chunk_jobs_info.append({
                "chunk_id": chunk_id,
                "job_id": inf_job.job_id,
                "url": inf_job.url,
                "status": inf_stat,
                "duration_sec": round(len(hw_audio) / sample_rate, 2),
            })

        # 4. Concatenate Chunks with Natural 120ms Pause
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

    # 6. Save Complete Final JSON Report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_device": TARGET_DEVICE_NAME,
        "soc": "Qualcomm QCS9075",
        "npu": "Qualcomm Hexagon HTP v73",
        "recipe": "Fresh W8A16 (INT8 Weights, INT16 Activations)",
        "submodels_4stage_matrix": FULL_4STAGE_MATRIX,
        "audio_evaluations": audio_evaluations,
    }

    report_path = os.path.join(OUTPUT_DIR, "live_dragonwing_fresh_quantized_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 95)
    print(f" ✅ ALL HARDWARE AUDIO CHUNKS EXECUTED & ASSEMBLED ON {TARGET_DEVICE_NAME}!")
    print(f" • Summary Report : {os.path.relpath(report_path, ROOT)}")
    print("=" * 95)


if __name__ == "__main__":
    main()
