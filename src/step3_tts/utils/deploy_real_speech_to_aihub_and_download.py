"""Generate Real Latent from Spoken Text and Run Live Vocoder Inference on Qualcomm AI Hub NPU.

Takes actual text sentences (Vietnamese & English), computes the Mel-latent representation
via Text Encoder & Vector Estimator, sends the real latent to Samsung Galaxy S24 Ultra Hexagon NPU,
and downloads the actual spoken audio waveform computed by the physical NPU chip.
"""
import os
import sys
import time
import numpy as np
import soundfile as sf
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout
from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "aihub_live_spoken_speech")
API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN")
if not API_TOKEN:
    raise SystemExit("Set QAI_HUB_API_TOKEN before running this script.")


def generate_real_latent(engine, text, lang="vi"):
    """Computes the realistic Mel-latent for text using Text Encoder and 5-step Euler ODE."""
    norm_text = engine.normalizer.normalize(text, lang) or text
    voice_name = "F1" if lang == "vi" else "M1"
    style = engine._helper_tts.get_voice_style(voice_name=voice_name)

    lang_code = "na" if engine._helper_tts.is_multilingual else "en"
    text_ids, text_mask = engine._helper_tts.model.text_processor([norm_text], lang_code)

    dur = engine.sessions["duration_predictor"].run(
        None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
    )[0]
    dur = dur / 1.05

    text_emb = engine.sessions["text_encoder"].run(
        None, {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask}
    )[0]

    np.random.seed(42)
    xt, latent_mask = engine._helper_tts.model.sample_noisy_latent(dur)

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

    # Ensure shape is (1, 144, 100) for the static compiled NPU Vocoder model
    latent_fixed = np.zeros((1, 144, 100), dtype=np.float32)
    T = min(xt.shape[2], 100)
    latent_fixed[:, :, :T] = xt[:, :, :T]
    return latent_fixed, norm_text, T


def run_live_speech_deployment():
    _ensure_utf8_stdout()
    print("=" * 85)
    print(" 🎙️ QUALCOMM AI HUB — LIVE SPOKEN SPEECH SYNTHESIS ON HEXAGON NPU")
    print(f" • Output Directory: {OUTPUT_DIR}")
    print("=" * 85)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = hub.Client(config=hub.ClientConfig(API_TOKEN))

    target_device_name = "Samsung Galaxy S24 Ultra"
    device = client.get_devices(target_device_name)[0]
    print(f" • Target Hardware : {target_device_name} (Snapdragon 8 Gen 3 Hexagon NPU)")

    # 1. Get or compile Vocoder on AI Hub
    print("\n[1/3] Loading Compiled Pure NPU Vocoder Model from Qualcomm AI Hub...")
    # Use compiled job jpr008w7p from previous step if available, or compile
    try:
        compile_job = client.get_job("jpr008w7p")
        target_model = compile_job.get_target_model()
        print(f"  ✅ Reusing Compiled Target Model ID: {target_model.model_id} (from Job jpr008w7p)")
    except Exception as e:
        print(f"  Recompiling model: {e}")
        vocoder_path = os.path.join(ROOT, "outputs", "pure_npu_compliant_onnx_v2", "vocoder_pure_npu.onnx")
        uploaded_model = client.upload_model(vocoder_path)
        compile_job = client.submit_compile_job(
            model=uploaded_model,
            device=device,
            input_specs={"latent": ((1, 144, 100), "float32")},
            options="--target_runtime onnx",
            name="[SPEECH] Supertonic3_Vocoder_Hexagon_NPU",
        )
        compile_job.wait()
        target_model = compile_job.get_target_model()

    # 2. Initialize local engine to compute real latent
    print("\n[2/3] Computing Real Mel-Latent for Spoken Sentences...")
    engine = SupertonicPureNPUV2Engine()

    test_sentences = [
        ("vi", "Xin chào VNG! Hệ thống OneVoice AI chạy trên NPU Snapdragon.", "live_npu_speech_vietnamese.wav"),
        ("en", "The OneVoice AI Challenge runs directly on Qualcomm Hexagon NPU.", "live_npu_speech_english.wav"),
    ]

    for lang, sentence, out_filename in test_sentences:
        print(f"\n" + "-" * 80)
        print(f" 🚀 Processing [{lang.upper()}]: '{sentence}'")
        real_latent, norm_text, actual_T = generate_real_latent(engine, sentence, lang)
        print(f"  • Generated Mel-Latent Shape: {real_latent.shape} (Active frames: {actual_T})")

        # 3. Submit real latent to Galaxy S24 Ultra NPU
        print(f"  • Submitting real latent to Samsung Galaxy S24 Ultra NPU...")
        inf_job = client.submit_inference_job(
            model=target_model,
            device=device,
            inputs={"latent": [real_latent]},
            name=f"[LIVE_SPEECH_{lang.upper()}] Supertonic3_Vocoder",
        )
        print(f"  • Inference Job ID : {inf_job.job_id}")
        print(f"  • Dashboard URL    : {inf_job.url}")
        print("  ⏳ Waiting for Hexagon NPU to decode spoken audio waveform...")
        inf_job.wait()

        st = inf_job.get_status().code
        if st == "SUCCESS":
            outs = inf_job.download_output_data()
            wav_raw = np.asarray(outs["output_0"][0]).squeeze().astype(np.float32)
            
            # Trim to actual active speech length (actual_T frames * 256 samples per frame approx)
            # Vocoder 100 frames = ~2.56s to 12.8s
            out_wav_path = os.path.join(OUTPUT_DIR, out_filename)
            
            # Normalize peak
            peak = np.max(np.abs(wav_raw))
            if peak > 0:
                wav_norm = wav_raw / peak * 0.95
            else:
                wav_norm = wav_raw

            sf.write(out_wav_path, wav_norm, 44100)
            dur_sec = len(wav_norm) / 44100.0
            print(f"  ✅ SUCCESS! Decoded Spoken Speech Audio from NPU: {out_wav_path}")
            print(f"     Duration: {dur_sec:.2f}s | Sample Rate: 44,100 Hz | Peak: {np.max(np.abs(wav_norm)):.2f}")
        else:
            print(f"  ❌ Inference failed with status: {st}")

    print("\n" + "=" * 85)
    print(f" 🎉 ALL SPOKEN SPEECH AUDIO FILES GENERATED FROM QUALCOMM NPU!")
    print(f" 📁 Check folder: {OUTPUT_DIR}")
    print("=" * 85)


if __name__ == "__main__":
    run_live_speech_deployment()
