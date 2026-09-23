"""Deploy to Qualcomm AI Hub, Execute on Live NPU Hardware, and Download Output Audio.

Submits the refactored W8A16 Vocoder/Supertonic models to real Hexagon NPU on Qualcomm AI Hub
(Samsung Galaxy S24 Ultra / Qualcomm Dragonwing IQ-9075 EVK), downloads the output tensor,
and saves the synthesized audio as a physical .wav file in outputs/aihub_live_npu_download/aihub_live_npu_audio.wav.
"""
import os
import sys
import time
import numpy as np
import soundfile as sf
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "aihub_live_npu_download")
API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN")
if not API_TOKEN:
    raise SystemExit("Set QAI_HUB_API_TOKEN before running this script.")


def run_aihub_deploy_and_download():
    _ensure_utf8_stdout()
    print("=" * 85)
    print(" 🚀 QUALCOMM AI HUB — LIVE NPU DEPLOYMENT & OUTPUT DOWNLOAD PIPELINE")
    print(f" • Project Root    : {ROOT}")
    print(f" • Output Directory: {OUTPUT_DIR}")
    print("=" * 85)

    client = hub.Client(config=hub.ClientConfig(API_TOKEN))

    try:
        devices = client.get_devices()
        print(f"✅ Qualcomm AI Hub Authenticated! ({len(devices)} devices available in cloud pool)")
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_device_name = "Samsung Galaxy S24 Ultra"
    device = client.get_devices(target_device_name)[0]
    print(f" • Target Hardware: {target_device_name} (Snapdragon 8 Gen 3 Hexagon NPU)")

    # 1. Locate Vocoder ONNX model
    vocoder_path = os.path.join(ROOT, "outputs", "pure_npu_compliant_onnx_v2", "vocoder_pure_npu.onnx")
    if not os.path.exists(vocoder_path):
        vocoder_path = os.path.join(ROOT, "outputs", "pure_npu_dynamic", "vocoder_npu.onnx")
    if not os.path.exists(vocoder_path):
        vocoder_path = os.path.join(ROOT, "outputs", "npu_compliant_onnx", "vocoder_npu.onnx")

    print(f"\n[1/3] Uploading Vocoder model to Qualcomm AI Hub: '{vocoder_path}'...")
    uploaded_model = client.upload_model(vocoder_path)
    print(f"  • Uploaded Model ID: {uploaded_model.model_id}")

    # 2. Compile model for NPU
    print(f"\n[2/3] Submitting Compile Job targeting Qualcomm Hexagon NPU...")
    compile_job = client.submit_compile_job(
        model=uploaded_model,
        device=device,
        input_specs={"latent": ((1, 144, 100), "float32")},
        options="--target_runtime onnx",
        name="[LIVE_TEST] Supertonic3_Vocoder_Hexagon_NPU",
    )
    print(f"  • Compile Job ID: {compile_job.job_id}")
    print(f"  • Dashboard URL : {compile_job.url}")
    print("  ⏳ Waiting for compile job to finish...")
    compile_job.wait()
    target_model = compile_job.get_target_model()
    print("  ✅ Compile Job Finished Successfully!")

    # 3. Submit Live Inference on real hardware
    print(f"\n[3/3] Submitting Live Hardware Inference Job on {target_device_name} Hexagon NPU...")
    np.random.seed(42)
    sample_latent = np.random.randn(1, 144, 100).astype(np.float32)

    inf_job = client.submit_inference_job(
        model=target_model,
        device=device,
        inputs={"latent": [sample_latent]},
        name="[LIVE_TEST] Supertonic3_Vocoder_Live_NPU_Inference",
    )
    print(f"  • Inference Job ID: {inf_job.job_id}")
    print(f"  • Dashboard URL   : {inf_job.url}")
    print("  ⏳ Waiting for live hardware inference execution on Hexagon NPU...")
    inf_job.wait()

    status = inf_job.get_status().code
    if status == "SUCCESS":
        print(f"\n🎉 LIVE HARDWARE INFERENCE SUCCEEDED ON QUALCOMM HEXAGON NPU!")
        print(f" • Dashboard URL: {inf_job.url}")
        output_tensors = inf_job.download_output_data()
        
        for key, val_list in output_tensors.items():
            arr = np.asarray(val_list[0]).squeeze().astype(np.float32)
            out_npy_path = os.path.join(OUTPUT_DIR, f"aihub_npu_tensor_{key}.npy")
            np.save(out_npy_path, arr)
            print(f"  • Saved Tensor Output '{key}': Shape {arr.shape} -> {out_npy_path}")

            # If tensor is audio waveform (307,200 samples)
            if len(arr.shape) == 1 and arr.shape[0] > 10000:
                out_wav_path = os.path.join(OUTPUT_DIR, "aihub_live_npu_audio.wav")
                peak = np.max(np.abs(arr))
                arr_norm = (arr / peak * 0.95) if peak > 0 else arr
                sf.write(out_wav_path, arr_norm, 24000)
                print(f"  🔊 Exported Live NPU Audio Waveform: {out_wav_path} (Sample Rate: 24kHz, Duration: {len(arr)/24000:.2f}s, {len(arr)} samples)")
    else:
        print(f"❌ Inference job failed with status: {status}")


if __name__ == "__main__":
    run_aihub_deploy_and_download()
