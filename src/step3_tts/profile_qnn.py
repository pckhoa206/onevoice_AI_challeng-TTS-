"""Qualcomm AI Hub (qai-hub) Remote Profiling Script for Step 3 TTS.
Automates submitting Piper ONNX, Supertonic ONNX, and MeloTTS checkpoints
to Qualcomm AI Hub to obtain true Snapdragon NPU latency, power, and memory metrics.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout

DEFAULT_TARGET_DEVICE = "Snapdragon 8 Elite QCP"


def profile_model(model_path: str, device_name: str = DEFAULT_TARGET_DEVICE):
    """Submit an ONNX model to Qualcomm AI Hub for remote NPU profiling."""
    print(f"[profile_qnn] Connecting to Qualcomm AI Hub for model: {model_path}")
    try:
        import qai_hub as hub
    except ImportError:
        print("[profile_qnn] 'qai_hub' library not installed. Install via: pip install qai-hub")
        print("[profile_qnn] To configure API key: qai-hub configure --api_token <YOUR_TOKEN>")
        return

    if not os.path.exists(model_path):
        print(f"[profile_qnn] Error: Model file '{model_path}' not found.")
        return

    print(f"[profile_qnn] Submitting model '{model_path}' to device '{device_name}'...")
    try:
        device = hub.Device(device_name)
        model = hub.upload_model(model_path)
        profile_job = hub.submit_profile_job(
            model=model,
            device=device,
            options="--compute_unit NPU",
        )
        print(f"[profile_qnn] Profile job submitted successfully. Job ID: {profile_job.job_id}")
        print("[profile_qnn] Waiting for profile job completion...")
        profile_job.wait()
        
        profile_data = profile_job.download_profile()
        print(f"[profile_qnn] Profile completed! Results for {os.path.basename(model_path)}:")
        print(profile_data)
    except Exception as e:
        print(f"[profile_qnn] Failed to profile model on Qualcomm AI Hub: {e}")


def main():
    parser = argparse.ArgumentParser(description="Qualcomm AI Hub Remote Profiling for Step 3 TTS")
    parser.add_argument("--model", type=str, default="vi_VN-vais1000-medium.onnx", help="Path to ONNX model file")
    parser.add_argument("--device", type=str, default=DEFAULT_TARGET_DEVICE, help="Target Qualcomm device")
    args = parser.parse_args()

    profile_model(args.model, args.device)


if __name__ == "__main__":
    main()
