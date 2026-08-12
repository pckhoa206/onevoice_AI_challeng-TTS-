"""Qualcomm AI Hub Deployer for Supertonic 3 TTS.
Automates submitting Supertonic 3 ONNX sub-models (vector_estimator, vocoder, text_encoder)
to Qualcomm AI Hub Workbench (workbench.aihub.qualcomm.com) for NPU compilation, quantization,
and remote profiling on Snapdragon hardware (Snapdragon 8 Elite / QCS6490 Rubik Pi 3).
"""
import os
import sys
import argparse
from typing import Optional, Dict, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout

DEFAULT_TARGET_DEVICE = "Snapdragon 8 Elite QRD"

# Static input shape & dtype specifications for Qualcomm Hexagon NPU compilation
STATIC_INPUT_SPECS: Dict[str, Dict[str, Tuple[Tuple[int, ...], str]]] = {
    "duration_predictor": {
        "text_ids": ((1, 64), "int64"),
        "style_dp": ((1, 8, 16), "float32"),
        "text_mask": ((1, 1, 64), "float32"),
    },
    "vocoder": {
        "latent": ((1, 144, 100), "float32"),
    },
    "vector_estimator": {
        "noisy_latent": ((1, 144, 100), "float32"),
        "text_emb": ((1, 256, 64), "float32"),
        "style_ttl": ((1, 50, 256), "float32"),
        "latent_mask": ((1, 1, 100), "float32"),
        "text_mask": ((1, 1, 64), "float32"),
        "current_step": ((1,), "float32"),
        "total_step": ((1,), "float32"),
    },
    "text_encoder": {
        "text_ids": ((1, 64), "int64"),
        "style_ttl": ((1, 50, 256), "float32"),
        "text_mask": ((1, 1, 64), "float32"),
    },
}


def get_input_specs_for_model(model_path: str) -> Optional[Dict[str, Tuple[Tuple[int, ...], str]]]:
    filename = os.path.basename(model_path).lower()
    for key, spec in STATIC_INPUT_SPECS.items():
        if key in filename:
            return spec
    return None


def deploy_supertonic_submodel(
    model_path: str,
    device_name: str = DEFAULT_TARGET_DEVICE,
    precision: str = "fp16",
    output_dir: str = "outputs/qnn_binaries",
) -> Optional[str]:
    """Submit a Supertonic ONNX submodel to Qualcomm AI Hub for NPU compilation & profiling."""
    print("=" * 75)
    print(f"  QUALCOMM AI HUB WORKBENCH — SUPERTONIC 3 NPU DEPLOYER")
    print("=" * 75)
    print(f"  • Submodel Path  : {model_path}")
    print(f"  • Target Device  : {device_name}")
    print(f"  • NPU Precision  : {precision.upper()}")
    print("=" * 75)

    try:
        import qai_hub as hub
    except ImportError:
        print("\n[deploy_supertonic] Error: 'qai-hub' library is not installed.")
        print("[deploy_supertonic] Install via: pip install qai-hub")
        print("[deploy_supertonic] Configure API Token: qai-hub configure --api_token <YOUR_TOKEN>")
        return None

    if not os.path.exists(model_path):
        print(f"\n[deploy_supertonic] Error: Submodel file '{model_path}' not found.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    input_specs = get_input_specs_for_model(model_path)

    try:
        # 1. Target Qualcomm Device
        device = hub.Device(device_name)
        print(f"\n[1/4] Connected to Qualcomm Device: {device.name}")

        # 2. Upload ONNX model to AI Hub Workbench
        print(f"[2/4] Uploading '{os.path.basename(model_path)}' to Qualcomm AI Hub Workbench...")
        uploaded_model = hub.upload_model(model_path)
        print(f"      Uploaded Model ID: {uploaded_model.model_id}")

        # 3. Submit NPU Compilation Job (Hexagon HTP NPU)
        print(f"[3/4] Submitting NPU Compilation Job (Hexagon HTP NPU)...")
        options_str = f"--target_runtime onnx --compute_unit npu --precision {precision}"
        if input_specs:
            print(f"      Static Input Specs & Dtypes Provided: {input_specs}")
            compile_job = hub.submit_compile_job(
                model=uploaded_model,
                device=device,
                input_specs=input_specs,
                options=options_str,
            )
        else:
            compile_job = hub.submit_compile_job(
                model=uploaded_model,
                device=device,
                options=options_str,
            )

        print(f"      Compile Job ID: {compile_job.job_id}")
        print(f"      Web Dashboard Link: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
        print("      Waiting for NPU compilation completion...")
        compile_job.wait()

        target_model = compile_job.get_target_model()

        # 4. Profile on-device Latency & Power
        print(f"[4/4] Submitting Remote Profile Job on target Snapdragon device...")
        profile_job = hub.submit_profile_job(
            model=target_model,
            device=device,
        )
        print(f"      Profile Job ID: {profile_job.job_id}")
        print(f"      Web Dashboard Link: https://workbench.aihub.qualcomm.com/jobs/{profile_job.job_id}/")
        profile_job.wait()

        profile_data = profile_job.download_profile()
        print("\n" + "=" * 75)
        print("  QUALCOMM NPU PROFILING RESULTS:")
        print("=" * 75)
        print(profile_data)

        # 5. Download compiled QNN NPU binary for edge deployment
        submodel_basename = os.path.splitext(os.path.basename(model_path))[0]
        out_bin_path = os.path.join(output_dir, f"{submodel_basename}_npu_{precision}.bin")
        target_model.download(out_bin_path)

        print("\n" + "=" * 75)
        print(f"  ✅ DEPLOYMENT PACKAGE EXPORTED SUCCESSFULLY")
        print(f"  • Downloaded QNN Binary: {out_bin_path}")
        print(f"  • Ready for ONNX Runtime QNN Execution Provider on device!")
        print("=" * 75)
        return out_bin_path

    except Exception as e:
        print(f"\n[deploy_supertonic] Deployment failed: {e}")
        return None


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Qualcomm AI Hub Deployer for Supertonic 3 TTS")
    parser.add_argument("--model", type=str, default="/Users/khoa/.cache/supertonic3/onnx/duration_predictor.onnx", help="Path to Supertonic submodel ONNX")
    parser.add_argument("--device", type=str, default=DEFAULT_TARGET_DEVICE, help="Target Qualcomm device")
    parser.add_argument("--precision", type=str, default="w8a16", choices=["w8a16", "fp16", "int8"], help="NPU quantization precision (w8a16: Weight INT8, Activation INT16)")
    args = parser.parse_args()

    deploy_supertonic_submodel(args.model, args.device, args.precision)


if __name__ == "__main__":
    main()
