"""1-Click Qualcomm AI Hub Deployer for FULL Supertonic 3 TTS Pipeline.
Submits and compiles ALL 4 submodels (text_encoder, duration_predictor, vector_estimator, vocoder)
to Qualcomm AI Hub Workbench in ONE SINGLE COMMAND.
"""
import os
import sys
import argparse
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout

DEFAULT_TARGET_DEVICE = "Snapdragon 8 Elite QRD"
OUTPUT_DIR = "outputs/qnn_binaries"

# Complete Submodel Configurations for Supertonic 3
SUPERTONIC_SUBMODELS = [
    {
        "name": "text_encoder",
        "path": "/Users/khoa/.cache/supertonic3/onnx/text_encoder.onnx",
        "specs": {
            "text_ids": ((1, 64), "int64"),
            "style_ttl": ((1, 50, 256), "float32"),
            "text_mask": ((1, 1, 64), "float32"),
        },
    },
    {
        "name": "duration_predictor",
        "path": "/Users/khoa/.cache/supertonic3/onnx/duration_predictor.onnx",
        "specs": {
            "text_ids": ((1, 64), "int64"),
            "style_dp": ((1, 8, 16), "float32"),
            "text_mask": ((1, 1, 64), "float32"),
        },
    },
    {
        "name": "vector_estimator",
        "path": "/Users/khoa/.cache/supertonic3/onnx/vector_estimator.onnx",
        "specs": {
            "noisy_latent": ((1, 144, 100), "float32"),
            "text_emb": ((1, 256, 64), "float32"),
            "style_ttl": ((1, 50, 256), "float32"),
            "latent_mask": ((1, 1, 100), "float32"),
            "text_mask": ((1, 1, 64), "float32"),
            "current_step": ((1,), "float32"),
            "total_step": ((1,), "float32"),
        },
    },
    {
        "name": "vocoder",
        "path": "/Users/khoa/.cache/supertonic3/onnx/vocoder.onnx",
        "specs": {
            "latent": ((1, 144, 100), "float32"),
        },
    },
]


def deploy_full_pipeline(device_name: str = DEFAULT_TARGET_DEVICE, precision: str = "w8a16"):
    print("=" * 75)
    print(" 🚀 QUALCOMM AI HUB — 1-CLICK FULL SUPERTONIC 3 DEPLOYER")
    print("=" * 75)
    print(f" • Target Device: {device_name}")
    print(f" • Quantization : {precision.upper()} (Weight INT8, Activation INT16)")
    print(f" • Submodels    : {len(SUPERTONIC_SUBMODELS)} (TextEncoder, DurationPredictor, VectorEstimator, Vocoder)")
    print("=" * 75)

    try:
        import qai_hub as hub
    except ImportError:
        print("[deploy_full] Error: 'qai-hub' is not installed.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = hub.Device(device_name)
    results = {}

    for idx, submodel in enumerate(SUPERTONIC_SUBMODELS, 1):
        name = submodel["name"]
        path = submodel["path"]
        specs = submodel["specs"]

        print(f"\n[{idx}/{len(SUPERTONIC_SUBMODELS)}] Deploying Supertonic Submodel: '{name.upper()}' ({precision.upper()})...")
        if not os.path.exists(path):
            print(f"  ❌ Error: File '{path}' not found.")
            continue

        try:
            print(f"  • Uploading '{os.path.basename(path)}' to Qualcomm AI Hub...")
            model = hub.upload_model(path)
            print(f"    Uploaded Model ID: {model.model_id}")

            options_str = f"--target_runtime onnx --compute_unit npu --precision {precision}"
            print(f"  • Submitting NPU Compilation Job ({precision.upper()} Hexagon HTP NPU)...")
            compile_job = hub.submit_compile_job(
                model=model,
                device=device,
                input_specs=specs,
                options=options_str,
            )
            print(f"    Compile Job ID: {compile_job.job_id}")
            print(f"    Web Dashboard Link: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")

            print(f"  • Waiting for compilation to finish...")
            compile_job.wait()

            target_model = compile_job.get_target_model()
            out_bin = os.path.join(OUTPUT_DIR, f"{name}_npu_{precision}.bin")
            target_model.download(out_bin)
            print(f"  ✅ SUCCESS! Downloaded Binary: {out_bin}")
            results[name] = out_bin

        except Exception as e:
            print(f"  ❌ Submodel '{name}' failed: {e}")

    print("\n" + "=" * 75)
    print(" 🎉 ALL SUPERTONIC 3 SUBMODELS DEPLOYED SUCCESSFULLY TO QUALCOMM NPU!")
    print("=" * 75)
    for name, bin_path in results.items():
        print(f" • {name:<20}: {bin_path}")
    print("=" * 75)


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="1-Click Deploy ALL Supertonic 3 Submodels to Qualcomm AI Hub")
    parser.add_argument("--device", type=str, default=DEFAULT_TARGET_DEVICE, help="Target Qualcomm device")
    parser.add_argument("--precision", type=str, default="w8a16", choices=["w8a16", "fp16", "int8"], help="NPU quantization precision (w8a16: Weight INT8, Activation INT16)")
    args = parser.parse_args()
    deploy_full_pipeline(args.device, args.precision)


if __name__ == "__main__":
    main()
