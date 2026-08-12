"""Quantize Supertonic 3 Submodels using Qualcomm AI Hub W8A16 (Weight INT8, Activation INT16).

This script uses Qualcomm AI Hub `submit_quantize_job` with:
  • weights_dtype     : hub.QuantizeDtype.INT8
  • activations_dtype : hub.QuantizeDtype.INT16
Followed by `submit_compile_job` for Hexagon HTP NPU execution.
"""
import os
import sys
import argparse
import numpy as np
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout

DEFAULT_TARGET_DEVICE = "Snapdragon 8 Elite QRD"
OUTPUT_DIR = "outputs/qnn_binaries_w8a16"

SUPERTONIC_SUBMODELS = [
    {
        "name": "duration_predictor",
        "path": "/Users/khoa/.cache/supertonic3/onnx/duration_predictor.onnx",
        "specs": {"text_ids": ((1, 64), "int64"), "style_dp": ((1, 8, 16), "float32"), "text_mask": ((1, 1, 64), "float32")},
        "sample_calib": {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64)],
            "style_dp": [np.random.randn(1, 8, 16).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "vocoder",
        "path": "/Users/khoa/.cache/supertonic3/onnx/vocoder.onnx",
        "specs": {"latent": ((1, 144, 100), "float32")},
        "sample_calib": {
            "latent": [np.random.randn(1, 144, 100).astype(np.float32)],
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
        "sample_calib": {
            "noisy_latent": [np.random.randn(1, 144, 100).astype(np.float32)],
            "text_emb": [np.random.randn(1, 256, 64).astype(np.float32)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32)],
            "latent_mask": [np.ones((1, 1, 100), dtype=np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
            "current_step": [np.array([1.0], dtype=np.float32)],
            "total_step": [np.array([5.0], dtype=np.float32)],
        },
    },
    {
        "name": "text_encoder",
        "path": "/Users/khoa/.cache/supertonic3/onnx/text_encoder.onnx",
        "specs": {
            "text_ids": ((1, 64), "int64"),
            "style_ttl": ((1, 50, 256), "float32"),
            "text_mask": ((1, 1, 64), "float32"),
        },
        "sample_calib": {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
]


def quantize_w8a16_pipeline(device_name: str = DEFAULT_TARGET_DEVICE):
    print("=" * 75)
    print(" ⚖️ QUALCOMM AI HUB — W8A16 QUANTIZATION PIPELINE")
    print(" • Weights Precision    : INT8 (8-bit Integer)")
    print(" • Activations Precision: INT16 (16-bit Integer)")
    print(" • Target Device        : " + device_name)
    print("=" * 75)

    try:
        import qai_hub as hub
    except ImportError:
        print("[quantize_w8a16] Error: 'qai-hub' is not installed.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = hub.Device(device_name)
    results = {}

    for idx, submodel in enumerate(SUPERTONIC_SUBMODELS, 1):
        name = submodel["name"]
        path = submodel["path"]
        specs = submodel["specs"]
        calib_data = submodel["sample_calib"]

        print(f"\n[{idx}/{len(SUPERTONIC_SUBMODELS)}] Quantizing Submodel: '{name.upper()}' (W8A16)...")
        if not os.path.exists(path):
            print(f"  ❌ File '{path}' not found.")
            continue

        try:
            print(f"  1. Uploading '{os.path.basename(path)}' to Qualcomm AI Hub...")
            model = hub.upload_model(path)
            print(f"     Uploaded Model ID: {model.model_id}")

            print(f"  2. Converting dynamic shapes to static ONNX model...")
            static_compile_job = hub.submit_compile_job(
                model=model,
                device=device,
                input_specs=specs,
                options="--target_runtime onnx",
            )
            print(f"     Static Compile Job ID: {static_compile_job.job_id}")
            print(f"     Waiting for static shape conversion...")
            static_compile_job.wait()

            static_model = static_compile_job.get_target_model()
            print(f"     Static Model ID: {static_model.model_id}")

            print(f"  3. Submitting W8A16 Quantize Job (Weights=INT8, Activations=INT16)...")
            quant_job = hub.submit_quantize_job(
                model=static_model,
                calibration_data=calib_data,
                weights_dtype=hub.QuantizeDtype.INT8,
                activations_dtype=hub.QuantizeDtype.INT16,
                name=f"{name}_w8a16",
            )
            print(f"     Quantize Job ID: {quant_job.job_id}")
            print(f"     Web Dashboard Link: https://workbench.aihub.qualcomm.com/jobs/{quant_job.job_id}/")

            print(f"     Waiting for W8A16 quantization to finish...")
            quant_job.wait()

            quantized_model = quant_job.get_target_model()
            print(f"     Quantized Model ID: {quantized_model.model_id}")

            print(f"  4. Submitting NPU Compilation Job for Hexagon HTP NPU...")
            compile_job = hub.submit_compile_job(
                model=quantized_model,
                device=device,
                input_specs=specs,
                options="--target_runtime onnx --compute_unit npu",
            )
            print(f"     Compile Job ID: {compile_job.job_id}")
            print(f"     Waiting for NPU compilation...")
            compile_job.wait()

            target_model = compile_job.get_target_model()
            out_bin = os.path.join(OUTPUT_DIR, f"{name}_npu_w8a16.bin")
            target_model.download(out_bin)
            print(f"  ✅ SUCCESS! Downloaded W8A16 Binary: {out_bin}")
            results[name] = out_bin

        except Exception as e:
            print(f"  ❌ Submodel '{name}' failed W8A16 Quantization: {e}")

    print("\n" + "=" * 75)
    print(" 🎉 W8A16 QUANTIZATION PIPELINE SUMMARY")
    print("=" * 75)
    for model_name, bin_path in results.items():
        size_mb = os.path.getsize(bin_path + ".zip") / (1024 * 1024) if os.path.exists(bin_path + ".zip") else 0
        print(f" • {model_name:<18}: {bin_path} ({size_mb:.2f} MB)")
    print("=" * 75)


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="W8A16 Quantization Deployer for Supertonic 3")
    parser.add_argument("--device", type=str, default=DEFAULT_TARGET_DEVICE, help="Target Qualcomm device")
    args = parser.parse_args()
    quantize_w8a16_pipeline(args.device)


if __name__ == "__main__":
    main()
