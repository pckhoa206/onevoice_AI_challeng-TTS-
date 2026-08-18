"""Compile Pure NPU W8A16 QNN Context Binaries for Supertonic 3 Submodels.

Submits the refactored NPU-compliant ONNX models to Qualcomm AI Hub with:
  • weights_dtype     : hub.QuantizeDtype.INT8
  • activations_dtype : hub.QuantizeDtype.INT16
  • options           : --target_runtime qnn_context_binary --compute_unit npu --truncate_64bit_io
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

DEFAULT_TARGET_DEVICE = "Samsung Galaxy S24 Ultra"
REFACTORED_DIR = "outputs/npu_compliant_onnx"
OUTPUT_DIR = "outputs/pure_npu_binaries_w8a16"

PURE_NPU_SUBMODELS = [
    {
        "name": "duration_predictor",
        "path": "outputs/npu_compliant_onnx/duration_predictor_npu.onnx",
        "specs": {"text_ids": ((1, 64), "int64"), "style_dp": ((1, 8, 16), "float32"), "text_mask": ((1, 1, 64), "float32")},
        "sample_calib": {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64)],
            "style_dp": [np.random.randn(1, 8, 16).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "text_encoder",
        "path": "outputs/npu_compliant_onnx/text_encoder_npu.onnx",
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
    {
        "name": "vector_estimator",
        "path": "outputs/npu_compliant_onnx/vector_estimator_npu.onnx",
        "specs": {
            "noisy_latent": ((1, 144, 100), "float32"),
            "text_emb": ((1, 256, 32), "float32"),
            "style_ttl": ((1, 50, 256), "float32"),
            "latent_mask": ((1, 1, 100), "float32"),
            "text_mask": ((1, 1, 32), "float32"),
            "current_step": ((1,), "float32"),
            "total_step": ((1,), "float32"),
        },
        "sample_calib": {
            "noisy_latent": [np.random.randn(1, 144, 100).astype(np.float32)],
            "text_emb": [np.random.randn(1, 256, 32).astype(np.float32)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32)],
            "latent_mask": [np.ones((1, 1, 100), dtype=np.float32)],
            "text_mask": [np.ones((1, 1, 32), dtype=np.float32)],
            "current_step": [np.array([1.0], dtype=np.float32)],
            "total_step": [np.array([5.0], dtype=np.float32)],
        },
    },
    {
        "name": "vocoder",
        "path": "outputs/npu_compliant_onnx/vocoder_npu.onnx",
        "specs": {"latent": ((1, 144, 100), "float32")},
        "sample_calib": {
            "latent": [np.random.randn(1, 144, 100).astype(np.float32)],
        },
    },
]


def compile_pure_npu_pipeline(device_name: str = DEFAULT_TARGET_DEVICE):
    _ensure_utf8_stdout()
    print("=" * 80)
    print(" 🚀 QUALCOMM AI HUB — PURE 100% NPU COMPILATION PIPELINE")
    print(" • Compute Unit        : Qualcomm Hexagon HTP NPU (100% Offload)")
    print(" • Weights Precision    : INT8")
    print(" • Activations Precision: INT16")
    print(f" • Target Device        : {device_name}")
    print("=" * 80)

    try:
        import qai_hub as hub
    except ImportError:
        print("[compile_pure_npu] Error: 'qai-hub' is not installed.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = hub.Device(device_name)
    results = {}

    for idx, submodel in enumerate(PURE_NPU_SUBMODELS, 1):
        name = submodel["name"]
        path = submodel["path"]
        specs = submodel["specs"]
        calib_data = submodel["sample_calib"]

        print(f"\n[{idx}/{len(PURE_NPU_SUBMODELS)}] Processing 100% NPU Submodel: '{name.upper()}'...")
        if not os.path.exists(path):
            print(f"  ❌ Refactored model '{path}' not found.")
            continue

        try:
            print(f"  1. Uploading refactored '{os.path.basename(path)}' to Qualcomm AI Hub...")
            model = hub.upload_model(path)
            print(f"     Uploaded Model ID: {model.model_id}")

            print(f"  2. Submitting Static ONNX Compilation...")
            static_compile_job = hub.submit_compile_job(
                model=model,
                device=device,
                input_specs=specs,
                options="--target_runtime onnx",
            )
            print(f"     Static Compile Job ID: {static_compile_job.job_id}")
            static_compile_job.wait()
            static_model = static_compile_job.get_target_model()

            print(f"  3. Submitting W8A16 Quantize Job...")
            quant_job = hub.submit_quantize_job(
                model=static_model,
                calibration_data=calib_data,
                weights_dtype=hub.QuantizeDtype.INT8,
                activations_dtype=hub.QuantizeDtype.INT16,
                name=f"{name}_pure_npu_w8a16",
            )
            print(f"     Quantize Job ID: {quant_job.job_id}")
            quant_job.wait()
            quantized_model = quant_job.get_target_model()

            print(f"  4. Submitting QNN Context Binary NPU Compilation (--compute_unit npu)...")
            compile_job = hub.submit_compile_job(
                model=quantized_model,
                device=device,
                input_specs=specs,
                options="--target_runtime qnn_context_binary --compute_unit npu --truncate_64bit_io",
            )
            print(f"     NPU Compile Job ID: {compile_job.job_id}")
            print(f"     Dashboard URL: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
            compile_job.wait()

            target_model = compile_job.get_target_model()
            out_bin = os.path.join(OUTPUT_DIR, f"{name}_pure_npu_w8a16.bin")
            target_model.download(out_bin)
            print(f"  ✅ SUCCESS! Downloaded 100% NPU Binary: {out_bin}")
            results[name] = out_bin

        except Exception as e:
            print(f"  ❌ Submodel '{name}' failed NPU compilation: {e}")

    print("\n" + "=" * 80)
    print(" 🎉 PURE 100% NPU COMPILATION PIPELINE SUMMARY")
    print("=" * 80)
    for model_name, bin_path in results.items():
        print(f" • {model_name:<18}: {bin_path}")
    print("=" * 80)


if __name__ == "__main__":
    compile_pure_npu_pipeline()
