"""Compile All 4 Refactored Supertonic 3 Submodels to 100% Pure NPU on Qualcomm AI Hub.

Submits:
  1. Duration Predictor Pure NPU (outputs/pure_npu_compliant_onnx_v2/duration_predictor_pure_npu.onnx)
  2. Text Encoder Pure NPU (outputs/pure_npu_compliant_onnx_v2/text_encoder_pure_npu.onnx)
  3. Vector Estimator Unrolled Pure NPU (outputs/pure_npu_compliant_onnx_v2/vector_estimator_unrolled_5step_pure_npu.onnx)
  4. Vocoder Pure NPU (outputs/pure_npu_compliant_onnx_v2/vocoder_pure_npu.onnx)

Compile Target:
  • Target Device: Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3)
  • Compute Unit : --compute_unit npu (100% Pure NPU Offload)
  • Runtime      : --target_runtime qnn_context_binary --truncate_64bit_io
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

DEFAULT_TARGET_DEVICE = "Samsung Galaxy S24 Ultra"
MODEL_DIR = "outputs/pure_npu_compliant_onnx_v2"
OUTPUT_BIN_DIR = "outputs/pure_npu_binaries_v2"

SUBMODELS = [
    {
        "name": "duration_predictor",
        "path": os.path.join(MODEL_DIR, "duration_predictor_pure_npu.onnx"),
        "specs": {"char_emb": ((1, 64, 64), "float32"), "style_dp": ((1, 8, 16), "float32"), "text_mask": ((1, 1, 64), "float32")},
        "sample_calib": {
            "char_emb": [np.random.randn(1, 64, 64).astype(np.float32)],
            "style_dp": [np.random.randn(1, 8, 16).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "text_encoder",
        "path": os.path.join(MODEL_DIR, "text_encoder_pure_npu.onnx"),
        "specs": {
            "char_emb": ((1, 64, 256), "float32"),
            "style_ttl": ((1, 50, 256), "float32"),
            "text_mask": ((1, 1, 64), "float32"),
        },
        "sample_calib": {
            "char_emb": [np.random.randn(1, 64, 256).astype(np.float32)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "vector_estimator_unrolled",
        "path": os.path.join(MODEL_DIR, "vector_estimator_unrolled_5step_pure_npu.onnx"),
        "specs": {
            "noisy_latent": ((1, 144, 100), "float32"),
            "text_emb": ((1, 256, 64), "float32"),
            "style_ttl": ((1, 50, 256), "float32"),
            "latent_mask": ((1, 1, 100), "float32"),
            "text_mask": ((1, 1, 64), "float32"),
        },
        "sample_calib": {
            "noisy_latent": [np.random.randn(1, 144, 100).astype(np.float32)],
            "text_emb": [np.random.randn(1, 256, 64).astype(np.float32)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32)],
            "latent_mask": [np.ones((1, 1, 100), dtype=np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "vocoder",
        "path": os.path.join(MODEL_DIR, "vocoder_pure_npu.onnx"),
        "specs": {"latent": ((1, 144, 100), "float32")},
        "sample_calib": {
            "latent": [np.random.randn(1, 144, 100).astype(np.float32)],
        },
    },
]


def compile_pure_npu_v2(device_name: str = DEFAULT_TARGET_DEVICE):
    _ensure_utf8_stdout()
    print("=" * 85)
    print(" 🚀 QUALCOMM AI HUB — PURE 100% NPU V2 COMPILATION PIPELINE")
    print(f" • Target Device  : {device_name}")
    print(" • Target Runtime : QNN Context Binary (.bin)")
    print(" • Compute Unit   : Qualcomm Hexagon HTP NPU (100% Offload, 0% CPU Fallback)")
    print("=" * 85)

    try:
        import qai_hub as hub
    except ImportError:
        print("Error: 'qai-hub' not installed.")
        return

    os.makedirs(OUTPUT_BIN_DIR, exist_ok=True)
    device = hub.Device(device_name)
    results = {}

    for idx, item in enumerate(SUBMODELS, 1):
        name = item["name"]
        path = item["path"]
        specs = item["specs"]
        calib = item["sample_calib"]

        print(f"\n[{idx}/{len(SUBMODELS)}] Compiling 100% Pure NPU Submodel: '{name}'...")
        if not os.path.exists(path):
            print(f"  ❌ File missing: '{path}'")
            continue

        try:
            print(f"  1. Uploading refactored model '{os.path.basename(path)}'...")
            model = hub.upload_model(path)
            print(f"     Uploaded Model ID: {model.model_id}")

            print(f"  2. Submitting QNN Context Binary Compile Job (--compute_unit npu)...")
            compile_job = hub.submit_compile_job(
                model=model,
                device=device,
                input_specs=specs,
                options="--target_runtime qnn_context_binary --compute_unit npu --truncate_64bit_io",
                name=f"pure_npu_v2_{name}",
            )
            print(f"     Compile Job ID: {compile_job.job_id}")
            print(f"     Dashboard URL: {compile_job.url}")

            results[name] = {
                "job_id": compile_job.job_id,
                "url": compile_job.url,
            }

        except Exception as err:
            print(f"  ❌ Submodel '{name}' failed submission: {err}")

    print("\n" + "=" * 85)
    print(" 🏁 PURE NPU V2 COMPILATION JOBS SUBMITTED")
    print("=" * 85)
    for name, info in results.items():
        print(f" • [{name:<25}]: Job ID = {info['job_id']} | URL = {info['url']}")


if __name__ == "__main__":
    compile_pure_npu_v2()
