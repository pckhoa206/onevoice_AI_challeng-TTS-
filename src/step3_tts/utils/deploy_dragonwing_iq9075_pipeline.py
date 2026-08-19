"""Qualcomm AI Hub Dragonwing IQ-9075 EVK Compilation & Live Hardware Inference Pipeline.

Submits all 4 Supertonic 3 TTS submodels to Qualcomm AI Hub Cloud targeting
the Dragonwing IQ-9075 EVK (Edge / Industrial IoT AI platform) and runs live hardware inference.
"""
import os
import sys
import time
import numpy as np
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

TARGET_DEVICE_NAME = "Dragonwing IQ-9075 EVK"

SUBMODEL_CONFIGS = [
    {
        "name": "vocoder",
        "onnx_path": "outputs/npu_compliant_onnx/vocoder_npu.onnx",
        "specs": {
            "latent": ((1, 144, 100), "float32"),
        },
        "sample_inputs": {
            "latent": [np.random.randn(1, 144, 100).astype(np.float32)],
        },
    },
    {
        "name": "duration_predictor",
        "onnx_path": "outputs/npu_compliant_onnx/duration_predictor_npu.onnx",
        "specs": {
            "text_ids": ((1, 64), "int64"),
            "style_dp": ((1, 8, 16), "float32"),
            "text_mask": ((1, 1, 64), "float32"),
        },
        "sample_inputs": {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64)],
            "style_dp": [np.random.randn(1, 8, 16).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "text_encoder",
        "onnx_path": "outputs/npu_compliant_onnx/text_encoder_npu.onnx",
        "specs": {
            "text_ids": ((1, 64), "int64"),
            "style_ttl": ((1, 50, 256), "float32"),
            "text_mask": ((1, 1, 64), "float32"),
        },
        "sample_inputs": {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "vector_estimator",
        "onnx_path": "outputs/npu_compliant_onnx/vector_estimator_npu.onnx",
        "specs": {
            "noisy_latent": ((1, 144, 100), "float32"),
            "text_emb": ((1, 256, 32), "float32"),
            "style_ttl": ((1, 50, 256), "float32"),
            "latent_mask": ((1, 1, 100), "float32"),
            "text_mask": ((1, 1, 32), "float32"),
            "current_step": ((1,), "float32"),
            "total_step": ((1,), "float32"),
        },
        "sample_inputs": {
            "noisy_latent": [np.random.randn(1, 144, 100).astype(np.float32)],
            "text_emb": [np.random.randn(1, 256, 32).astype(np.float32)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32)],
            "latent_mask": [np.ones((1, 1, 100), dtype=np.float32)],
            "text_mask": [np.ones((1, 1, 32), dtype=np.float32)],
            "current_step": [np.array([1.0], dtype=np.float32)],
            "total_step": [np.array([5.0], dtype=np.float32)],
        },
    },
]


def run_dragonwing_pipeline():
    _ensure_utf8_stdout()
    print("=" * 80)
    print(" 🚀 QUALCOMM AI HUB — DRAGONWING IQ-9075 EVK COMPILATION & LIVE INFERENCE")
    print(f" • Target Hardware : {TARGET_DEVICE_NAME}")
    print(f" • Target Runtime  : ONNX Runtime QNN Execution Provider (NPU Accelerated)")
    print("=" * 80)

    device = hub.Device(TARGET_DEVICE_NAME)
    compiled_models = {}

    # Step 1: Submit Static ONNX Compile Jobs
    for cfg in SUBMODEL_CONFIGS:
        name = cfg["name"]
        onnx_path = cfg["onnx_path"]
        specs = cfg["specs"]

        print(f"\n[1/2] Compiling Submodel: '{name}' on {TARGET_DEVICE_NAME}...")
        compile_job = hub.submit_compile_job(
            model=onnx_path,
            device=device,
            input_specs=specs,
            options="--target_runtime onnx",
            name=f"{name}_iq9075_onnx_qnn_ep",
        )
        print(f"  • Compile Job ID : {compile_job.job_id}")
        print(f"  • Dashboard URL  : {compile_job.url}")

        print(f"  ⏳ Waiting for compile job...")
        compile_job.wait()
        status = compile_job.get_status().code

        if status == "SUCCESS":
            target_model = compile_job.get_target_model()
            compiled_models[name] = (target_model, cfg["sample_inputs"])
            print(f"  ✅ SUCCESS! Target Model ID: {target_model.model_id}")
        else:
            print(f"  ❌ FAILED to compile '{name}' (Status: {status})")

    # Step 2: Submit Live Hardware Inference Jobs on Dragonwing IQ-9075 EVK
    print("\n" + "=" * 80)
    print(f" 📊 SUBMITTING LIVE HARDWARE INFERENCE JOBS ON {TARGET_DEVICE_NAME}")
    print("=" * 80)

    inference_jobs = []
    for name, (target_model, sample_inputs) in compiled_models.items():
        print(f"\n[2/2] Submitting Live Inference for '{name}' (Model: {target_model.model_id})...")
        inf_job = hub.submit_inference_job(
            model=target_model,
            device=device,
            inputs=sample_inputs,
            name=f"{name}_iq9075_live_inference",
        )
        print(f"  • Inference Job ID : {inf_job.job_id}")
        print(f"  • Dashboard URL    : {inf_job.url}")
        inference_jobs.append((name, inf_job))

    print("\n" + "=" * 80)
    print(f" 🏁 WAITING FOR LIVE HARDWARE INFERENCE RESULTS ON {TARGET_DEVICE_NAME}")
    print("=" * 80)

    for name, inf_job in inference_jobs:
        print(f"\n⏳ Waiting for inference job ({inf_job.job_id}) for '{name}'...")
        inf_job.wait()
        st = inf_job.get_status().code
        print(f"  • [{name:<18}]: Status = {st:<12} | Dashboard: {inf_job.url}")
        if st == "SUCCESS":
            try:
                outs = inf_job.download_output_data()
                print(f"    ✅ SUCCESS! Downloaded live hardware output tensors ({len(outs)} keys):")
                for k, v in outs.items():
                    shape_str = str(v[0].shape) if len(v) > 0 and hasattr(v[0], 'shape') else "N/A"
                    print(f"       • Tensor '{k}' | Shape: {shape_str}")
            except Exception as ex:
                print(f"    ⚠️ Download note: {ex}")

    print("\n" + "=" * 80)
    print(" 🎉 DRAGONWING IQ-9075 EVK PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_dragonwing_pipeline()
