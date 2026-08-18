"""Qualcomm AI Hub Live Hardware Inference Runner for Supertonic 3 Submodels.

Submits live inference jobs on target device (Samsung Galaxy S24 Ultra) via Qualcomm AI Hub,
executing model inference on real Snapdragon 8 Gen 3 hardware and downloading output tensors.
"""
import os
import sys
import time
import numpy as np
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

TARGET_DEVICE_NAME = "Samsung Galaxy S24 Ultra"

# Models deployed on Qualcomm AI Hub
SUBMODELS_FOR_INFERENCE = [
    {
        "name": "duration_predictor",
        "model_id": "mqvkw61xn",
        "inputs": {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64)],
            "style_dp": [np.random.randn(1, 8, 16).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "text_encoder",
        "model_id": "mn10xw28q",
        "inputs": {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32)],
        },
    },
    {
        "name": "vector_estimator",
        "model_id": "mnwx48wpm",
        "inputs": {
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
        "name": "vocoder",
        "model_id": "mq2yg1z7m",
        "inputs": {
            "latent": [np.random.randn(1, 144, 100).astype(np.float32)],
        },
    },
]


def run_qualcomm_inference_pipeline():
    _ensure_utf8_stdout()
    print("=" * 80)
    print(" 🚀 QUALCOMM AI HUB — LIVE HARDWARE INFERENCE DEPLOYMENT")
    print(f" • Target Hardware : {TARGET_DEVICE_NAME}")
    print(f" • Target SoC      : Qualcomm Snapdragon 8 Gen 3 (Hexagon NPU)")
    print("=" * 80)

    device = hub.Device(TARGET_DEVICE_NAME)
    inference_jobs = []

    # 1. Submit Live Hardware Inference Jobs for all submodels
    for item in SUBMODELS_FOR_INFERENCE:
        name = item["name"]
        model_id = item["model_id"]
        inputs = item["inputs"]

        print(f"\n[Inference] Submodel: '{name}' (Model ID: {model_id})...")
        print(f"  1. Loading Model from Qualcomm AI Hub Cloud...")
        model = hub.get_model(model_id)

        print(f"  2. Submitting Live Hardware Inference Job on {TARGET_DEVICE_NAME}...")
        inf_job = hub.submit_inference_job(
            model=model,
            device=device,
            inputs=inputs,
            name=f"{name}_live_npu_inference",
        )
        print(f"     Inference Job ID : {inf_job.job_id}")
        print(f"     Dashboard URL    : {inf_job.url}")

        inference_jobs.append({
            "name": name,
            "job": inf_job,
        })

    # 2. Wait for jobs and retrieve output tensors
    print("\n" + "=" * 80)
    print(" 📊 WAITING FOR LIVE INFERENCE RESULTS & DOWNLOADING OUTPUT TENSORS")
    print("=" * 80)

    summary_table = []
    for res in inference_jobs:
        name = res["name"]
        job = res["job"]

        print(f"\n⏳ Waiting for live inference job ({job.job_id}) for '{name}'...")
        job.wait()
        status = job.get_status().code

        if status == "SUCCESS":
            try:
                output_data = job.download_output_data()
                print(f"  ✅ SUCCESS! Live Hardware Inference completed.")
                print(f"     Output Tensors ({len(output_data)} keys):")
                for key, val_list in output_data.items():
                    shape_str = str(val_list[0].shape) if len(val_list) > 0 and hasattr(val_list[0], 'shape') else "N/A"
                    print(f"       • Output '{key}' | Shape: {shape_str}")
                summary_table.append((name, job.job_id, "SUCCESS", job.url))
            except Exception as ex:
                print(f"  ⚠️ Completed with output parse note: {ex}")
                summary_table.append((name, job.job_id, f"SUCCESS (Note: {ex})", job.url))
        else:
            summary_table.append((name, job.job_id, f"FAILED ({status})", job.url))

    print("\n" + "=" * 80)
    print(" 🏁 QUALCOMM AI HUB LIVE HARDWARE INFERENCE SUMMARY REPORT")
    print("=" * 80)
    for name, jid, status, url in summary_table:
        print(f" • [{name:<20}] | Job ID: {jid} | Status: {status}")
        print(f"   Dashboard URL: {url}")
    print("=" * 80)


if __name__ == "__main__":
    run_qualcomm_inference_pipeline()
