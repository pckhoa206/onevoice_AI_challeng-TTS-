"""Empirical Verification Script for Cosine Similarity & SNR of W8A16 Quantized Supertonic 3.

Compares output tensors of Original FP32 ONNX submodels vs W8A16 Quantized submodels
on Qualcomm AI Hub Workbench using ONNXRuntime and Exact Tensor Math.
"""
import os
import sys
import zipfile
import numpy as np
import onnxruntime as ort
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout

ORIGINAL_MODELS = {
    "duration_predictor": "/Users/khoa/.cache/supertonic3/onnx/duration_predictor.onnx",
    "vocoder": "/Users/khoa/.cache/supertonic3/onnx/vocoder.onnx",
    "vector_estimator": "/Users/khoa/.cache/supertonic3/onnx/vector_estimator.onnx",
    "text_encoder": "/Users/khoa/.cache/supertonic3/onnx/text_encoder.onnx",
}

W8A16_DIR = "outputs/qnn_binaries_w8a16"


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute exact Cosine Similarity between two numpy arrays."""
    u = vec1.flatten().astype(np.float64)
    v = vec2.flatten().astype(np.float64)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 1.0 if norm_u == norm_v else 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def compute_snr_db(orig: np.ndarray, quant: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB."""
    u = orig.flatten().astype(np.float64)
    v = quant.flatten().astype(np.float64)
    noise = u - v
    signal_power = np.mean(u ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return 100.0
    return float(10 * np.log10(signal_power / noise_power))


def generate_dummy_inputs(submodel_name: str) -> Dict[str, np.ndarray]:
    """Generate fixed deterministic input tensors for each submodel."""
    np.random.seed(42)
    if submodel_name == "duration_predictor":
        return {
            "text_ids": np.random.randint(1, 100, size=(1, 64), dtype=np.int64),
            "style_dp": np.random.randn(1, 8, 16).astype(np.float32),
            "text_mask": np.ones((1, 1, 64), dtype=np.float32),
        }
    elif submodel_name == "vocoder":
        return {
            "latent": np.random.randn(1, 144, 100).astype(np.float32),
        }
    elif submodel_name == "vector_estimator":
        return {
            "noisy_latent": np.random.randn(1, 144, 100).astype(np.float32),
            "text_emb": np.random.randn(1, 256, 64).astype(np.float32),
            "style_ttl": np.random.randn(1, 50, 256).astype(np.float32),
            "latent_mask": np.ones((1, 1, 100), dtype=np.float32),
            "text_mask": np.ones((1, 1, 64), dtype=np.float32),
            "current_step": np.array([1.0], dtype=np.float32),
            "total_step": np.array([5.0], dtype=np.float32),
        }
    elif submodel_name == "text_encoder":
        return {
            "text_ids": np.random.randint(1, 100, size=(1, 64), dtype=np.int64),
            "style_ttl": np.random.randn(1, 50, 256).astype(np.float32),
            "text_mask": np.ones((1, 1, 64), dtype=np.float32),
        }
    return {}


def run_benchmark():
    _ensure_utf8_stdout()
    print("=" * 80)
    print(" 🧪 GROUND-TRUTH EMPIRICAL VERIFICATION: QUALCOMM AI HUB W8A16 ONNX INFERENCE")
    print("=" * 80)

    results = []
    temp_extract_base = "outputs/extract_qnn_verify"
    os.makedirs(temp_extract_base, exist_ok=True)

    for name, orig_path in ORIGINAL_MODELS.items():
        zip_path = os.path.join(W8A16_DIR, f"{name}_npu_w8a16.bin.onnx.zip")
        print(f"\n[Testing Submodel: {name.upper()}]")
        print(f" • Original Model Path : {orig_path}")
        print(f" • W8A16 Binary Package: {zip_path}")

        if not os.path.exists(orig_path):
            print(f" ❌ Original file missing: {orig_path}")
            continue

        if not os.path.exists(zip_path):
            print(f" ❌ W8A16 zip file missing: {zip_path}")
            continue

        # 1. Run Original FP32 ONNX
        inputs = generate_dummy_inputs(name)
        sess_orig = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
        out_orig = sess_orig.run(None, inputs)[0]

        # 2. Extract and run REAL ONNX from Qualcomm AI Hub Compiled Zip Package
        extract_dir = os.path.join(temp_extract_base, name)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
            onnx_files = [os.path.join(extract_dir, f) for f in zf.namelist() if f.endswith('.onnx')]
            opt_onnx_path = onnx_files[0]
            print(f" • Extracted Ground-Truth ONNX: {opt_onnx_path}")

        sess_quant = ort.InferenceSession(opt_onnx_path, providers=["CPUExecutionProvider"])
        out_w8a16 = sess_quant.run(None, inputs)[0]

        cosine_sim = compute_cosine_similarity(out_orig, out_w8a16)
        snr_db = compute_snr_db(out_orig, out_w8a16)
        mae = float(np.mean(np.abs(out_orig - out_w8a16)))

        print(f" • Measured Cosine Similarity : \033[92m{cosine_sim:.5f}\033[0m")
        print(f" • Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB")
        print(f" • Mean Absolute Error (MAE)  : {mae:.6f}")

        results.append({
            "submodel": name,
            "cosine": cosine_sim,
            "snr_db": snr_db,
            "mae": mae,
            "status": "PASSED (>= 0.930)" if cosine_sim >= 0.930 else "WARNING (< 0.930)"
        })

    print("\n" + "=" * 80)
    print(" 📊 REAL GROUND-TRUTH EMPIRICAL EVALUATION SUMMARY (QUALCOMM ONNX)")
    print("=" * 80)
    print(f" {'Sub-Model':<20} | {'Cosine Sim':<12} | {'SNR (dB)':<10} | {'MAE':<10} | {'Status'}")
    print("-" * 80)
    total_cos = 0
    for r in results:
        print(f" {r['submodel']:<20} | {r['cosine']:<12.5f} | {r['snr_db']:<10.2f} | {r['mae']:<10.6f} | {r['status']}")
        total_cos += r["cosine"]

    avg_cos = total_cos / len(results) if results else 0
    print("=" * 80)
    print(f" 🏆 REAL AVERAGE COSINE SIMILARITY: \033[92m{avg_cos:.5f}\033[0m")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

