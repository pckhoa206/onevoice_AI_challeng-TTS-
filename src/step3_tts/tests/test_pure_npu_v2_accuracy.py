"""Numerical Verification Test for Refactored 100% Pure NPU V2 Models.

Evaluates Cosine Similarity and MAE between original FP32 models and refactored Pure NPU V2 models.
"""
import os
import sys
import time
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

ORIGINAL_DIR = "/Users/khoa/.cache/supertonic3/onnx"
NPU_DIR = "outputs/pure_npu_compliant_onnx_v2"
EMB_DIR = "outputs/embedding_tables"


def test_pure_npu_v2_models():
    _ensure_utf8_stdout()
    print("=" * 85)
    print(" 🧪 TEST: NUMERICAL ACCURACY VERIFICATION OF REFACTORED PURE NPU V2 SUBMODELS")
    print("=" * 85)

    dp_emb = np.load(os.path.join(EMB_DIR, "duration_predictor_char_embedder.npy"))
    te_emb = np.load(os.path.join(EMB_DIR, "text_encoder_char_embedder.npy"))

    sample_text_ids = np.random.randint(1, 100, size=(1, 64), dtype=np.int64)
    sample_text_mask = np.ones((1, 1, 64), dtype=np.float32)
    sample_style_dp = np.random.randn(1, 8, 16).astype(np.float32)
    sample_style_ttl = np.random.randn(1, 50, 256).astype(np.float32)
    sample_latent = np.random.randn(1, 144, 100).astype(np.float32)
    sample_latent_mask = np.ones((1, 1, 100), dtype=np.float32)
    sample_text_emb = np.random.randn(1, 256, 64).astype(np.float32)

    # Convert text_ids to char_emb via table lookup
    dp_char_emb = dp_emb[sample_text_ids].astype(np.float32)
    te_char_emb = te_emb[sample_text_ids].astype(np.float32)

    test_cases = [
        (
            "duration_predictor",
            {"text_ids": sample_text_ids, "style_dp": sample_style_dp, "text_mask": sample_text_mask},
            {"char_emb": dp_char_emb, "style_dp": sample_style_dp, "text_mask": sample_text_mask},
        ),
        (
            "text_encoder",
            {"text_ids": sample_text_ids, "style_ttl": sample_style_ttl, "text_mask": sample_text_mask},
            {"char_emb": te_char_emb, "style_ttl": sample_style_ttl, "text_mask": sample_text_mask},
        ),
        (
            "vector_estimator",
            {
                "noisy_latent": sample_latent,
                "text_emb": sample_text_emb,
                "style_ttl": sample_style_ttl,
                "latent_mask": sample_latent_mask,
                "text_mask": sample_text_mask,
                "current_step": np.array([1.0], dtype=np.float32),
                "total_step": np.array([5.0], dtype=np.float32),
            },
            {
                "noisy_latent": sample_latent,
                "text_emb": sample_text_emb,
                "style_ttl": sample_style_ttl,
                "latent_mask": sample_latent_mask,
                "text_mask": sample_text_mask,
                "current_step": np.array([1.0], dtype=np.float32),
                "total_step": np.array([5.0], dtype=np.float32),
            },
        ),
        (
            "vocoder",
            {"latent": sample_latent},
            {"latent": sample_latent},
        ),
    ]

    all_passed = True
    for name, orig_inputs, npu_inputs in test_cases:
        orig_path = os.path.join(ORIGINAL_DIR, f"{name}.onnx")
        npu_path = os.path.join(NPU_DIR, f"{name}_pure_npu.onnx")

        if not os.path.exists(npu_path):
            print(f" • [{name:<18}]: ❌ File missing: '{npu_path}'")
            all_passed = False
            continue

        orig_sess = ort.InferenceSession(orig_path, providers=["CPUExecutionProvider"])
        npu_sess = ort.InferenceSession(npu_path, providers=["CPUExecutionProvider"])

        t0 = time.time()
        orig_out = orig_sess.run(None, orig_inputs)[0]
        t_orig = (time.time() - t0) * 1000

        t0 = time.time()
        npu_out = npu_sess.run(None, npu_inputs)[0]
        t_npu = (time.time() - t0) * 1000

        dot_prod = np.dot(orig_out.flatten(), npu_out.flatten())
        norm_orig = np.linalg.norm(orig_out)
        norm_npu = np.linalg.norm(npu_out)
        cosine_sim = dot_prod / (norm_orig * norm_npu) if norm_orig > 0 and norm_npu > 0 else 0.0
        mae = float(np.mean(np.abs(orig_out - npu_out)))

        # Tanh FastGeLU preserves ~0.999+ fidelity
        status = "✅ PASSED" if cosine_sim >= 0.98 else "❌ FAILED"
        print(
            f" • [{name:<18}]: Cosine Sim = {cosine_sim:.6f} | MAE = {mae:.6f} | "
            f"Orig Latency = {t_orig:5.1f}ms | NPU Model Latency = {t_npu:5.1f}ms | {status}"
        )

        if cosine_sim < 0.98:
            all_passed = False

    print("=" * 85)
    return all_passed


def main():
    passed = test_pure_npu_v2_models()
    if passed:
        print("\n 🎉 ALL 4 REFACTORED PURE NPU V2 SUBMODELS PASSED ACCURACY VERIFICATION!")
    else:
        print("\n ⚠️ SOME SUBMODELS FAILED VERIFICATION.")


if __name__ == "__main__":
    main()
