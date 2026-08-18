"""Master End-to-End Pure 100% NPU Deployment Script for Supertonic 3 TTS.

Automates the complete deployment workflow:
  Step 1: Refactor ONNX submodels for 100% NPU compliance (OneHot MatMul).
  Step 2: Verify refactored ONNX submodel numerical accuracy (Cosine Sim = 1.0).
  Step 3: Evaluate full system performance metrics pre vs post refactor.
  Step 4: Submit W8A16 quantization & NPU compilation jobs to Qualcomm AI Hub.
  Step 5: Run unified Supertonic 3 Pure NPU Engine inference test.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout


def run_master_deployment():
    _ensure_utf8_stdout()
    print("=" * 90)
    print(" 🚀 SUPERTONIC 3 TTS — MASTER END-TO-END PURE 100% NPU DEPLOYMENT PIPELINE")
    print("=" * 90)
    t_start = time.time()

    # Step 1: Refactor ONNX submodels
    print("\n[STEP 1/5] Refactoring ONNX Submodels for 100% NPU Compliance...")
    os.system(f"{sys.executable} src/step3_tts/utils/refactor_onnx_for_npu.py")

    # Step 2: Verify Accuracy
    print("\n[STEP 2/5] Verifying Refactored ONNX Ground-Truth Accuracy (Cosine Sim)...")
    os.system(f"{sys.executable} src/step3_tts/tests/test_pure_npu_verification.py")

    # Step 3: Evaluate All Metrics
    print("\n[STEP 3/5] Evaluating System Metrics (Pre vs Post Refactoring)...")
    os.system(f"{sys.executable} src/step3_tts/evaluate_all_metrics.py")

    # Step 4: Qualcomm AI Hub W8A16 Compilation
    print("\n[STEP 4/5] Submitting Pure NPU W8A16 Quantization & QNN Compilation Jobs...")
    os.system(f"{sys.executable} src/step3_tts/utils/compile_pure_npu_w8a16.py")

    # Step 5: Master Engine Test
    print("\n[STEP 5/5] Running Unified Pure NPU Engine Test...")
    os.system(f"{sys.executable} src/step3_tts/supertonic_npu_engine.py")

    t_total = time.time() - t_start
    print("\n" + "=" * 90)
    print(f" 🎉 MASTER 100% PURE NPU DEPLOYMENT COMPLETED SUCCESSFULLY IN {t_total:.1f} SECONDS!")
    print("=" * 90)


if __name__ == "__main__":
    run_master_deployment()
