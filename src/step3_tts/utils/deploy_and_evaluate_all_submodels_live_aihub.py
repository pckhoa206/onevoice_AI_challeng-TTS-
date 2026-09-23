"""Deploy All 4 Submodels to Qualcomm AI Hub, Execute on Live Hardware, Download Outputs, and Evaluate Accuracy.

Target Hardware: Dragonwing IQ-9075 EVK (Qualcomm QCS9075 SoC)
Submodels:
  1. Duration Predictor (mq3w6plrm)
  2. Text Encoder (mqej7z45m)
  3. Vector Estimator (mn7gx0y3n)
  4. Vocoder (mmdj34w3m)
"""
import os
import sys
import time
import json
from typing import Dict, Any, Tuple
import numpy as np
import soundfile as sf
import onnxruntime as ort
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
OUTPUT_DIR = os.path.join(ROOT, "outputs", "aihub_live_all_submodels")
MODELS_DIR = os.path.join(ROOT, "outputs", "pure_npu_dynamic")

TARGET_DEVICE_NAME = "Dragonwing IQ-9075 EVK"
API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN")
if not API_TOKEN:
    raise SystemExit("Set QAI_HUB_API_TOKEN before running this script.")

# Compiled Target Model IDs on Qualcomm AI Hub for Dragonwing IQ-9075 EVK
TARGET_MODELS = {
    "duration_predictor": "mq3w6plrm",
    "text_encoder": "mqej7z45m",
    "vector_estimator": "mn7gx0y3n",
    "vocoder": "mmdj34w3m",
}

# Local ONNX model filenames for Ground Truth reference
LOCAL_MODELS = {
    "duration_predictor": os.path.join(MODELS_DIR, "duration_predictor_npu.onnx"),
    "text_encoder": os.path.join(MODELS_DIR, "text_encoder_npu.onnx"),
    "vector_estimator": os.path.join(MODELS_DIR, "vector_estimator_npu.onnx"),
    "vocoder": os.path.join(MODELS_DIR, "vocoder_npu.onnx"),
}


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive numerical accuracy metrics between ground truth and hardware output."""
    gt_f = gt.flatten().astype(np.float64)
    pred_f = pred.flatten().astype(np.float64)

    # Cosine Similarity
    norm_gt = np.linalg.norm(gt_f)
    norm_pred = np.linalg.norm(pred_f)
    if norm_gt > 1e-12 and norm_pred > 1e-12:
        cosine_sim = float(np.dot(gt_f, pred_f) / (norm_gt * norm_pred))
    else:
        cosine_sim = 1.0 if np.allclose(gt_f, pred_f) else 0.0

    # Absolute errors
    abs_diff = np.abs(gt_f - pred_f)
    mae = float(np.mean(abs_diff))
    max_ae = float(np.max(abs_diff))
    rmse = float(np.sqrt(np.mean(abs_diff ** 2)))

    # Signal-to-Noise Ratio (SNR in dB)
    noise_energy = np.sum((gt_f - pred_f) ** 2)
    signal_energy = np.sum(gt_f ** 2)
    if noise_energy > 1e-15 and signal_energy > 1e-15:
        snr_db = float(10.0 * np.log10(signal_energy / noise_energy))
    elif noise_energy <= 1e-15:
        snr_db = 100.0  # essentially infinite SNR / perfect match
    else:
        snr_db = 0.0

    return {
        "cosine_similarity": round(cosine_sim, 6),
        "mae": round(mae, 6),
        "max_error": round(max_ae, 6),
        "rmse": round(rmse, 6),
        "snr_db": round(snr_db, 2),
    }


def prepare_inputs() -> Dict[str, Dict[str, np.ndarray]]:
    """Prepares valid, deterministic input tensors matching the exact compiled shapes."""
    np.random.seed(42)

    # Submodel 1: Duration Predictor
    dp_inputs = {
        "text_ids": np.random.randint(1, 100, size=(1, 64), dtype=np.int64),
        "style_dp": np.random.randn(1, 8, 16).astype(np.float32),
        "text_mask": np.ones((1, 1, 64), dtype=np.float32),
    }

    # Submodel 2: Text Encoder
    te_inputs = {
        "text_ids": np.random.randint(1, 100, size=(1, 64), dtype=np.int64),
        "style_ttl": np.random.randn(1, 50, 256).astype(np.float32),
        "text_mask": np.ones((1, 1, 64), dtype=np.float32),
    }

    # Submodel 3: Vector Estimator
    ve_inputs = {
        "noisy_latent": np.random.randn(1, 144, 100).astype(np.float32),
        "text_emb": np.random.randn(1, 256, 64).astype(np.float32),
        "style_ttl": np.random.randn(1, 50, 256).astype(np.float32),
        "latent_mask": np.ones((1, 1, 100), dtype=np.float32),
        "text_mask": np.ones((1, 1, 64), dtype=np.float32),
        "current_step": np.array([1.0], dtype=np.float32),
        "total_step": np.array([5.0], dtype=np.float32),
    }

    # Submodel 4: Vocoder (using realistic latent distribution)
    voc_inputs = {
        "latent": np.random.randn(1, 144, 100).astype(np.float32) * 0.5,
    }

    return {
        "duration_predictor": dp_inputs,
        "text_encoder": te_inputs,
        "vector_estimator": ve_inputs,
        "vocoder": voc_inputs,
    }


def compute_local_ground_truth(inputs_dict: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Runs all 4 models locally using ONNX Runtime FP32 to obtain baseline Ground Truth."""
    print("\n" + "=" * 85)
    print(" 🖥️  COMPUTING LOCAL REFERENCE GROUND TRUTH (ONNX Runtime FP32)")
    print("=" * 85)
    gt_outputs = {}

    for name, model_path in LOCAL_MODELS.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing local model: {model_path}")
        
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inps = inputs_dict[name]
        t0 = time.time()
        out = sess.run(None, inps)[0]
        dur_ms = (time.time() - t0) * 1000.0
        gt_outputs[name] = np.asarray(out)
        print(f" • [Local Baseline] {name:<18}: Output Shape = {str(out.shape):<18} | Time = {dur_ms:6.1f} ms")

    return gt_outputs


def deploy_and_evaluate_all():
    _ensure_utf8_stdout()
    print("=" * 85)
    print(" 🚀 QUALCOMM AI HUB — LIVE HARDWARE DEPLOYMENT & ACCURACY EVALUATION")
    print(f" • Target Hardware : {TARGET_DEVICE_NAME} (Qualcomm QCS9075 SoC / Hexagon HTP NPU)")
    print(f" • Output Directory: {OUTPUT_DIR}")
    print("=" * 85)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = hub.Client(config=hub.ClientConfig(api_token=API_TOKEN))

    # 1. Connect to Target Device
    devices = client.get_devices(TARGET_DEVICE_NAME)
    if not devices:
        raise RuntimeError(f"Target device '{TARGET_DEVICE_NAME}' not found in Qualcomm AI Hub cloud pool.")
    device = devices[0]
    print(f"✅ Connected to Target Device: {device.name} (OS: {device.os})")

    # 2. Prepare Inputs and compute local ground truth
    all_inputs = prepare_inputs()
    gt_outputs = compute_local_ground_truth(all_inputs)

    # 3. Submit Live Hardware Inference Jobs for all 4 submodels
    print("\n" + "=" * 85)
    print(" 📡 SUBMITTING LIVE HARDWARE INFERENCE JOBS TO QUALCOMM AI HUB")
    print("=" * 85)

    submitted_jobs = {}
    for name, model_id in TARGET_MODELS.items():
        print(f"\n[Submodel: {name}]")
        print(f" • Target Model ID : {model_id}")
        target_model = client.get_model(model_id)

        # Wrap numpy arrays into input format required by AI Hub: Dict[str, List[np.ndarray]]
        formatted_inputs = {k: [v] for k, v in all_inputs[name].items()}

        job = client.submit_inference_job(
            model=target_model,
            device=device,
            inputs=formatted_inputs,
            name=f"[EVAL_STAGE2] Supertonic3_{name}_Live_Hardware",
        )
        print(f" • Inference Job ID: {job.job_id}")
        print(f" • Dashboard URL   : {job.url}")
        submitted_jobs[name] = job

    # 4. Wait for All Jobs to Complete on Physical Hardware
    print("\n" + "=" * 85)
    print(" ⏳ WAITING FOR HARDWARE EXECUTION ON QUALCOMM DRAGONWING IQ-9075 EVK")
    print("=" * 85)

    npu_outputs = {}
    job_statuses = {}

    for name, job in submitted_jobs.items():
        print(f"\n• Waiting for [{name}] (Job ID: {job.job_id})...")
        job.wait()
        st = job.get_status().code
        job_statuses[name] = {
            "job_id": job.job_id,
            "status": st,
            "url": job.url,
        }
        print(f"  Status: {st} | Dashboard: {job.url}")

        if st == "SUCCESS":
            outs = job.download_output_data()
            first_key = list(outs.keys())[0]
            arr = np.asarray(outs[first_key][0]).squeeze().astype(np.float32)
            npu_outputs[name] = arr
            
            # Save raw hardware tensor
            out_npy_path = os.path.join(OUTPUT_DIR, f"live_hardware_{name}_output.npy")
            np.save(out_npy_path, arr)
            print(f"  ✅ Saved Raw Tensor Output: {out_npy_path} (Shape: {arr.shape})")

            # Also save ground truth tensor for direct side-by-side inspection
            gt_npy_path = os.path.join(OUTPUT_DIR, f"ground_truth_{name}_output.npy")
            np.save(gt_npy_path, gt_outputs[name])

            # If vocoder, save WAV audio
            if name == "vocoder":
                wav_path = os.path.join(OUTPUT_DIR, "live_hardware_vocoder_synthesis.wav")
                peak = np.max(np.abs(arr))
                norm_wav = (arr / peak * 0.95) if peak > 0 else arr
                sf.write(wav_path, norm_wav, 24000)
                print(f"  🔊 Exported Live Hardware Audio: {wav_path} (Duration: {len(arr)/24000:.2f}s @ 24kHz)")
        else:
            print(f"  ❌ FAILED on hardware! Status: {st}")

    # 5. Evaluate Accuracy
    print("\n" + "=" * 85)
    print(" 📊 COMPREHENSIVE NUMERICAL ACCURACY EVALUATION (NPU HARDWARE vs LOCAL GT)")
    print("=" * 85)

    evaluation_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_device": TARGET_DEVICE_NAME,
        "chipset": "Qualcomm QCS9075 (Hexagon HTP NPU)",
        "jobs": job_statuses,
        "metrics": {},
    }

    header = f"{'Submodel Name':<20} | {'Cosine Sim':<11} | {'MAE':<10} | {'Max Error':<11} | {'SNR (dB)':<9} | {'Status'}"
    print(header)
    print("-" * len(header))

    for name in TARGET_MODELS.keys():
        if name in npu_outputs and name in gt_outputs:
            gt_arr = gt_outputs[name].squeeze()
            npu_arr = npu_outputs[name].squeeze()

            # Ensure matching shapes
            min_len = min(gt_arr.size, npu_arr.size)
            metrics = compute_metrics(gt_arr.flatten()[:min_len], npu_arr.flatten()[:min_len])
            evaluation_report["metrics"][name] = metrics

            status_str = "✅ EXCELLENT" if metrics["cosine_similarity"] >= 0.99 else ("⚠️ ACCEPTABLE" if metrics["cosine_similarity"] >= 0.90 else "❌ MISMATCH")
            print(
                f"{name:<20} | "
                f"{metrics['cosine_similarity']:<11.6f} | "
                f"{metrics['mae']:<10.6f} | "
                f"{metrics['max_error']:<11.6f} | "
                f"{metrics['snr_db']:<9.2f} | "
                f"{status_str}"
            )
        else:
            print(f"{name:<20} | {'N/A':<11} | {'N/A':<10} | {'N/A':<11} | {'N/A':<9} | ❌ FAILED")

    print("-" * len(header))

    # Save JSON report
    report_json_path = os.path.join(OUTPUT_DIR, "accuracy_evaluation_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Full Accuracy Evaluation Report saved to:\n   {report_json_path}")
    print("=" * 85)
    print(" 🎉 DEPLOYMENT, LIVE HARDWARE EXECUTION, AND EVALUATION COMPLETE!")
    print("=" * 85)


if __name__ == "__main__":
    deploy_and_evaluate_all()
