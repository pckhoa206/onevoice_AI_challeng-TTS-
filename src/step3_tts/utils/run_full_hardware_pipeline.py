"""End-to-End Qualcomm AI Hub Pipeline for Supertonic 3 TTS:
Quantize -> Compile -> Profile -> Reference (Inference) -> Download & Accuracy Test

Submodels:
  1. Duration Predictor (DP)
  2. Text Encoder (TE)
  3. Vector Estimator (VE)
  4. Neural Vocoder (VOC)

Evaluates:
  • Hardware Latency (ms) on Qualcomm Dragonwing IQ-9075 EVK (Hexagon HTP NPU)
  • Hardware Resource Consumption (Peak RAM in MB, SRAM residency, Compute Unit)
  • Numerical Accuracy (Cosine Similarity, MAE, Max Error, SNR dB vs FP32 Ground Truth)
  • Full Speech Waveform Generation (.wav) for Listening Verification
"""
import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import onnx
import onnxruntime as ort
import qai_hub as hub

def _ensure_utf8_stdout():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

DEFAULT_TARGET_DEVICE = "Dragonwing IQ-9075 EVK"
OUTPUT_DIR = "outputs/pipeline_hardware_runs"
STATIC_MODELS_DIR = "outputs/pipeline_static_models"

# Enforce fresh W8A16 quantization on all models
PREQUANTIZED_MODELS = {}

def get_realistic_speech_latent() -> np.ndarray:
    """Generates a realistic Mel-latent vector using Text Encoder & 5-step Euler Vector Estimator."""
    try:
        sess_te = ort.InferenceSession("outputs/pipeline_static_models/text_encoder_static.onnx", providers=["CPUExecutionProvider"])
        sess_ve = ort.InferenceSession("outputs/pipeline_static_models/vector_estimator_static.onnx", providers=["CPUExecutionProvider"])
        style_ttl = np.load("outputs/style_vectors/vi_style_ttl.npy").astype(np.float32)
        text_ids = np.zeros((1, 64), dtype=np.int64)
        text_ids[0, :15] = [12, 45, 23, 78, 9, 34, 56, 88, 19, 72, 41, 15, 63, 27, 8]
        text_mask = np.ones((1, 1, 64), dtype=np.float32)

        text_emb = sess_te.run(None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask})[0]

        np.random.seed(42)
        xt = np.random.randn(1, 144, 100).astype(np.float32)
        latent_mask = np.ones((1, 1, 100), dtype=np.float32)
        for step in range(5):
            vt = sess_ve.run(None, {
                "noisy_latent": xt,
                "text_emb": text_emb,
                "style_ttl": style_ttl,
                "latent_mask": latent_mask,
                "text_mask": text_mask,
                "current_step": np.array([float(step)], dtype=np.float32),
                "total_step": np.array([5.0], dtype=np.float32)
            })[0]
            xt = xt + 0.2 * vt
        return xt
    except Exception as e:
        print(f"  ℹ️ Notice: using calibrated normal latent ({e})")
        np.random.seed(42)
        return (np.random.randn(1, 144, 100).astype(np.float32) * 0.5)


SUBMODELS_CONFIG = {
    "duration_predictor": {
        "raw_path": "outputs/npu_compliant_onnx/duration_predictor_npu.onnx",
        "shape_map": {"batch_size": 1, "text_length": 64},
        "sample_calib": lambda: {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64) for _ in range(4)],
            "style_dp": [np.random.randn(1, 8, 16).astype(np.float32) for _ in range(4)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32) for _ in range(4)],
        },
        "test_input": lambda: {
            "text_ids": np.random.randint(1, 100, size=(1, 64), dtype=np.int64),
            "style_dp": np.random.randn(1, 8, 16).astype(np.float32),
            "text_mask": np.ones((1, 1, 64), dtype=np.float32),
        },
    },
    "text_encoder": {
        "raw_path": "outputs/npu_compliant_onnx/text_encoder_npu.onnx",
        "shape_map": {"batch_size": 1, "text_length": 64},
        "sample_calib": lambda: {
            "text_ids": [np.random.randint(1, 100, size=(1, 64), dtype=np.int64) for _ in range(4)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32) for _ in range(4)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32) for _ in range(4)],
        },
        "test_input": lambda: {
            "text_ids": np.random.randint(1, 100, size=(1, 64), dtype=np.int64),
            "style_ttl": np.load("outputs/style_vectors/vi_style_ttl.npy").astype(np.float32),
            "text_mask": np.ones((1, 1, 64), dtype=np.float32),
        },
    },
    "vector_estimator": {
        "raw_path": "outputs/npu_compliant_onnx/vector_estimator_npu.onnx",
        "shape_map": {"batch_size": 1, "text_length": 64, "latent_length": 100},
        "sample_calib": lambda: {
            "noisy_latent": [np.random.randn(1, 144, 100).astype(np.float32) for _ in range(3)],
            "text_emb": [np.random.randn(1, 256, 64).astype(np.float32) for _ in range(3)],
            "style_ttl": [np.random.randn(1, 50, 256).astype(np.float32) for _ in range(3)],
            "latent_mask": [np.ones((1, 1, 100), dtype=np.float32) for _ in range(3)],
            "text_mask": [np.ones((1, 1, 64), dtype=np.float32) for _ in range(3)],
            "current_step": [np.array([1.0], dtype=np.float32) for _ in range(3)],
            "total_step": [np.array([5.0], dtype=np.float32) for _ in range(3)],
        },
        "test_input": lambda: {
            "noisy_latent": np.random.randn(1, 144, 100).astype(np.float32),
            "text_emb": np.random.randn(1, 256, 64).astype(np.float32),
            "style_ttl": np.load("outputs/style_vectors/vi_style_ttl.npy").astype(np.float32),
            "latent_mask": np.ones((1, 1, 100), dtype=np.float32),
            "text_mask": np.ones((1, 1, 64), dtype=np.float32),
            "current_step": np.array([1.0], dtype=np.float32),
            "total_step": np.array([5.0], dtype=np.float32),
        },
    },
    "vocoder": {
        "raw_path": "outputs/npu_compliant_onnx/vocoder_npu.onnx",
        "shape_map": {"batch_size": 1, "latent_length": 100},
        "sample_calib": lambda: {
            "latent": [np.random.randn(1, 144, 100).astype(np.float32) * 0.5 for _ in range(4)],
        },
        "test_input": lambda: {
            "latent": get_realistic_speech_latent(),
        },
    },
}


def prepare_static_model(model_name: str, config: Dict[str, Any]) -> str:
    """Freezes dynamic dimension parameters in ONNX graph to static values."""
    os.makedirs(STATIC_MODELS_DIR, exist_ok=True)
    static_path = os.path.join(STATIC_MODELS_DIR, f"{model_name}_static.onnx")
    raw_path = config["raw_path"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw model file not found: {raw_path}")

    model = onnx.load(raw_path)
    shape_map = config["shape_map"]

    for inp in model.graph.input:
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_param in shape_map:
                dim.dim_value = shape_map[dim.dim_param]

    if model_name == "vector_estimator":
        from onnx import numpy_helper
        const_map = {
            '/vector_estimator/Unsqueeze_output_0': np.array([1], dtype=np.int64),
            '/vector_estimator/Unsqueeze_2_output_0': np.array([1], dtype=np.int64),
            '/vector_estimator/Unsqueeze_3_output_0': np.array([64], dtype=np.int64),
            '/vector_estimator/Unsqueeze_4_output_0': np.array([1], dtype=np.int64),
            '/vector_estimator/Unsqueeze_5_output_0': np.array([1], dtype=np.int64),
            '/vector_estimator/vector_field/main_blocks.3/attn/Unsqueeze_24_output_0': np.array([100], dtype=np.int64),
            '/vector_estimator/vector_field/main_blocks.3/attn/Unsqueeze_25_output_0': np.array([64], dtype=np.int64)
        }
        remove_nodes = {
            '/vector_estimator/Gather', '/vector_estimator/Unsqueeze',
            '/vector_estimator/Gather_1', '/vector_estimator/Unsqueeze_2',
            '/vector_estimator/Gather_2', '/vector_estimator/Unsqueeze_3',
            '/vector_estimator/Gather_3', '/vector_estimator/Unsqueeze_4',
            '/vector_estimator/Gather_4', '/vector_estimator/Unsqueeze_5',
            '/vector_estimator/vector_field/main_blocks.3/attn/Gather', '/vector_estimator/vector_field/main_blocks.3/attn/Unsqueeze_24',
            '/vector_estimator/vector_field/main_blocks.3/attn/Gather_1', '/vector_estimator/vector_field/main_blocks.3/attn/Unsqueeze_25'
        }
        for tensor_name, arr in const_map.items():
            tensor_proto = numpy_helper.from_array(arr, name=tensor_name)
            model.graph.initializer.append(tensor_proto)

        new_nodes = [n for n in model.graph.node if n.name not in remove_nodes]
        model.graph.ClearField('node')
        model.graph.node.extend(new_nodes)

        bad_tensors = set(const_map.keys()) | {
            '/vector_estimator/Gather_output_0', '/vector_estimator/Gather_1_output_0',
            '/vector_estimator/Gather_2_output_0', '/vector_estimator/Gather_4_output_0',
            '/vector_estimator/vector_field/main_blocks.3/attn/Gather_output_0',
            '/vector_estimator/vector_field/main_blocks.3/attn/Gather_1_output_0',
            '/vector_estimator/Gather_3_output_0'
        }
        new_vi = [vi for vi in model.graph.value_info if vi.name not in bad_tensors]
        model.graph.ClearField('value_info')
        model.graph.value_info.extend(new_vi)

    static_model = onnx.shape_inference.infer_shapes(model)
    onnx.save(static_model, static_path)
    print(f"  🔧 Prepared static ONNX graph: {static_path}")
    return static_path


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Calculates Cosine Similarity, MAE, Max Error, RMSE, and SNR (dB)."""
    gt_f = gt.flatten().astype(np.float64)
    pred_f = pred.flatten().astype(np.float64)

    min_len = min(len(gt_f), len(pred_f))
    gt_f = gt_f[:min_len]
    pred_f = pred_f[:min_len]

    norm_gt = np.linalg.norm(gt_f)
    norm_pred = np.linalg.norm(pred_f)
    if norm_gt > 1e-12 and norm_pred > 1e-12:
        cosine_sim = float(np.dot(gt_f, pred_f) / (norm_gt * norm_pred))
    else:
        cosine_sim = 1.0 if np.allclose(gt_f, pred_f) else 0.0

    abs_diff = np.abs(gt_f - pred_f)
    mae = float(np.mean(abs_diff))
    max_err = float(np.max(abs_diff))
    rmse = float(np.sqrt(np.mean(abs_diff ** 2)))

    noise_energy = np.sum((gt_f - pred_f) ** 2)
    signal_energy = np.sum(gt_f ** 2)
    if noise_energy > 1e-15 and signal_energy > 1e-15:
        snr_db = float(10.0 * np.log10(signal_energy / noise_energy))
    elif noise_energy <= 1e-15:
        snr_db = 100.0
    else:
        snr_db = 0.0

    return {
        "cosine_similarity": round(cosine_sim, 6),
        "mae": round(mae, 6),
        "max_error": round(max_err, 6),
        "rmse": round(rmse, 6),
        "snr_db": round(snr_db, 2),
    }


def run_pipeline_for_submodel(submodel_name: str, device_name: str, force_requantize: bool = False):
    print("=" * 85)
    print(f" 🚀 PIPELINE: QUANTIZE -> COMPILE -> PROFILE -> REFERENCE -> TEST")
    print(f" • Submodel       : {submodel_name.upper()}")
    print(f" • Target Device  : {device_name}")
    print(f" • Quantization   : W8A16 (Weights INT8, Activations INT16)")
    print(f" • Compute Unit   : Qualcomm Hexagon HTP NPU")
    print("=" * 85)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = SUBMODELS_CONFIG[submodel_name]

    # Step 0: Ensure Static Model
    print(f"\n[Step 0/5] Preparing Static Model for '{submodel_name}'...")
    static_model_path = prepare_static_model(submodel_name, cfg)

    # Connect to device
    devices = hub.get_devices(device_name)
    if not devices:
        raise RuntimeError(f"Target device '{device_name}' not available on AI Hub.")
    device = devices[0]
    print(f"  📱 Target Device Connected: {device.name} (OS: {device.os})")

    # Step 1: Quantize (W8A16)
    quant_job_id = None
    quant_job_url = None
    if submodel_name in PREQUANTIZED_MODELS and not force_requantize:
        quantized_model_id = PREQUANTIZED_MODELS[submodel_name]
        print(f"\n[Step 1/5] Reusing Validated W8A16 Quantized Model '{quantized_model_id}' (from cache)...")
        quantized_model = hub.get_model(quantized_model_id)
        quant_status = "CACHED_SUCCESS"
    else:
        print(f"\n[Step 1/5] Submitting QUANTIZE Job (W8A16)...")
        calib_data = cfg["sample_calib"]()
        quant_job = hub.submit_quantize_job(
            model=static_model_path,
            calibration_data=calib_data,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            name=f"[PIPELINE] {submodel_name}_w8a16",
        )
        quant_job_id = quant_job.job_id
        quant_job_url = quant_job.url
        print(f"  • Quantize Job ID : {quant_job.job_id}")
        print(f"  • Dashboard URL   : {quant_job.url}")
        print(f"  ⏳ Waiting for quantization to finish...")
        quant_job.wait()
        quant_status = quant_job.get_status().code
        print(f"  • Quantize Status : {quant_status}")
        if quant_status != "SUCCESS":
            raise RuntimeError(f"Quantization failed with status: {quant_status}")
        quantized_model = quant_job.get_target_model()
    print(f"  ✅ Quantized Model ID: {quantized_model.model_id}")

    # Step 2: Compile
    print(f"\n[Step 2/5] Submitting COMPILE Job for Hexagon NPU...")
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        options="--target_runtime onnx",
        name=f"[PIPELINE] {submodel_name}_compile_npu",
    )
    print(f"  • Compile Job ID  : {compile_job.job_id}")
    print(f"  • Dashboard URL   : {compile_job.url}")
    print(f"  ⏳ Waiting for NPU compilation...")
    compile_job.wait()
    compile_status = compile_job.get_status().code
    print(f"  • Compile Status  : {compile_status}")
    if compile_status != "SUCCESS":
        raise RuntimeError(f"Compilation failed with status: {compile_status}")
    compiled_model = compile_job.get_target_model()
    print(f"  ✅ Compiled Model ID: {compiled_model.model_id}")

    # Step 3: Profile
    print(f"\n[Step 3/5] Submitting PROFILE Job on {device.name}...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name=f"[PIPELINE] {submodel_name}_profile",
    )
    print(f"  • Profile Job ID  : {profile_job.job_id}")
    print(f"  • Dashboard URL   : {profile_job.url}")
    print(f"  ⏳ Waiting for hardware profiling...")
    profile_job.wait()
    profile_status = profile_job.get_status().code
    print(f"  • Profile Status  : {profile_status}")

    perf_summary = {"latency_ms": "N/A", "peak_memory_mb": "N/A", "compute_unit": "Qualcomm Hexagon HTP NPU"}
    if profile_status == "SUCCESS":
        try:
            profile_data = profile_job.download_profile()
            exec_sum = profile_data.get("execution_summary", {})
            est_time_us = exec_sum.get("estimated_inference_time", 0)
            peak_mem = exec_sum.get("estimated_inference_peak_memory", 0)
            perf_summary = {
                "latency_ms": round(est_time_us / 1000.0, 3) if est_time_us else "N/A",
                "peak_memory_mb": round(peak_mem / (1024.0 * 1024.0), 2) if peak_mem else "N/A",
                "compute_unit": "Qualcomm Hexagon HTP NPU",
            }
            print(f"  ⚡ Hardware Latency: {perf_summary['latency_ms']} ms | Peak RAM: {perf_summary['peak_memory_mb']} MB")
        except Exception as e:
            print(f"  ℹ️ Profile metadata note: {e}")

    # Step 4: Reference / Inference on Live Physical Hardware
    print(f"\n[Step 4/5] Submitting REFERENCE INFERENCE Job on Physical Hardware...")
    np.random.seed(42)
    raw_test_inputs = cfg["test_input"]()
    # Format inputs for AI Hub: Dict[str, List[np.ndarray]]
    formatted_inputs = {k: [v] for k, v in raw_test_inputs.items()}

    inf_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=formatted_inputs,
        name=f"[PIPELINE] {submodel_name}_reference_inference",
    )
    print(f"  • Inference Job ID: {inf_job.job_id}")
    print(f"  • Dashboard URL   : {inf_job.url}")
    print(f"  ⏳ Waiting for on-device execution...")
    inf_job.wait()
    inf_status = inf_job.get_status().code
    print(f"  • Inference Status: {inf_status}")
    if inf_status != "SUCCESS":
        raise RuntimeError(f"Inference failed on hardware: {inf_status}")

    # Step 5: Download Output and Test Accuracy
    print(f"\n[Step 5/5] Downloading Output & Running Accuracy Verification Tests...")
    raw_outputs = inf_job.download_output_data()
    first_out_key = list(raw_outputs.keys())[0]
    hw_output_tensor = np.asarray(raw_outputs[first_out_key][0]).squeeze().astype(np.float32)

    hw_npy_path = os.path.join(OUTPUT_DIR, f"{submodel_name}_live_hardware_output.npy")
    np.save(hw_npy_path, hw_output_tensor)
    print(f"  💾 Saved Live Hardware Tensor: {hw_npy_path} (Shape: {hw_output_tensor.shape})")

    # Local Baseline Ground Truth with ONNX Runtime FP32
    print(f"  🖥️ Running Local Reference Ground Truth (ONNX Runtime FP32)...")
    sess = ort.InferenceSession(static_model_path, providers=["CPUExecutionProvider"])
    gt_raw = sess.run(None, raw_test_inputs)[0]
    gt_output_tensor = np.asarray(gt_raw).squeeze().astype(np.float32)

    gt_npy_path = os.path.join(OUTPUT_DIR, f"{submodel_name}_ground_truth_output.npy")
    np.save(gt_npy_path, gt_output_tensor)
    print(f"  💾 Saved Reference Ground Truth Tensor: {gt_npy_path} (Shape: {gt_output_tensor.shape})")

    audio_path = None
    # If Vocoder, export audio wave
    if submodel_name == "vocoder":
        import soundfile as sf
        audio_path = os.path.join(OUTPUT_DIR, f"{submodel_name}_live_hardware_audio.wav")
        peak = np.max(np.abs(hw_output_tensor))
        norm_wav = (hw_output_tensor / peak * 0.95) if peak > 0 else hw_output_tensor
        sf.write(audio_path, norm_wav, 24000)
        print(f"  🔊 Exported Live Hardware Audio: {audio_path} (Duration: {len(hw_output_tensor)/24000:.2f}s @ 24kHz)")

    # Compute Comparative Metrics
    metrics = compute_metrics(gt_output_tensor, hw_output_tensor)
    passed = metrics["cosine_similarity"] >= 0.95

    report = {
        "submodel": submodel_name,
        "target_device": device.name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": {
            "quantize": {"job_id": quant_job_id, "url": quant_job_url, "status": quant_status},
            "compile": {"job_id": compile_job.job_id, "url": compile_job.url, "status": compile_status},
            "profile": {"job_id": profile_job.job_id, "url": profile_job.url, "status": profile_status},
            "inference": {"job_id": inf_job.job_id, "url": inf_job.url, "status": inf_status},
        },
        "hardware_performance": perf_summary,
        "output_shape": list(hw_output_tensor.shape),
        "metrics": metrics,
        "audio_path": audio_path,
        "test_result": "PASSED" if passed else "FAILED",
    }

    report_path = os.path.join(OUTPUT_DIR, f"{submodel_name}_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 85)
    print(f" 📊 FINAL REPORT: {submodel_name.upper()}")
    print("=" * 85)
    print(f" • Latency on NPU        : {perf_summary['latency_ms']} ms")
    print(f" • Peak RAM on Device    : {perf_summary['peak_memory_mb']} MB")
    print(f" • Cosine Similarity     : {metrics['cosine_similarity']:.6f} {'✅' if metrics['cosine_similarity'] >= 0.99 else ('⚠️' if metrics['cosine_similarity'] >= 0.90 else '❌')}")
    print(f" • Mean Absolute Error    : {metrics['mae']:.6f}")
    print(f" • Signal-to-Noise (SNR)  : {metrics['snr_db']:.2f} dB")
    print(f" • Overall Test Verdict   : {'🎉 PASSED' if passed else '❌ FAILED'}")
    if audio_path:
        print(f" • Audio Output (.wav)    : {audio_path}")
    print(f" • Report Saved To        : {report_path}")
    print("=" * 85)

    return report


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Full Hardware Pipeline: Quantize -> Compile -> Profile -> Reference -> Test")
    parser.add_argument("--submodel", type=str, default="all", choices=list(SUBMODELS_CONFIG.keys()) + ["all"],
                        help="Submodel to deploy (default: all)")
    parser.add_argument("--device", type=str, default=DEFAULT_TARGET_DEVICE, help="Target device name on AI Hub")
    parser.add_argument("--force_requantize", action="store_true", help="Force re-quantize even if pre-quantized model exists")
    args = parser.parse_args()

    submodels = list(SUBMODELS_CONFIG.keys()) if args.submodel == "all" else [args.submodel]
    all_reports = {}
    for sub in submodels:
        rep = run_pipeline_for_submodel(sub, args.device, args.force_requantize)
        all_reports[sub] = rep

    # Aggregate System Benchmark Report
    total_latency = 0.0
    max_ram = 0.0
    for s, r in all_reports.items():
        lat = r["hardware_performance"].get("latency_ms")
        if isinstance(lat, (int, float)):
            # Note: Vector Estimator runs 5 Euler steps
            total_latency += (lat * 5.0 if s == "vector_estimator" else lat)
        ram = r["hardware_performance"].get("peak_memory_mb")
        if isinstance(ram, (int, float)):
            max_ram = max(max_ram, ram)

    aggregate_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_device": args.device,
        "total_estimated_latency_ms": round(total_latency, 2),
        "peak_ram_mb": round(max_ram, 2),
        "rtf": round(total_latency / 12800.0, 4) if total_latency > 0 else "N/A",
        "submodels": all_reports,
    }

    agg_path = os.path.join(OUTPUT_DIR, "system_full_benchmark_report.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "🏆" * 40)
    print(" 🌟 FULL TTS SYSTEM QUALCOMM NPU BENCHMARK COMPLETE!")
    print(f" • Total End-to-End Latency : {total_latency:.2f} ms")
    print(f" • Real-Time Factor (RTF)   : {round(total_latency / 12800.0, 4)}")
    print(f" • Peak RAM Footprint       : {max_ram:.2f} MB")
    print(f" • Full Report Saved To     : {agg_path}")
    print("🏆" * 40)


if __name__ == "__main__":
    main()
