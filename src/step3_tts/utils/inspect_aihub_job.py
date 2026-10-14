#!/usr/bin/env python3
"""Qualcomm AI Hub Job Inspector & Diagnostics Tool.

Inspects any Qualcomm AI Hub Job (Quantize, Compile, Profile, Inference) to diagnose:
  1. Job Status & Error Details (if FAILED or TIMEOUT).
  2. Compiler Logs & Converter Warnings.
  3. Hardware Layer Placement: NPU (Hexagon HTP) vs CPU Fallback.
  4. Physical Hardware Latency (ms) & Peak Memory (MB).
  5. Downloadable Artifacts & Direct Workbench Links.

Usage:
  .venv/bin/python inspect_aihub_job.py <job_id>
  .venv/bin/python inspect_aihub_job.py --summary  # Inspects all project jobs
"""

import os
import sys
import json
from typing import Dict, Any, Optional

try:
    import qai_hub as hub
except ImportError:
    print("[!] Error: qai_hub package not installed. Run: pip install qai-hub")
    sys.exit(1)

# Ensure utf-8 stdout
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

DEFAULT_TOKEN = "e6hfu2dafj980obg2o60rs5y80ou091ct8g8aofk"
API_TOKEN = os.environ.get("QAI_HUB_API_TOKEN", DEFAULT_TOKEN)

# Key project jobs to inspect
PROJECT_JOBS = {
    "E2E All-in-One Graph (381 MB)": {
        "Quantize W8A16": "jpe6dlyvg",
        "Compile HTP v73": "jp0ozvj0p",
        "Profile Dragonwing": "jgd6qdwlp",
    },
    "Submodel Chain (Dragonwing IQ-9075 EVK)": {
        "Duration Predictor Profile": "jprlqq2kp",
        "Text Encoder Profile": "jp2r6696g",
        "Vector Estimator Compile (QNN DLC)": "jp0oz410p",
        "Vector Estimator Profile": "jgol4j1kg",
        "Vector Estimator Inference": "jp3ojwwnp",
        "Neural Vocoder Profile": "jp0mqql0g",
        "Neural Vocoder Inference": "jgk2nn9og",
    },
    "Samsung Galaxy S24 Ultra": {
        "Vocoder Profile (QNN Binary)": "jp1nk6ykg",
        "Vocoder Inference": "jpxl8x79p",
    }
}


def get_client() -> hub.Client:
    return hub.Client(config=hub.ClientConfig(api_token=API_TOKEN))


def inspect_single_job(job_id: str, client: Optional[hub.Client] = None):
    if client is None:
        client = get_client()

    print("=" * 85)
    print(f" 🔍 INSPECTING QUALCOMM AI HUB JOB: {job_id}")
    print("=" * 85)

    try:
        job = client.get_job(job_id)
    except Exception as e:
        print(f"[!] Error fetching job '{job_id}': {e}")
        return

    # 1. Basic Metadata
    job_type = type(job).__name__.replace("Job", "")
    status_obj = job.get_status()
    status_code = getattr(status_obj, "code", "UNKNOWN")
    status_msg = getattr(status_obj, "message", "")
    device = getattr(job, "device", None)
    device_name = device.name if device else "Cloud Engine / Unassigned"

    print(f" • Job ID          : {job.job_id}")
    print(f" • Job Type        : {job_type}")
    print(f" • Target Device   : {device_name}")
    print(f" • Workbench URL   : https://workbench.aihub.qualcomm.com/jobs/{job.job_id}/")
    print(f" • Current Status  : {status_code}")

    if status_msg:
        print(f" • Status Message  : {status_msg}")

    # 2. Detailed Inspection based on Job Type
    if status_code != "SUCCESS":
        print("\n" + "-" * 85)
        print(" ⚠️ DIAGNOSTIC: JOB KHÔNG THÀNH CÔNG HOẶC ĐANG CHẠY")
        print("-" * 85)
        print(f" • Trạng thái: {status_code}")
        if "timeout" in status_code.lower() or "timeout" in status_msg.lower():
            print(" • Nguyên nhân chẩn đoán: CLOUD_TIMEOUT (Vượt quá giới hạn 300 giây của AI Hub Cloud).")
            print("   -> Mô hình quá lớn hoặc quá trình nạp bộ nhớ vào board mạch vật lý kéo dài > 5 phút.")
        elif "mem" in status_msg.lower() or "alloc" in status_msg.lower():
            print(" • Nguyên nhân chẩn đoán: MEMORY_ALLOCATION_ERROR (Tràn bộ nhớ hoặc thiếu vùng nhớ liên tục).")
            print("   -> Cần dùng runtime QNN DLC thay cho ONNX wrapper, hoặc chia nhỏ submodel.")
        else:
            print(f" • Thông điệp lỗi chi tiết từ Qualcomm Compiler: {status_msg}")

    # If it's a Compile Job
    if "Compile" in job_type and status_code == "SUCCESS":
        print("\n" + "-" * 85)
        print(" ⚙️ CHI TIẾT BIÊN DỊCH (COMPILATION REPORT)")
        print("-" * 85)
        try:
            target_model = job.get_target_model()
            print(f" • Target Model ID : {target_model.model_id}")
            print(f" • Đánh giá        : Biên dịch thành công mã máy Hexagon HTP binary context.")
        except Exception as e:
            print(f" • Compile info: {e}")

    # If it's a Profile Job
    if "Profile" in job_type and status_code == "SUCCESS":
        print("\n" + "-" * 85)
        print(" ⏱️ CHI TIẾT ĐO HIỆU NĂNG PHẦN CỨNG (PROFILE BENCHMARKS)")
        print("-" * 85)
        try:
            prof = job.download_profile()
            summary = prof.get("execution_summary", {})
            inf_time_us = summary.get("estimated_inference_time", 0)
            peak_ram_bytes = summary.get("estimated_inference_peak_memory", 0)
            
            print(f" • Hardware Latency : {inf_time_us / 1000.0:.3f} ms ({inf_time_us} µs)")
            print(f" • Peak RAM Memory  : {peak_ram_bytes / (1024 * 1024):.2f} MB ({peak_ram_bytes} bytes)")
            
            # Layer placement breakdown
            detail = prof.get("execution_detail", [])
            npu_ops = 0
            cpu_ops = 0
            for layer in detail:
                dev = layer.get("device", "").lower()
                if "npu" in dev or "htp" in dev or "dsp" in dev:
                    npu_ops += 1
                else:
                    cpu_ops += 1
            total_ops = npu_ops + cpu_ops
            if total_ops > 0:
                print(f" • Layer Placement  : {npu_ops}/{total_ops} lớp chạy trên NPU ({npu_ops/total_ops*100:.1f}%), {cpu_ops} lớp fallback CPU")
        except Exception as e:
            print(f" • Profile download note: {e}")

    print("=" * 85)
    print(f" 🔗 Mở liên kết sau trên trình duyệt để soi toàn bộ compiler log & biểu đồ 3D:")
    print(f"    https://workbench.aihub.qualcomm.com/jobs/{job.job_id}/")
    print("=" * 85 + "\n")


def inspect_summary():
    client = get_client()
    print("=" * 85)
    print(" 📋 TỔNG HỢP KIỂM TRA TOÀN BỘ JOBS DEPLOYMENT TRÊN QUALCOMM AI HUB")
    print("=" * 85)

    for group_name, jobs in PROJECT_JOBS.items():
        print(f"\n📂 [{group_name}]")
        for sub_name, jid in jobs.items():
            try:
                job = client.get_job(jid)
                status_code = getattr(job.get_status(), "code", "UNKNOWN")
                status_icon = "✅" if status_code == "SUCCESS" else ("⏳" if status_code == "RUNNING" else "⚠️")
                print(f"  {status_icon} {sub_name:<35} : {jid:<12} | Status: {status_code}")
            except Exception as e:
                print(f"  ❌ {sub_name:<35} : {jid:<12} | Lỗi: {e}")
    print("\n" + "=" * 85)
    print(" Tip: Để xem chi tiết 1 job cụ thể (compiler log, memory, layer offload), gõ:")
    print("      .venv/bin/python inspect_aihub_job.py <job_id>")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--job", "-j"] and len(sys.argv) > 2:
        inspect_single_job(sys.argv[2].strip())
    elif len(sys.argv) > 1 and sys.argv[1] not in ["--summary", "-s"]:
        inspect_single_job(sys.argv[1].strip())
    else:
        inspect_summary()
