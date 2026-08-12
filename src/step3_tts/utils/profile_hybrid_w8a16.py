"""Hybrid Offload Production Profiler for Supertonic 3 W8A16 on Qualcomm AI Hub.

Strategy:
  1. vocoder_w8a16            --> NPU (--compute_unit npu)
  2. vector_estimator_w8a16   --> NPU (--compute_unit npu)
  3. text_encoder_w8a16       --> CPU (--compute_unit cpu)
  4. duration_predictor_w8a16 --> CPU (--compute_unit cpu)

Target Device: Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3)
"""
import qai_hub as hub

NPU_MODELS = {
    "vocoder_w8a16": "jp36lwelp",
    "vector_estimator_w8a16": "jp4369wv5",
}

CPU_MODELS = {
    "text_encoder_w8a16": "jgo874vxp",
    "duration_predictor_w8a16": "jp068vl6p",
}

def main():
    devices = hub.get_devices("Samsung Galaxy S24 Ultra")
    if not devices:
        raise RuntimeError("No Samsung Galaxy S24 Ultra device found")
    
    device = devices[0]
    print(f"🚀 Submitting Hybrid Offload Profile Jobs to: {device.name}")
    print("=" * 80)

    summary = {}
    
    # NPU Submodels
    for name, quant_job_id in NPU_MODELS.items():
        quant_job = hub.get_job(quant_job_id)
        model = quant_job.get_target_model()
        print(f" • [NPU Offload] Submitting PROFILE Job for [{name:<25}]...")
        prof_job = hub.submit_profile_job(
            model=model,
            device=device,
            options="--compute_unit npu"
        )
        prof_job.set_name(f"[OFFICIAL_HYBRID] Supertonic3_{name}_NPU")
        summary[name] = {
            "compute": "NPU",
            "job_id": prof_job.job_id,
            "url": f"https://workbench.aihub.qualcomm.com/jobs/{prof_job.job_id}/"
        }
        print(f"   --> NPU Profile Job ID: {prof_job.job_id}")

    # CPU Submodels
    for name, quant_job_id in CPU_MODELS.items():
        quant_job = hub.get_job(quant_job_id)
        model = quant_job.get_target_model()
        print(f" • [CPU Host   ] Submitting PROFILE Job for [{name:<25}]...")
        prof_job = hub.submit_profile_job(
            model=model,
            device=device,
            options="--compute_unit cpu"
        )
        prof_job.set_name(f"[OFFICIAL_HYBRID] Supertonic3_{name}_CPU")
        summary[name] = {
            "compute": "CPU",
            "job_id": prof_job.job_id,
            "url": f"https://workbench.aihub.qualcomm.com/jobs/{prof_job.job_id}/"
        }
        print(f"   --> CPU Profile Job ID: {prof_job.job_id}")

    print("=" * 80)
    print(" ✅ ALL 4 HYBRID W8A16 PROFILE JOBS SUBMITTED SUCCESSFULLY!")
    print("=" * 80)
    for name, info in summary.items():
        print(f" • [{name:<25}] ({info['compute']:<3}): Job ID = {info['job_id']} | URL = {info['url']}")

if __name__ == "__main__":
    main()
