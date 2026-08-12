"""Profile all 4 Qualcomm W8A16 Quantized Models on Snapdragon 8 Elite QRD.

Submits profiling jobs for:
  1. duration_predictor_w8a16 (Job ID: jp068vl6p)
  2. vocoder_w8a16            (Job ID: jp36lwelp)
  3. vector_estimator_w8a16   (Job ID: jp4369wv5)
  4. text_encoder_w8a16       (Job ID: jgo874vxp)
"""
import qai_hub as hub

QUANT_JOBS = {
    "duration_predictor_w8a16": "jp068vl6p",
    "vocoder_w8a16": "jp36lwelp",
    "vector_estimator_w8a16": "jp4369wv5",
    "text_encoder_w8a16": "jgo874vxp",
}

def main():
    devices = hub.get_devices("Snapdragon 8 Elite QRD")
    if not devices:
        raise RuntimeError("No Snapdragon 8 Elite QRD device found")
    
    device = devices[0]
    print(f"🚀 Submitting 4 W8A16 Profile Jobs to Target Device: {device.name}")
    print("=" * 80)

    profile_results = {}
    for name, quant_job_id in QUANT_JOBS.items():
        job = hub.get_job(quant_job_id)
        model = job.get_target_model()
        print(f" • Submitting Profile for [{name:<25}] (Model ID: {model.model_id})...")
        prof_job = hub.submit_profile_job(model=model, device=device)
        profile_results[name] = {
            "profile_job_id": prof_job.job_id,
            "url": f"https://workbench.aihub.qualcomm.com/jobs/{prof_job.job_id}/"
        }
        print(f"   --> Submitted Profile Job ID: {prof_job.job_id}")

    print("=" * 80)
    print(" ✅ ALL 4 W8A16 PROFILE JOBS SUBMITTED SUCCESSFULLY!")
    print("=" * 80)
    for name, info in profile_results.items():
        print(f" • [{name:<25}]: Job ID = {info['profile_job_id']} | URL = {info['url']}")

if __name__ == "__main__":
    main()
