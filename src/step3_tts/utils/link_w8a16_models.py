"""Link all 4 Qualcomm W8A16 Models to QNN Context Binary on Qualcomm AI Hub.

Submits LINK jobs for:
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
    devices = hub.get_devices("Samsung Galaxy S24 Ultra")
    if not devices:
        raise RuntimeError("No Samsung Galaxy S24 Ultra device found")
    
    device = devices[0]
    print(f"🚀 Submitting 4 W8A16 LINK Jobs to Target Device: {device.name}")
    print("=" * 80)

    link_results = {}
    for name, quant_job_id in QUANT_JOBS.items():
        job = hub.get_job(quant_job_id)
        model = job.get_target_model()
        print(f" • Submitting LINK Job for [{name:<25}] (Model ID: {model.model_id})...")
        link_job = hub.submit_link_job(
            models=model,
            device=device,
            name=f"[OFFICIAL_LINK] Supertonic3_{name}"
        )
        link_results[name] = {
            "link_job_id": link_job.job_id,
            "url": f"https://workbench.aihub.qualcomm.com/jobs/{link_job.job_id}/"
        }
        print(f"   --> Submitted LINK Job ID: {link_job.job_id}")

    print("=" * 80)
    print(" ✅ ALL 4 W8A16 LINK JOBS SUBMITTED SUCCESSFULLY!")
    print("=" * 80)
    for name, info in link_results.items():
        print(f" • [{name:<25}]: Job ID = {info['link_job_id']} | URL = {info['url']}")

if __name__ == "__main__":
    main()
