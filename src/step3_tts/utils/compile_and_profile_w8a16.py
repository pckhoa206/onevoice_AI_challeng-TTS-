"""Full Production Pipeline to Compile & Profile Supertonic 3 W8A16 on Qualcomm AI Hub.

Applies:
  1. `--target_runtime qnn_context_binary`
  2. `--truncate_64bit_io`
Target Device: Samsung Galaxy S24 Ultra (Snapdragon 8 Gen 3)
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
    print(f"🚀 Submitting 4 W8A16 Compile & Profile Jobs to Target Device: {device.name}")
    print("=" * 80)

    summary = {}
    for name, quant_job_id in QUANT_JOBS.items():
        quant_job = hub.get_job(quant_job_id)
        raw_model = quant_job.get_target_model()
        print(f" • [Step 1/2] Submitting COMPILE Job for [{name:<25}] (Model: {raw_model.model_id})...")
        compile_job = hub.submit_compile_job(
            model=raw_model,
            device=device,
            options="--target_runtime qnn_context_binary --truncate_64bit_io"
        )
        print(f"   --> Compiled Job ID: {compile_job.job_id}")

        compiled_model = compile_job.get_target_model()
        print(f" • [Step 2/2] Submitting PROFILE Job for [{name:<25}] (QNN Model: {compiled_model.model_id})...")
        profile_job = hub.submit_profile_job(
            model=compiled_model,
            device=device
        )
        print(f"   --> Profile Job ID: {profile_job.job_id}")

        summary[name] = {
            "compile_job_id": compile_job.job_id,
            "profile_job_id": profile_job.job_id,
            "url": f"https://workbench.aihub.qualcomm.com/jobs/{profile_job.job_id}/"
        }

    print("=" * 80)
    print(" ✅ ALL 4 W8A16 QNN CONTEXT BINARY COMPILE & PROFILE JOBS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    for name, info in summary.items():
        print(f" • [{name:<25}]: Profile Job ID = {info['profile_job_id']} | URL = {info['url']}")

if __name__ == "__main__":
    main()
