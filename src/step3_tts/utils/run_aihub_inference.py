"""Run Cloud On-Device Inference on Qualcomm AI Hub for all 4 Supertonic 3 W8A16 Submodels.

Submits INFERENCE jobs to Samsung Galaxy S24 Ultra for:
  1. vocoder_w8a16            (Job ID: jp36lwelp)
  2. duration_predictor_w8a16 (Job ID: jp068vl6p)
  3. text_encoder_w8a16       (Job ID: jgo874vxp)
  4. vector_estimator_w8a16   (Job ID: jp4369wv5)
"""
import qai_hub as hub
import numpy as np

def main():
    devices = hub.get_devices("Samsung Galaxy S24 Ultra")
    if not devices:
        raise RuntimeError("No Samsung Galaxy S24 Ultra device found")
    
    device = devices[0]
    print(f"🚀 Submitting 4 INFERENCE Jobs to Qualcomm Cloud Device: {device.name}")
    print("=" * 80)

    # 1. Vocoder W8A16
    job_voc = hub.get_job("jp36lwelp")
    model_voc = job_voc.get_target_model()
    latent_data = np.random.randn(1, 144, 64).astype(np.float32)
    print(" • [1/4] Submitting INFERENCE for [vocoder_w8a16]...")
    inf_voc = hub.submit_inference_job(
        model=model_voc,
        inputs=dict(latent=[latent_data]),
        device=device
    )
    inf_voc.set_name("[OFFICIAL_INFERENCE] Supertonic3_vocoder_w8a16")
    print(f"   --> Inference Job ID: {inf_voc.job_id}")

    # 2. Duration Predictor W8A16
    job_dp = hub.get_job("jp068vl6p")
    model_dp = job_dp.get_target_model()
    text_ids_data = np.ones((1, 64), dtype=np.int32)
    print(" • [2/4] Submitting INFERENCE for [duration_predictor_w8a16]...")
    inf_dp = hub.submit_inference_job(
        model=model_dp,
        inputs=dict(text_ids=[text_ids_data]),
        device=device
    )
    inf_dp.set_name("[OFFICIAL_INFERENCE] Supertonic3_duration_predictor_w8a16")
    print(f"   --> Inference Job ID: {inf_dp.job_id}")

    # 3. Text Encoder W8A16
    job_te = hub.get_job("jgo874vxp")
    model_te = job_te.get_target_model()
    style_ttl_data = np.random.randn(1, 50, 256).astype(np.float32)
    print(" • [3/4] Submitting INFERENCE for [text_encoder_w8a16]...")
    inf_te = hub.submit_inference_job(
        model=model_te,
        inputs=dict(text_ids=[text_ids_data], style_ttl=[style_ttl_data]),
        device=device
    )
    inf_te.set_name("[OFFICIAL_INFERENCE] Supertonic3_text_encoder_w8a16")
    print(f"   --> Inference Job ID: {inf_te.job_id}")

    # 4. Vector Estimator W8A16
    job_ve = hub.get_job("jp4369wv5")
    model_ve = job_ve.get_target_model()
    sample_data = np.random.randn(1, 144, 64).astype(np.float32)
    timestep_data = np.array([0.5], dtype=np.float32)
    text_emb_data = np.random.randn(1, 256, 64).astype(np.float32)
    print(" • [4/4] Submitting INFERENCE for [vector_estimator_w8a16]...")
    inf_ve = hub.submit_inference_job(
        model=model_ve,
        inputs=dict(
            sample=[sample_data],
            timestep=[timestep_data],
            text_emb=[text_emb_data],
            style_ttl=[style_ttl_data]
        ),
        device=device
    )
    inf_ve.set_name("[OFFICIAL_INFERENCE] Supertonic3_vector_estimator_w8a16")
    print(f"   --> Inference Job ID: {inf_ve.job_id}")

    print("=" * 80)
    print(" ✅ ALL 4 INFERENCE JOBS SUBMITTED TO QUALCOMM AI HUB CLOUD DEVICE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
