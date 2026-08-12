"""Compile vector_estimator sub-model on Qualcomm AI Hub using existing Model ID."""
import os
import sys
import qai_hub as hub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout

DEVICE_NAME = "Snapdragon 8 Elite QRD"
OUTPUT_DIR = "outputs/qnn_binaries"


def compile_existing(model_id: str, submodel_name: str, input_specs: dict):
    print("=" * 75)
    print(f"  QUALCOMM AI HUB — COMPILING EXISTING {submodel_name.upper()} (Model ID: {model_id})")
    print("=" * 75)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = hub.Device(DEVICE_NAME)
    model = hub.get_model(model_id)

    print(f"[1/3] Submitting NPU Compile Job for '{submodel_name}'...")
    compile_job = hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options="--target_runtime onnx --compute_unit npu",
    )
    print(f"      Compile Job ID: {compile_job.job_id}")
    print(f"      Web Link: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
    print("      Waiting for compilation...")
    compile_job.wait()

    target_model = compile_job.get_target_model()

    print(f"[2/3] Submitting Remote Profile Job on Snapdragon 8 Elite...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )
    print(f"      Profile Job ID: {profile_job.job_id}")
    print(f"      Web Link: https://workbench.aihub.qualcomm.com/jobs/{profile_job.job_id}/")
    profile_job.wait()

    profile_data = profile_job.download_profile()
    print("\n" + "=" * 75)
    print(f"  NPU PROFILING RESULTS ({submodel_name.upper()}):")
    print("=" * 75)
    print(profile_data)

    out_path = os.path.join(OUTPUT_DIR, f"{submodel_name}_npu_fp16.bin")
    target_model.download(out_path)
    print(f"\n✅ DOWNLOADED QNN BINARY: {out_path}\n")


def main():
    _ensure_utf8_stdout()

    # Compile Vector Estimator (Model ID: mq919oyrn) with float32 step inputs
    vector_specs = {
        "noisy_latent": ((1, 144, 100), "float32"),
        "text_emb": ((1, 256, 64), "float32"),
        "style_ttl": ((1, 50, 256), "float32"),
        "latent_mask": ((1, 1, 100), "float32"),
        "text_mask": ((1, 1, 64), "float32"),
        "current_step": ((1,), "float32"),
        "total_step": ((1,), "float32"),
    }
    compile_existing("mq919oyrn", "vector_estimator", vector_specs)


if __name__ == "__main__":
    main()
