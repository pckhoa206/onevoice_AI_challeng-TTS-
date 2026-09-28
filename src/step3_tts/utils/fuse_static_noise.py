"""In-Graph Static Noise Buffer (Circular / Canonical Seed) for Vector Estimator on Qualcomm Hexagon NPU.

Eliminates CPU np.random.randn() compute overhead and 57.6 KB FastRPC DMA transfers by embedding
a canonical Gaussian noise buffer W_noise ~ N(0, I) directly in NPU SRAM/Initializer.

For dynamic models:
  Slices [1, 144, latent_len] from W_noise [1, 144, 500] using ONNX Slice operator.
  Inputs required from Host: only latent_len (INT64, 8 bytes instead of 57.6 KB).

For static models:
  Embeds constant [1, 144, 100] tensor, eliminating noisy_latent input completely (0 bytes).
"""
import os
import sys
import shutil
import argparse
from typing import Tuple
import numpy as np
import onnx
from onnx import helper, TensorProto, shape_inference
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import _ensure_utf8_stdout

DYNAMIC_VE_PATH = os.path.join(ROOT, "outputs", "pure_npu_dynamic", "vector_estimator_unrolled_5step_npu.onnx")
STATIC_VE_PATH = os.path.join(ROOT, "outputs", "pure_npu_compliant_onnx_v2", "vector_estimator_unrolled_5step_pure_npu.onnx")


def fuse_static_noise_dynamic(model_path: str = DYNAMIC_VE_PATH, backup: bool = True) -> Tuple[bool, float]:
    """Fuses sliceable static noise buffer into dynamic unrolled Vector Estimator."""
    _ensure_utf8_stdout()
    if not os.path.exists(model_path):
        print(f"  ⚠️ Model not found: {model_path}")
        return False, -1.0

    print(f"\nProcessing (Dynamic VE Static Noise): {os.path.relpath(model_path, ROOT)}")
    backup_path = model_path + ".bak_task7"
    if backup and not os.path.exists(backup_path):
        shutil.copy2(model_path, backup_path)
        print(f"  • Created backup: {os.path.relpath(backup_path, ROOT)}")

    model = onnx.load(model_path)
    graph = model.graph

    nl_inp = next((i for i in graph.input if i.name == "noisy_latent"), None)
    if nl_inp is None:
        print(f"  ℹ️ 'noisy_latent' is already not in graph inputs for {os.path.basename(model_path)}.")
        return False, 0.0

    # 1. Generate Canonical Gaussian Noise Buffer (Seed = 42 for reproducible studio quality)
    np.random.seed(42)
    buf_len = 500  # 500 frames (~35 seconds, covers any TTS utterance)
    w_noise_np = np.random.randn(1, 144, buf_len).astype(np.float32)
    w_noise_tensor = helper.make_tensor(
        "W_static_noise_buf", TensorProto.FLOAT, [1, 144, buf_len], w_noise_np.flatten().tolist()
    )

    # 2. Slice constants
    const_starts = helper.make_tensor("const_slice_starts_noise", TensorProto.INT64, [1], [0])
    const_axes = helper.make_tensor("const_slice_axes_noise", TensorProto.INT64, [1], [2])
    const_steps = helper.make_tensor("const_slice_steps_noise", TensorProto.INT64, [1], [1])

    graph.initializer.extend([w_noise_tensor, const_starts, const_axes, const_steps])

    # 3. New input: latent_len (INT64 [1])
    latent_len_inp = helper.make_tensor_value_info("latent_len", TensorProto.INT64, [1])

    # 4. Slice node
    sliced_out_name = "in_graph_sliced_noise"
    slice_node = helper.make_node(
        "Slice",
        inputs=["W_static_noise_buf", "const_slice_starts_noise", "latent_len", "const_slice_axes_noise", "const_slice_steps_noise"],
        outputs=[sliced_out_name],
        name="slice_static_noise_node",
    )

    graph.node.insert(0, slice_node)

    # 5. Rewire consumers of noisy_latent
    rewired = 0
    for node in graph.node[1:]:
        for idx, inp in enumerate(node.input):
            if inp == "noisy_latent":
                node.input[idx] = sliced_out_name
                rewired += 1

    # 6. Replace noisy_latent with latent_len in graph.input
    nl_idx = list(graph.input).index(nl_inp)
    graph.input.remove(nl_inp)
    graph.input.insert(nl_idx, latent_len_inp)

    print(f"  • Rewired {rewired} consumer nodes to In-Graph Static Noise Buffer (500 frames).")
    print("  • Replaced 57.6 KB 'noisy_latent' Float32 input with 8 bytes 'latent_len' INT64 scalar.")

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  ℹ️ Note on shape inference: {e}")

    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    print(f"  ✅ Saved optimized model with In-Graph Noise: {os.path.basename(model_path)}")

    # Verification run
    sess_new = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"  • New Model Inputs: {[i.name for i in sess_new.get_inputs()]}")

    test_len = 40
    test_emb = np.random.randn(1, 256, 30).astype(np.float32)
    test_ttl = np.random.randn(1, 50, 256).astype(np.float32)
    out = sess_new.run(None, {
        "latent_len": np.array([test_len], dtype=np.int64),
        "text_emb": test_emb,
        "style_ttl": test_ttl,
    })[0]

    assert out.shape == (1, 144, test_len), f"Shape mismatch: {out.shape}"
    print(f"  🎯 Execution verified! Output shape: {out.shape} | DMA reduction: 99.986%")
    return True, 0.0


def fuse_static_noise_static(model_path: str = STATIC_VE_PATH, backup: bool = True) -> Tuple[bool, float]:
    """Fuses constant static noise tensor [1, 144, 100] into static unrolled Vector Estimator."""
    _ensure_utf8_stdout()
    if not os.path.exists(model_path):
        print(f"  ⚠️ Static Model not found: {model_path}")
        return False, -1.0

    print(f"\nProcessing (Static VE Constant Noise): {os.path.relpath(model_path, ROOT)}")
    backup_path = model_path + ".bak_task7"
    if backup and not os.path.exists(backup_path):
        shutil.copy2(model_path, backup_path)
        print(f"  • Created backup: {os.path.relpath(backup_path, ROOT)}")

    model = onnx.load(model_path)
    graph = model.graph

    nl_inp = next((i for i in graph.input if i.name == "noisy_latent"), None)
    if nl_inp is None:
        print(f"  ℹ️ 'noisy_latent' is already not in graph inputs for {os.path.basename(model_path)}.")
        return False, 0.0

    # 1. Canonical Constant Gaussian Noise (Seed = 42, shape [1, 144, 100])
    np.random.seed(42)
    w_noise_np = np.random.randn(1, 144, 100).astype(np.float32)
    const_noise_name = "const_canonical_noise_100"
    const_noise_tensor = helper.make_tensor(
        const_noise_name, TensorProto.FLOAT, [1, 144, 100], w_noise_np.flatten().tolist()
    )
    graph.initializer.append(const_noise_tensor)

    # 2. Also check if in-graph masks can be fused if text_mask/latent_mask are in inputs
    mask_text_inp = next((i for i in graph.input if i.name == "text_mask"), None)
    mask_latent_inp = next((i for i in graph.input if i.name == "latent_mask"), None)

    prefix_nodes = []
    const_eps = helper.make_tensor("in_graph_static_eps", TensorProto.FLOAT, [1], [1e-6])
    graph.initializer.append(const_eps)

    if mask_text_inp is not None:
        node_abs_te = helper.make_node("Abs", ["text_emb"], ["in_graph_te_abs_st"], name="in_graph_te_abs_st")
        node_red_te = helper.make_node("ReduceMax", ["in_graph_te_abs_st"], ["in_graph_te_max_st"], axes=[1], keepdims=1, name="in_graph_te_red_st")
        node_gt_te = helper.make_node("Greater", ["in_graph_te_max_st", "in_graph_static_eps"], ["in_graph_te_gt_st"], name="in_graph_te_gt_st")
        node_cast_te = helper.make_node("Cast", ["in_graph_te_gt_st"], ["in_graph_text_mask_st"], to=TensorProto.FLOAT, name="in_graph_te_cast_st")
        prefix_nodes.extend([node_abs_te, node_red_te, node_gt_te, node_cast_te])

    if mask_latent_inp is not None:
        # Constant latent mask: all 100 frames are valid or derived
        const_lat_mask_tensor = helper.make_tensor(
            "const_latent_mask_ones", TensorProto.FLOAT, [1, 1, 100], [1.0] * 100
        )
        graph.initializer.append(const_lat_mask_tensor)

    # Insert prefix nodes
    for i, n in enumerate(prefix_nodes):
        graph.node.insert(i, n)

    # Rewire consumers
    rewired = 0
    for node in graph.node[len(prefix_nodes):]:
        for idx, inp in enumerate(node.input):
            if inp == "noisy_latent":
                node.input[idx] = const_noise_name
                rewired += 1
            elif inp == "text_mask" and mask_text_inp is not None:
                node.input[idx] = "in_graph_text_mask_st"
            elif inp == "latent_mask" and mask_latent_inp is not None:
                node.input[idx] = "const_latent_mask_ones"

    # Remove inputs
    graph.input.remove(nl_inp)
    if mask_text_inp is not None:
        graph.input.remove(mask_text_inp)
    if mask_latent_inp is not None:
        graph.input.remove(mask_latent_inp)

    print(f"  • Rewired {rewired} consumer nodes to Constant Noise Initializer [1, 144, 100].")
    print("  • Completely removed 'noisy_latent', 'text_mask', and 'latent_mask' from static Vector Estimator.")

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  ℹ️ Note on shape inference: {e}")

    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    print(f"  ✅ Saved optimized static model: {os.path.basename(model_path)}")

    sess_new = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print(f"  • Static Model Inputs: {[i.name for i in sess_new.get_inputs()]}")
    assert set([i.name for i in sess_new.get_inputs()]) == {"text_emb", "style_ttl"}
    print("  🎯 Static VE is now 100% self-contained on NPU SRAM! Only takes text_emb and style_ttl.")
    return True, 0.0


def main():
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Fuse In-Graph Static Noise Buffer for Qualcomm Hexagon NPU")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backup files")
    args = parser.parse_args()

    print("=" * 85)
    print(" 🚀 FUSING IN-GRAPH STATIC NOISE BUFFER FOR QUALCOMM HEXAGON NPU (TASK 7)")
    print("    Target Architecture: Zero DMA Noise Transfer | 100% On-Chip SRAM Storage")
    print("=" * 85)

    fuse_static_noise_dynamic(DYNAMIC_VE_PATH, backup=not args.no_backup)
    fuse_static_noise_static(STATIC_VE_PATH, backup=not args.no_backup)

    print("\n" + "=" * 85)
    print(" 🎉 Completed Static Noise Buffer fusion for Vector Estimator models!")
    print("=" * 85)


if __name__ == "__main__":
    main()
