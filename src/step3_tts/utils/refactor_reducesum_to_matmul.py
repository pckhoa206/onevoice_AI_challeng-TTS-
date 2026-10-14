#!/usr/bin/env python3
"""Surgically refactors ReduceSum -> MatMul(ones) in Supertonic Vector Estimator for Qualcomm HTP NPU.

Replaces:
  ReduceSum(axes=[1, 2], keepdims=0) on mask tensors of shape [B, L, 1]
With:
  1. Reshape([B, L, 1] -> [B, L])
  2. MatMul([B, L] x ones[L, 1] -> [B, 1])
  3. Reshape([B, 1] -> [B])

Eliminates the silent numerical corruption bug on Qualcomm Hexagon HTP NPU
where ReduceSum returned 1 instead of the real length (62/100), magnifying
the RoPE rotary angle by ~62x and causing severe acoustic noise / gibberish speech.
"""

import os
import sys
import shutil
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TARGET_MODELS = [
    os.path.join(ROOT, "outputs", "pipeline_static_models", "vector_estimator_static.onnx"),
    os.path.join(ROOT, "outputs", "pure_npu_compliant_onnx_v2", "vector_estimator_unrolled_5step_pure_npu.onnx"),
    os.path.join(ROOT, "outputs", "pure_npu_dynamic", "vector_estimator_unrolled_5step_npu.onnx"),
    os.path.join(ROOT, "outputs", "pure_npu_dynamic", "vector_estimator_npu.onnx"),
    os.path.join(ROOT, "outputs", "pipeline_static_models", "supertonic_end_to_end_tts_static.onnx"),
]


def patch_reducesum_in_model(model_path: str, backup: bool = True) -> int:
    """Patches all ReduceSum(axes=[1,2]) on masks in the given ONNX model."""
    if not os.path.exists(model_path):
        print(f"  ⚠️ Model not found: {model_path}")
        return 0

    print("=" * 80)
    print(f" 🔬 SURGICALLY PATCHING: {os.path.relpath(model_path, ROOT)}")
    print("=" * 80)

    if backup:
        bak_path = model_path + ".bak_reducesum"
        if not os.path.exists(bak_path):
            shutil.copy2(model_path, bak_path)
            print(f"  • Created backup at '{os.path.relpath(bak_path, ROOT)}'")

    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph

    # Map initializers
    init_map = {init.name: init for init in graph.initializer}

    nodes_to_remove = []
    new_nodes = []
    nodes_replaced = 0

    # Cache for ones matrices and shapes
    cached_inits = set(init_map.keys())

    for idx, node in enumerate(graph.node):
        if node.op_type == "ReduceSum" and "main_blocks.3/attn/ReduceSum" in node.name:
            inp_tensor = node.input[0]
            out_tensor = node.output[0]

            # Determine length L from node name or input name
            # ReduceSum (latent) -> L = 100
            # ReduceSum_1 (text)  -> L = 64
            is_text_mask = "ReduceSum_1" in node.name or "Transpose_3" in inp_tensor
            L = 64 if is_text_mask else 100

            prefix = node.name.replace("/", "_").replace(".", "_")
            print(f"  • Patching node [{node.name}] (Input: {inp_tensor}, L={L})")

            # 1. Initializer: shape_2D [-1, L]
            shape_2d_name = f"{prefix}_shape_2d"
            if shape_2d_name not in cached_inits:
                shape_2d_tensor = helper.make_tensor(
                    shape_2d_name, TensorProto.INT64, [2], [-1, L]
                )
                graph.initializer.append(shape_2d_tensor)
                cached_inits.add(shape_2d_name)

            # 2. Initializer: ones_vector [L, 1]
            ones_name = f"{prefix}_ones_{L}x1"
            if ones_name not in cached_inits:
                ones_arr = np.ones((L, 1), dtype=np.float32)
                ones_tensor = numpy_helper.from_array(ones_arr, name=ones_name)
                graph.initializer.append(ones_tensor)
                cached_inits.add(ones_name)

            # 3. Initializer: shape_1D [-1]
            shape_1d_name = f"{prefix}_shape_1d"
            if shape_1d_name not in cached_inits:
                shape_1d_tensor = helper.make_tensor(
                    shape_1d_name, TensorProto.INT64, [1], [-1]
                )
                graph.initializer.append(shape_1d_tensor)
                cached_inits.add(shape_1d_name)

            # Node A: Reshape [B, L, 1] -> [B, L]
            reshaped_2d_out = f"{prefix}_reshaped_2d"
            node_reshape_2d = helper.make_node(
                "Reshape",
                inputs=[inp_tensor, shape_2d_name],
                outputs=[reshaped_2d_out],
                name=f"{node.name}_Reshape2D",
            )
            new_nodes.append(node_reshape_2d)

            # Node B: MatMul [B, L] x [L, 1] -> [B, 1]
            matmul_out = f"{prefix}_matmul_out"
            node_matmul = helper.make_node(
                "MatMul",
                inputs=[reshaped_2d_out, ones_name],
                outputs=[matmul_out],
                name=f"{node.name}_MatMulOnes",
            )
            new_nodes.append(node_matmul)

            # Node C: Reshape [B, 1] -> [B] (matching original ReduceSum output name)
            node_reshape_1d = helper.make_node(
                "Reshape",
                inputs=[matmul_out, shape_1d_name],
                outputs=[out_tensor],
                name=f"{node.name}_Reshape1D",
            )
            new_nodes.append(node_reshape_1d)

            nodes_replaced += 1
            continue

        new_nodes.append(node)

    if nodes_replaced > 0:
        graph.ClearField("node")
        graph.node.extend(new_nodes)
        onnx.save(model, model_path)
        print(f"  ✅ Successfully replaced {nodes_replaced} ReduceSum nodes with MatMul(ones) in '{os.path.basename(model_path)}'!")
    else:
        print(f"  ℹ️ No ReduceSum RoPE nodes found to patch in '{os.path.basename(model_path)}'.")

    return nodes_replaced


def main():
    total_patched = 0
    for mpath in TARGET_MODELS:
        total_patched += patch_reducesum_in_model(mpath)
    print("\n" + "=" * 80)
    print(f" 🩺 SURGICAL REFACTORING COMPLETED: {total_patched} ReduceSum nodes replaced across all models.")
    print("=" * 80)


if __name__ == "__main__":
    main()
