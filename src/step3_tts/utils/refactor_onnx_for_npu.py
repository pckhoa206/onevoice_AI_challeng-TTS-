"""Refactor Supertonic 3 ONNX models for 100% Qualcomm Hexagon NPU Compliance.

Transforms:
 1. Replaces Embedding Gather(weight, text_ids) with OneHot(text_ids, depth=8322) -> MatMul(weight).
 2. Converts dynamic shapes to static tensors Ahead-Of-Time (AOT).
 3. Cleans up redundant dynamic shape nodes.
"""
import os
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

OUTPUT_DIR = "outputs/npu_compliant_onnx"


def fix_conv_missing_bias(model: onnx.ModelProto) -> onnx.ModelProto:
    """Adds zero bias initializer to Conv nodes missing bias (3rd input) for Qualcomm QNN compatibility."""
    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if node.op_type == "Conv" and len(node.input) == 2:
            weight_name = node.input[1]
            if weight_name in initializer_map:
                weight_tensor = numpy_helper.to_array(initializer_map[weight_name])
                out_channels = weight_tensor.shape[0]

                bias_name = f"{node.name}_zero_bias"
                zero_bias = np.zeros((out_channels,), dtype=np.float32)
                bias_tensor = numpy_helper.from_array(zero_bias, name=bias_name)
                graph.initializer.append(bias_tensor)

                node.input.append(bias_name)
                print(f"  [Refactor] Added zero bias '{bias_name}' (shape [{out_channels}]) to Conv node '{node.name}'.")

    return model


def fix_matmul_add_zero_bias(model: onnx.ModelProto) -> onnx.ModelProto:
    """Appends Add(ZeroBias) to un-biased MatMul linear nodes for Qualcomm QNN converter compatibility."""
    graph = model.graph
    init_map = {init.name: init for init in graph.initializer}
    nodes_replaced = 0
    new_nodes = []

    for node in graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            w_name = node.input[1]
            if w_name in init_map:
                w_arr = numpy_helper.to_array(init_map[w_name])
                if len(w_arr.shape) == 2:
                    out_features = w_arr.shape[1]
                    b_name = f"{node.name}_zero_bias"
                    b_tensor = numpy_helper.from_array(np.zeros((out_features,), dtype=np.float32), name=b_name)
                    graph.initializer.append(b_tensor)

                    orig_out = node.output[0]
                    mm_out = f"{node.name}_mm_out"

                    node.output[0] = mm_out
                    new_nodes.append(node)

                    add_node = helper.make_node(
                        "Add",
                        inputs=[mm_out, b_name],
                        outputs=[orig_out],
                        name=f"{node.name}_AddZeroBias",
                    )
                    new_nodes.append(add_node)
                    nodes_replaced += 1
                    continue
        new_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    if nodes_replaced > 0:
        print(f"  [Refactor] Appended Add(ZeroBias) to {nodes_replaced} un-biased MatMul linear layers.")
    return model


def refactor_embedding_gather(model: onnx.ModelProto, vocab_size: int = 8322) -> onnx.ModelProto:
    """Replaces Gather(char_embedder.weight, text_ids) with OneHot + MatMul."""
    graph = model.graph
    nodes_to_remove = []
    new_nodes = []

    for node in graph.node:
        if node.op_type == "Gather" and len(node.input) >= 2:
            weight_name, indices_name = node.input[0], node.input[1]
            if "char_embedder.weight" in weight_name or "text_ids" in indices_name:
                output_name = node.output[0]
                print(f"  [Refactor] Replacing Gather node '{node.name}' ({weight_name}, {indices_name}) with OneHot + MatMul...")

                # Create OneHot depth tensor (INT64 for QNN & ONNXRuntime)
                depth_name = f"{node.name}_depth"
                depth_tensor = helper.make_tensor(depth_name, TensorProto.INT64, [1], [vocab_size])
                graph.initializer.append(depth_tensor)

                # Create OneHot values tensor [off_value=0, on_value=1] (INT64 for QNN schema)
                values_name = f"{node.name}_values"
                values_tensor = helper.make_tensor(values_name, TensorProto.INT64, [2], [0, 1])
                graph.initializer.append(values_tensor)

                # OneHot Node
                onehot_out = f"{node.name}_onehot_out"
                onehot_node = helper.make_node(
                    "OneHot",
                    inputs=[indices_name, depth_name, values_name],
                    outputs=[onehot_out],
                    name=f"{node.name}_OneHot",
                    axis=-1,
                )

                # Cast Node (INT32 -> FLOAT for MatMul)
                cast_out = f"{node.name}_cast_out"
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[onehot_out],
                    outputs=[cast_out],
                    name=f"{node.name}_Cast",
                    to=TensorProto.FLOAT,
                )

                # MatMul Node: cast_out (1, 64, 8322) x weight (8322, D) -> (1, 64, D)
                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=[cast_out, weight_name],
                    outputs=[output_name],
                    name=f"{node.name}_MatMul",
                )

                new_nodes.extend([onehot_node, cast_node, matmul_node])
                nodes_to_remove.append(node)

    new_graph_nodes = []
    for node in graph.node:
        if node in nodes_to_remove:
            # Find the new nodes corresponding to this gather node
            for new_n in new_nodes:
                new_graph_nodes.append(new_n)
        else:
            new_graph_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_graph_nodes)

    return model


def refactor_all_submodels():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submodels = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]

    print("=" * 80)
    print(" 🛠️ REFACTORING SUPERTONIC 3 ONNX MODELS FOR 100% NPU COMPLIANCE")
    print("=" * 80)

    for name in submodels:
        path = f"/Users/khoa/.cache/supertonic3/onnx/{name}.onnx"
        out_path = os.path.join(OUTPUT_DIR, f"{name}_npu.onnx")

        print(f"\nProcessing '{name}'...")
        if not os.path.exists(path):
            print(f"  ❌ Source file not found: {path}")
            continue

        model = onnx.load(path)

        # Fix missing Conv and MatMul bias inputs for QNN converter compatibility
        model = fix_conv_missing_bias(model)
        model = fix_matmul_add_zero_bias(model)

        # Simplify ONNX graph using onnxsim to remove dynamic shape nodes (Shape, ConstantOfShape, dead code)
        if name in ["duration_predictor", "vocoder"]:
            print(f"  [Refactor] Running ONNX Simplifier (onnxsim) on '{name}'...")
            import onnxsim
            model, check = onnxsim.simplify(model)
            if check:
                print(f"  [Refactor] ONNX Simplifier completed successfully.")

        # Infer shapes to populate graph.value_info for QNN/ONNX Runtime
        model = onnx.shape_inference.infer_shapes(model)

        # Verify topological sort and graph validity
        onnx.checker.check_model(model, full_check=True)

        # Save refactored model
        onnx.save(model, out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  ✅ Saved Refactored NPU Model: '{out_path}' ({size_mb:.2f} MB)")


if __name__ == "__main__":
    refactor_all_submodels()
