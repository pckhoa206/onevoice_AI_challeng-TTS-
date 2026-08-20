"""Build Unrolled 5-Step Vector Estimator ONNX Graph for Hexagon NPU.

Chains 5 Euler ODE steps (dt = 0.2) in a single static computational graph
reusing the same weight initializers (zero memory footprint increase).
"""
import os
import sys
import copy
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

VE_PATH = "outputs/pure_npu_compliant_onnx_v2/vector_estimator_pure_npu.onnx"
OUTPUT_PATH = "outputs/pure_npu_compliant_onnx_v2/vector_estimator_unrolled_5step_pure_npu.onnx"


def build_unrolled_vector_estimator(total_steps: int = 5):
    _ensure_utf8_stdout()
    print("=" * 80)
    print(f" 🚀 BUILDING UNROLLED {total_steps}-STEP VECTOR ESTIMATOR NPU GRAPH")
    print("=" * 80)

    base_model = onnx.load(VE_PATH)
    base_graph = base_model.graph

    new_graph_nodes = []
    initializers = list(base_graph.initializer)

    # Initializers for dt (0.2) and total_step (5.0)
    dt_val = 1.0 / float(total_steps)
    dt_name = "const_euler_dt"
    dt_tensor = helper.make_tensor(dt_name, TensorProto.FLOAT, [1], [dt_val])
    initializers.append(dt_tensor)

    total_step_name = "const_total_step_5"
    total_step_tensor = helper.make_tensor(total_step_name, TensorProto.FLOAT, [1], [float(total_steps)])
    initializers.append(total_step_tensor)

    current_latent = "noisy_latent"

    for step_idx in range(1, total_steps + 1):
        step_str = f"step_{step_idx}"
        print(f" • Unrolling Step {step_idx}/{total_steps} (Euler ODE step t={step_idx})...")

        # Step constant initializer
        step_const_name = f"const_step_{step_idx}"
        step_tensor = helper.make_tensor(step_const_name, TensorProto.FLOAT, [1], [float(step_idx)])
        initializers.append(step_tensor)

        # Mapping for node inputs in this step
        io_map = {
            "noisy_latent": current_latent,
            "text_emb": "text_emb",
            "style_ttl": "style_ttl",
            "latent_mask": "latent_mask",
            "text_mask": "text_mask",
            "current_step": step_const_name,
            "total_step": total_step_name,
        }

        for node in base_graph.node:
            new_node = copy.deepcopy(node)
            new_node.name = f"{node.name}_{step_str}"

            # Map inputs
            new_inputs = []
            for inp in node.input:
                if inp in io_map:
                    new_inputs.append(io_map[inp])
                elif any(init.name == inp for init in base_graph.initializer):
                    new_inputs.append(inp)
                else:
                    new_inputs.append(f"{inp}_{step_str}")
            new_node.input[:] = new_inputs

            # Map outputs
            new_outputs = []
            for out in node.output:
                new_outputs.append(f"{out}_{step_str}")
            new_node.output[:] = new_outputs

            new_graph_nodes.append(new_node)

        # Output of this base graph run is denoised_latent_{step_str} (v_pred)
        v_pred_name = f"denoised_latent_{step_str}"

        # Euler ODE update: latent_{step_idx} = current_latent + dt * v_pred
        scaled_v_name = f"scaled_v_{step_str}"
        mul_node = helper.make_node(
            "Mul",
            inputs=[v_pred_name, dt_name],
            outputs=[scaled_v_name],
            name=f"Euler_Mul_dt_{step_str}",
        )
        new_graph_nodes.append(mul_node)

        next_latent_name = f"latent_after_{step_str}" if step_idx < total_steps else "denoised_latent"
        add_node = helper.make_node(
            "Add",
            inputs=[current_latent, scaled_v_name],
            outputs=[next_latent_name],
            name=f"Euler_Add_Latent_{step_str}",
        )
        new_graph_nodes.append(add_node)

        current_latent = next_latent_name

    new_inputs = [
        helper.make_tensor_value_info("noisy_latent", TensorProto.FLOAT, [1, 144, 100]),
        helper.make_tensor_value_info("text_emb", TensorProto.FLOAT, [1, 256, 64]),
        helper.make_tensor_value_info("style_ttl", TensorProto.FLOAT, [1, 50, 256]),
        helper.make_tensor_value_info("latent_mask", TensorProto.FLOAT, [1, 1, 100]),
        helper.make_tensor_value_info("text_mask", TensorProto.FLOAT, [1, 1, 64]),
    ]

    new_outputs = [
        helper.make_tensor_value_info("denoised_latent", TensorProto.FLOAT, [1, 144, 100]),
    ]

    unrolled_graph = helper.make_graph(
        nodes=new_graph_nodes,
        name="VectorEstimator_Unrolled_5Steps",
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=initializers,
    )

    unrolled_model = helper.make_model(
        unrolled_graph,
        producer_name="OneVoice_NPU_Transformer",
        opset_imports=[helper.make_opsetid("", 17)],
    )

    print(" • Running shape inference on unrolled graph...")
    unrolled_model = shape_inference.infer_shapes(unrolled_model)

    onnx.save(unrolled_model, OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"  ✅ Saved Unrolled 5-Step NPU Model: '{OUTPUT_PATH}' ({size_mb:.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    build_unrolled_vector_estimator(5)
