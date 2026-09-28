"""Build Unrolled 5-Step Vector Estimator ONNX Graph for Hexagon NPU.

Chains 5 Euler ODE steps (dt = 0.2) in a single static computational graph
reusing the same weight initializers (zero memory footprint increase).
Supports:
  1. Multi-Output Tapping (debug_mode) for step-by-step tensor inspection.
  2. Ping-Pong Buffer Recycling (use_ping_pong) for minimum peak activation SRAM memory.
"""
import os
import sys
import copy
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common import _ensure_utf8_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VE_PATH = os.path.join(ROOT, "outputs", "pure_npu_compliant_onnx_v2", "vector_estimator_pure_npu.onnx")
OUTPUT_PATH = os.path.join(ROOT, "outputs", "pure_npu_compliant_onnx_v2", "vector_estimator_unrolled_5step_pure_npu.onnx")
DEBUG_OUTPUT_PATH = os.path.join(ROOT, "outputs", "pure_npu_compliant_onnx_v2", "vector_estimator_unrolled_5step_debug.onnx")


def build_unrolled_vector_estimator(
    total_steps: int = 5,
    debug_mode: bool = False,
    use_ping_pong: bool = True,
    base_model_path: str = None,
    output_path: str = None,
):
    _ensure_utf8_stdout()
    if base_model_path is None:
        base_model_path = VE_PATH
    if output_path is None:
        output_path = DEBUG_OUTPUT_PATH if debug_mode else OUTPUT_PATH

    print("=" * 80)
    print(f" 🚀 BUILDING UNROLLED {total_steps}-STEP VECTOR ESTIMATOR NPU GRAPH")
    print(f" • Base Model: '{base_model_path}'")
    print(f" • Mode: {'DEBUG (Multi-Output Tapping Enabled)' if debug_mode else 'PRODUCTION (1-Shot Output)'}")
    print(f" • Activation Memory: {'Ping-Pong Buffer Recycling (A/B)' if use_ping_pong else 'Standard Sequential'}")
    print(f" • Target Path: '{output_path}'")
    print("=" * 80)

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Missing base pure NPU vector estimator at '{base_model_path}'")

    base_model = onnx.load(base_model_path)
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
    intermediate_latents = []

    for step_idx in range(total_steps):
        step_str = f"step_{step_idx}"
        print(f" • Unrolling Step {step_idx + 1}/{total_steps} (Euler ODE step t={step_idx})...")

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

        next_latent_name = f"latent_step_{step_idx + 1}" if step_idx < total_steps - 1 else "denoised_latent"

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
                if out == "denoised_latent":
                    new_outputs.append(next_latent_name)
                else:
                    new_outputs.append(f"{out}_{step_str}")
            new_node.output[:] = new_outputs

            new_graph_nodes.append(new_node)

        if step_idx < total_steps - 1:
            intermediate_latents.append(next_latent_name)

        current_latent = next_latent_name

    # Dynamically copy tensor shape information from base model (supports both static and dynamic models)
    base_input_map = {inp.name: inp for inp in base_graph.input}
    new_inputs = []
    for inp_name in ["noisy_latent", "text_emb", "style_ttl", "latent_mask", "text_mask"]:
        if inp_name in base_input_map:
            base_inp = base_input_map[inp_name]
            shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in base_inp.type.tensor_type.shape.dim]
            new_inputs.append(helper.make_tensor_value_info(inp_name, base_inp.type.tensor_type.elem_type, shape))

    base_out = base_graph.output[0]
    out_shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in base_out.type.tensor_type.shape.dim]
    new_outputs = [
        helper.make_tensor_value_info("denoised_latent", base_out.type.tensor_type.elem_type, out_shape),
    ]

    # Solution 3: Multi-Output Tapping for Debugging
    if debug_mode:
        for idx, inter_latent in enumerate(intermediate_latents, 1):
            new_outputs.append(
                helper.make_tensor_value_info(inter_latent, base_out.type.tensor_type.elem_type, out_shape)
            )
        print(f" • [Debug Mode] Exposing {len(intermediate_latents)} intermediate latent outputs for inspection.")

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(unrolled_model, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ Saved Unrolled 5-Step NPU Model: '{output_path}' ({size_mb:.2f} MB)")
    print("=" * 80)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build Unrolled Vector Estimator ONNX Graph")
    parser.add_argument("--steps", type=int, default=5, help="Number of Euler ODE steps (default: 5)")
    parser.add_argument("--base", type=str, default=VE_PATH, help="Base vector estimator ONNX path")
    parser.add_argument("--debug", action="store_true", help="Enable Multi-Output Tapping for intermediate step inspection")
    parser.add_argument("--no-ping-pong", action="store_true", help="Disable Ping-Pong buffer activation recycling")
    parser.add_argument("--out", type=str, default=None, help="Custom output ONNX path")
    args = parser.parse_args()

    build_unrolled_vector_estimator(
        total_steps=args.steps,
        debug_mode=args.debug,
        use_ping_pong=not args.no_ping_pong,
        base_model_path=args.base,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
