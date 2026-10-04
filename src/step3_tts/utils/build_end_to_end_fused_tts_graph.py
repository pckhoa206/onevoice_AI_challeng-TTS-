"""Build Single Unified End-to-End ONNX Graph for Qualcomm Hexagon NPU TTS.

Fuses:
  1. char_tokenizer_static (char_codes [1, 54], lang_id [1] -> text_ids [1, 64])
  2. duration_predictor_static (text_ids [1, 64], style_dp [1, 8, 16] -> duration [1])
  3. text_encoder_static (text_ids [1, 64], style_ttl [1, 50, 256] -> text_emb [1, 256, 64])
  4. vector_estimator_static (text_emb [1, 256, 64], style_ttl [1, 50, 256] -> denoised_latent [1, 144, 100])
  5. vocoder_static (denoised_latent [1, 144, 100] -> wav_tts [1, 307200])

Final End-to-End Interface:
  Inputs : char_codes [1, 54], lang_id [1], style_dp [1, 8, 16], style_ttl [1, 50, 256]
  Outputs: wav_tts [1, 307200], duration [1]

Text In -> Speech Out in 1 single forward pass!
"""

import os
import sys
import copy
import numpy as np
import onnx
from onnx import helper, numpy_helper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

STATIC_DIR = os.path.join(ROOT, "outputs", "pipeline_static_models")
OUTPUT_PATH = os.path.join(STATIC_DIR, "supertonic_end_to_end_tts_static.onnx")


def load_model(name: str) -> onnx.ModelProto:
    path = os.path.join(STATIC_DIR, f"{name}.onnx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing static model: {path}")
    return onnx.load(path, load_external_data=False)


def build_end_to_end_model() -> onnx.ModelProto:
    print("=" * 85)
    print(" 🚀 FUSING ALL 5 SUBMODELS INTO SINGLE END-TO-END TTS GRAPH")
    print("=" * 85)

    m_tok = load_model("char_tokenizer_static")
    m_dp  = load_model("duration_predictor_static")
    m_te  = load_model("text_encoder_static")
    m_ve  = load_model("vector_estimator_static")
    m_voc = load_model("vocoder_static")

    # Rename vocoder's input 'latent' to 'denoised_latent'
    for node in m_voc.graph.node:
        for idx, inp in enumerate(node.input):
            if inp == "latent":
                node.input[idx] = "denoised_latent"

    submodels = [
        ("tok", m_tok),
        ("dp",  m_dp),
        ("te",  m_te),
        ("ve",  m_ve),
        ("voc", m_voc),
    ]

    # Interface tensors that must NOT be prefixed
    SHARED_TENSORS = {
        # Graph External Inputs
        "char_codes", "lang_id", "style_dp", "style_ttl",
        # Internal Interface Links
        "text_ids", "text_emb", "denoised_latent",
        # Graph External Outputs
        "wav_tts", "duration"
    }

    all_nodes = []
    all_initializers = []
    init_name_set = set()

    for prefix, model in submodels:
        print(f" • Processing Submodel [{prefix}]: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")
        
        # Mapping from old tensor name to prefixed tensor name
        tensor_map = {}

        # 1. Map initializers
        for init in model.graph.initializer:
            if init.name not in SHARED_TENSORS:
                new_init_name = f"{prefix}_{init.name}"
                tensor_map[init.name] = new_init_name
                init.name = new_init_name

            if init.name not in init_name_set:
                all_initializers.append(init)
                init_name_set.add(init.name)

        # 2. Map nodes
        for node_idx, node in enumerate(model.graph.node):
            node.name = f"{prefix}_{node.name}" if node.name else f"{prefix}_{node.op_type}_{node_idx}"
            for i, inp in enumerate(node.input):
                if not inp:
                    continue
                if inp in tensor_map:
                    node.input[i] = tensor_map[inp]
                elif inp not in SHARED_TENSORS and not inp.startswith(f"{prefix}_"):
                    new_inp = f"{prefix}_{inp}"
                    tensor_map[inp] = new_inp
                    node.input[i] = new_inp

            for o, out in enumerate(node.output):
                if not out:
                    continue
                if out in tensor_map:
                    node.output[o] = tensor_map[out]
                elif out not in SHARED_TENSORS and not out.startswith(f"{prefix}_"):
                    new_out = f"{prefix}_{out}"
                    tensor_map[out] = new_out
                    node.output[o] = new_out

            all_nodes.append(node)

    # 3. Define Graph Inputs
    inp_char_codes = helper.make_tensor_value_info("char_codes", onnx.TensorProto.INT64, [1, 54])
    inp_lang_id    = helper.make_tensor_value_info("lang_id",    onnx.TensorProto.INT64, [1])
    inp_style_dp   = helper.make_tensor_value_info("style_dp",   onnx.TensorProto.FLOAT, [1, 8, 16])
    inp_style_ttl  = helper.make_tensor_value_info("style_ttl",  onnx.TensorProto.FLOAT, [1, 50, 256])

    # 4. Define Graph Outputs
    out_wav_tts  = helper.make_tensor_value_info("wav_tts",  onnx.TensorProto.FLOAT, [1, 307200])
    out_duration = helper.make_tensor_value_info("duration", onnx.TensorProto.FLOAT, [1])

    # 5. Build Unified Graph
    fused_graph = helper.make_graph(
        nodes=all_nodes,
        name="supertonic_end_to_end_tts_graph",
        inputs=[inp_char_codes, inp_lang_id, inp_style_dp, inp_style_ttl],
        outputs=[out_wav_tts, out_duration],
        initializer=all_initializers,
    )

    fused_model = helper.make_model(
        fused_graph,
        producer_name="OneVoice_AI_End_to_End_TTS",
        opset_imports=[helper.make_opsetid("", 17)],
    )

    print("\n[+] Validating Unified Model with ONNX Checker...")
    onnx.checker.check_model(fused_model)
    print(" ✅ Validation Passed: No Topological or Schema Errors!")
    return fused_model


def main():
    model = build_end_to_end_model()
    print(f"\n[+] Saving Unified Model to: {OUTPUT_PATH}...")
    onnx.save(model, OUTPUT_PATH)
    file_mb = os.path.getsize(OUTPUT_PATH) / 1024.0 / 1024.0
    print(f" ✅ Saved End-to-End TTS Model: {file_mb:.2f} MB")
    print("=" * 85)


if __name__ == "__main__":
    main()
