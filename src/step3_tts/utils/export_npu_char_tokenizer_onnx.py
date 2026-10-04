"""Export Static Character-to-Token ONNX Subgraph for Qualcomm Hexagon NPU.

Transforms raw character codepoints into exact [1, 64] text_ids tensor:
  - English Mode (lang_id = 0): Input char_codes [1, 54] -> W_ascii lookup -> Wrap <en> ... .</en> -> [1, 64]
  - Korean Mode  (lang_id = 1): Input char_codes [1, 54] (18 Hangul chars) -> W_hangul lookup -> 18x3 Jamo -> Wrap <ko> ... .</ko> -> [1, 64]

100% Static Dimensions, Zero Dynamic Loops, Pure ONNX Standard Operators.
Valid for Qualcomm Hexagon HTP v73 / v75 NPU.
"""

import os
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine

OUTPUT_DIR = os.path.join(ROOT, "outputs", "pipeline_static_models")


def build_char_tokenizer_model() -> onnx.ModelProto:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    engine = SupertonicPureNPUV2Engine()
    indexer = engine._helper_tts.model.text_processor.indexer

    # 1. Build lookup tables
    w_ascii = np.zeros(128, dtype=np.int64)
    for cp in range(128):
        tok = indexer[cp] if cp < len(indexer) and indexer[cp] != -1 else 0
        w_ascii[cp] = max(0, tok)

    w_hangul = np.zeros((11172, 3), dtype=np.int64)
    for cp in range(0xAC00, 0xD7A4):
        s_idx = cp - 0xAC00
        l_idx = s_idx // 588
        v_idx = (s_idx % 588) // 28
        t_idx = s_idx % 28
        tok_l = indexer[0x1100 + l_idx] if (0x1100 + l_idx) < len(indexer) else 0
        tok_v = indexer[0x1161 + v_idx] if (0x1161 + v_idx) < len(indexer) else 0
        tok_t = indexer[0x11A7 + t_idx] if t_idx > 0 and (0x11A7 + t_idx) < len(indexer) else 0
        w_hangul[s_idx, 0] = max(0, tok_l)
        w_hangul[s_idx, 1] = max(0, tok_v)
        w_hangul[s_idx, 2] = max(0, tok_t)

    # 2. Define Inputs & Outputs
    inp_char_codes = helper.make_tensor_value_info("char_codes", TensorProto.INT64, [1, 54])
    inp_lang_id = helper.make_tensor_value_info("lang_id", TensorProto.INT64, [1])
    out_text_ids = helper.make_tensor_value_info("text_ids", TensorProto.INT64, [1, 64])

    # 3. Create Constants (Initializers)
    init_w_ascii = numpy_helper.from_array(w_ascii, name="W_ascii")
    init_w_hangul = numpy_helper.from_array(w_hangul, name="W_hangul")

    tag_pre_en = np.array([[29, 64, 73, 31]], dtype=np.int64)
    tag_pre_ko = np.array([[29, 70, 74, 31]], dtype=np.int64)
    tag_post_en = np.array([[15, 29, 16, 64, 73, 31]], dtype=np.int64)
    tag_post_ko = np.array([[15, 29, 16, 70, 74, 31]], dtype=np.int64)

    init_tag_pre_en = numpy_helper.from_array(tag_pre_en, name="tag_pre_en")
    init_tag_pre_ko = numpy_helper.from_array(tag_pre_ko, name="tag_pre_ko")
    init_tag_post_en = numpy_helper.from_array(tag_post_en, name="tag_post_en")
    init_tag_post_ko = numpy_helper.from_array(tag_post_ko, name="tag_post_ko")

    init_const_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="const_1")
    init_const_0xac00 = numpy_helper.from_array(np.array([44032], dtype=np.int64), name="const_0xac00")
    init_clip_ascii_min = numpy_helper.from_array(np.array(0, dtype=np.int64), name="clip_ascii_min")
    init_clip_ascii_max = numpy_helper.from_array(np.array(127, dtype=np.int64), name="clip_ascii_max")
    init_clip_hangul_min = numpy_helper.from_array(np.array(0, dtype=np.int64), name="clip_hangul_min")
    init_clip_hangul_max = numpy_helper.from_array(np.array(11171, dtype=np.int64), name="clip_hangul_max")

    init_slice_starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="slice_starts")
    init_slice_ends = numpy_helper.from_array(np.array([18], dtype=np.int64), name="slice_ends")
    init_slice_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="slice_axes")

    init_shape_1_54 = numpy_helper.from_array(np.array([1, 54], dtype=np.int64), name="shape_1_54")

    initializers = [
        init_w_ascii, init_w_hangul,
        init_tag_pre_en, init_tag_pre_ko, init_tag_post_en, init_tag_post_ko,
        init_const_1, init_const_0xac00,
        init_clip_ascii_min, init_clip_ascii_max,
        init_clip_hangul_min, init_clip_hangul_max,
        init_slice_starts, init_slice_ends, init_slice_axes,
        init_shape_1_54
    ]

    # 4. Construct Graph Nodes
    nodes = []

    # English Path:
    # clamp_ascii = Clip(char_codes, 0, 127)
    nodes.append(helper.make_node("Clip", ["char_codes", "clip_ascii_min", "clip_ascii_max"], ["clamp_ascii"]))
    # content_en = Gather(W_ascii, clamp_ascii, axis=0) -> [1, 54]
    nodes.append(helper.make_node("Gather", ["W_ascii", "clamp_ascii"], ["content_en"], axis=0))

    # Korean Path:
    # slice_18 = Slice(char_codes, starts=[0], ends=[18], axes=[1]) -> [1, 18]
    nodes.append(helper.make_node("Slice", ["char_codes", "slice_starts", "slice_ends", "slice_axes"], ["slice_18"]))
    # hangul_raw_offset = Sub(slice_18, const_0xac00)
    nodes.append(helper.make_node("Sub", ["slice_18", "const_0xac00"], ["hangul_raw_offset"]))
    # hangul_clamped = Clip(hangul_raw_offset, 0, 11171)
    nodes.append(helper.make_node("Clip", ["hangul_raw_offset", "clip_hangul_min", "clip_hangul_max"], ["hangul_clamped"]))
    # hangul_jamo_3d = Gather(W_hangul, hangul_clamped, axis=0) -> [1, 18, 3]
    nodes.append(helper.make_node("Gather", ["W_hangul", "hangul_clamped"], ["hangul_jamo_3d"], axis=0))
    # content_ko = Reshape(hangul_jamo_3d, shape_1_54) -> [1, 54]
    nodes.append(helper.make_node("Reshape", ["hangul_jamo_3d", "shape_1_54"], ["content_ko"]))

    # Language Condition:
    # is_korean = Equal(lang_id, const_1) -> [1] (broadcasts to [1, 54])
    nodes.append(helper.make_node("Equal", ["lang_id", "const_1"], ["is_korean"]))

    # Select Content & Tags based on Language:
    # content_tokens = Where(is_korean, content_ko, content_en) -> [1, 54]
    nodes.append(helper.make_node("Where", ["is_korean", "content_ko", "content_en"], ["content_tokens"]))
    # tag_pre = Where(is_korean, tag_pre_ko, tag_pre_en) -> [1, 4]
    nodes.append(helper.make_node("Where", ["is_korean", "tag_pre_ko", "tag_pre_en"], ["tag_pre"]))
    # tag_post = Where(is_korean, tag_post_ko, tag_post_en) -> [1, 6]
    nodes.append(helper.make_node("Where", ["is_korean", "tag_post_ko", "tag_post_en"], ["tag_post"]))

    # Final Concatenation:
    # text_ids = Concat([tag_pre, content_tokens, tag_post], axis=1) -> [1, 64]
    nodes.append(helper.make_node("Concat", ["tag_pre", "content_tokens", "tag_post"], ["text_ids"], axis=1))

    # 5. Build Graph & Model
    graph = helper.make_graph(
        nodes=nodes,
        name="char_tokenizer_static_graph",
        inputs=[inp_char_codes, inp_lang_id],
        outputs=[out_text_ids],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="OneVoice_AI_QNN_Static_Tokenizer", opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


def main():
    print("=" * 80)
    print(" 🚀 EXPORTING QUALCOMM NPU STATIC CHAR TOKENIZER ONNX SUBGRAPH")
    print("=" * 80)
    model = build_char_tokenizer_model()
    out_path = os.path.join(OUTPUT_DIR, "char_tokenizer_static.onnx")
    onnx.save(model, out_path)
    file_kb = os.path.getsize(out_path) / 1024.0

    print(f" ✅ Model Successfully Created & Checked with ONNX Validator!")
    print(f" • File Path : {os.path.relpath(out_path, ROOT)}")
    print(f" • File Size : {file_kb:.2f} KB (Ultra-Lightweight for NPU SRAM)")
    print(f" • Inputs    : char_codes [1, 54], lang_id [1]")
    print(f" • Output    : text_ids [1, 64] (Int64)")
    print("=" * 80)


if __name__ == "__main__":
    main()
