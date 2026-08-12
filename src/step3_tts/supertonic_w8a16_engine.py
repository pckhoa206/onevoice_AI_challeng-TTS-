"""Direct W8A16 Supertonic 3 ONNX Engine for Local Machine Execution.

Loads and runs the 4 Qualcomm W8A16 Quantized ONNX Models directly on the local machine:
  - text_encoder_w8a16.bin.onnx (INT8 Weights, INT16 Activations)
  - duration_predictor_w8a16.bin.onnx (INT8 Weights, INT16 Activations)
  - vector_estimator_w8a16.bin.onnx (INT8 Weights, INT16 Activations)
  - vocoder_w8a16.bin.onnx (INT8 Weights, INT16 Activations)

Executes 100% W8A16 quantized ONNX inference with static shape formatting.
"""
import os
import sys
import time
import zipfile
import glob
import numpy as np
import soundfile as sf
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W8A16_ZIP_DIR = os.path.join(ROOT, "outputs", "qnn_binaries_w8a16")
EXTRACT_DIR = os.path.join(ROOT, "outputs", "w8a16_extracted_engine")


class SupertonicW8A16Engine:
    """End-to-End Supertonic 3 Engine running 100% W8A16 Quantized ONNX Models on Local Machine."""

    def __init__(self, w8a16_dir: str = W8A16_ZIP_DIR):
        self.w8a16_dir = w8a16_dir
        self.extract_dir = EXTRACT_DIR
        os.makedirs(self.extract_dir, exist_ok=True)
        
        self.submodel_names = ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]
        self.sessions = {}
        
        self._load_w8a16_models()
        self._init_helper_tts()

    def _load_w8a16_models(self):
        print("=" * 80)
        print(" ⚡ INITIALIZING LOCAL W8A16 SUPERTONIC 3 ENGINE (WEIGHT INT8, ACTIVATION INT16)")
        print("=" * 80)
        
        for name in self.submodel_names:
            zip_pattern = os.path.join(self.w8a16_dir, f"{name}_npu_w8a16*.zip")
            matches = glob.glob(zip_pattern)
            
            if not matches:
                raise FileNotFoundError(f"Missing W8A16 zip for {name} in {self.w8a16_dir}")
                
            zip_path = matches[0]
            model_extract_path = os.path.join(self.extract_dir, name)
            
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(model_extract_path)
                onnx_path = [os.path.join(model_extract_path, f) for f in zf.namelist() if f.endswith(".onnx")][0]
                
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self.sessions[name] = sess
            zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            print(f" • Loaded W8A16 Submodel [{name:<18}]: Package Size = {zip_size_mb:6.2f} MB | Status = READY (W8A16 ONNX)")
            
        print("=" * 80)

    def _init_helper_tts(self):
        from supertonic import TTS
        self._helper_tts = TTS(auto_download=True)

    def synthesize(
        self,
        text: str,
        lang: str = "en",
        total_steps: int = 5,
    ) -> Tuple[np.ndarray, dict]:
        """Synthesize text using 100% W8A16 Quantized ONNX Models."""
        t_start = time.time()
        
        lang_to_use = "en" if lang == "zh" else (lang if lang in ["vi", "en", "ko"] else "en")
        voice_name = "F1" if lang in ["vi", "zh"] else "M1"
        style = self._helper_tts.get_voice_style(voice_name=voice_name)
        
        # Tokenize text via Supertonic Unicode Processor
        raw_ids, raw_mask = self._helper_tts.model.text_processor([text], lang_to_use)
        
        # Format to W8A16 Static Shape (1, 64)
        seq_len = 64
        text_ids = np.zeros((1, seq_len), dtype=np.int64)
        valid = min(raw_ids.shape[1], seq_len)
        text_ids[0, :valid] = raw_ids[0, :valid]
        
        text_mask = np.zeros((1, 1, seq_len), dtype=np.float32)
        text_mask[0, 0, :valid] = 1.0
        
        style_ttl = np.asarray(style.ttl).astype(np.float32)
        style_dp = np.asarray(style.dp).astype(np.float32)

        # Step 1: W8A16 Text Encoder
        text_emb = self.sessions["text_encoder"].run(
            None, {"text_ids": text_ids, "style_ttl": style_ttl, "text_mask": text_mask}
        )[0]

        # Step 2: W8A16 Duration Predictor
        duration = self.sessions["duration_predictor"].run(
            None, {"text_ids": text_ids, "style_dp": style_dp, "text_mask": text_mask}
        )[0]

        # Step 3: W8A16 Vector Estimator Euler ODE Solver Loop
        latent = np.random.randn(1, 144, 100).astype(np.float32)
        latent_mask = np.ones((1, 1, 100), dtype=np.float32)
        dt = 1.0 / float(total_steps)
        
        for step_idx in range(1, total_steps + 1):
            v_est = self.sessions["vector_estimator"].run(
                None,
                {
                    "noisy_latent": latent,
                    "text_emb": text_emb,
                    "style_ttl": style_ttl,
                    "latent_mask": latent_mask,
                    "text_mask": text_mask,
                    "current_step": np.array([float(step_idx)], dtype=np.float32),
                    "total_step": np.array([float(total_steps)], dtype=np.float32),
                },
            )[0]
            latent = latent + dt * v_est

        # Step 4: W8A16 Neural Vocoder
        wav_out = self.sessions["vocoder"].run(None, {"latent": latent})[0]

        latency_ms = (time.time() - t_start) * 1000.0
        waveform = np.asarray(wav_out).squeeze().astype(np.float32)
        
        sample_rate = 44100
        duration_sec = len(waveform) / float(sample_rate)
        rtf_val = (latency_ms / 1000.0) / duration_sec if duration_sec > 0 else 0.0

        stats = {
            "latency_ms": round(latency_ms, 1),
            "rtf": round(rtf_val, 4),
            "duration_sec": round(duration_sec, 2),
            "sample_rate": sample_rate,
            "quantization": "W8A16 ONNX (Weight INT8, Activation INT16)",
        }

        return waveform, stats


def main():
    _ensure_utf8_stdout()
    engine = SupertonicW8A16Engine()
    
    test_text = "The OneVoice AI Challenge runs 100% W8A16 quantized Supertonic on Qualcomm Snapdragon NPU."
    waveform, stats = engine.synthesize(test_text, lang="en", total_steps=5)
    
    print(f"\n[SupertonicW8A16Engine Results]")
    print(f" • Input Text   : '{test_text}'")
    print(f" • Quantization : {stats['quantization']}")
    print(f" • Audio Sec    : {stats['duration_sec']}s ({waveform.shape[0]} samples)")
    print(f" • Latency TTFB : {stats['latency_ms']} ms")
    print(f" • Speed RTF    : {stats['rtf']} (Local Machine W8A16 Execution)")
    
    out_wav = os.path.join(ROOT, "outputs", "test_w8a16_local_synth.wav")
    sf.write(out_wav, waveform, stats["sample_rate"])
    print(f" • Saved Audio  : {out_wav}")


if __name__ == "__main__":
    main()
