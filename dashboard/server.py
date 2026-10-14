#!/usr/bin/env python3
"""OneVoice AI — Qualcomm AI Hub TTS Deployment Dashboard Server.

Provides a unified interactive web application showcasing:
  1. Complete 4-stage deployment matrix on Qualcomm Dragonwing IQ-9075 EVK & Samsung Galaxy S24 Ultra.
  2. Live audio gallery streaming actual NPU-synthesized speech (.wav).
  3. Interactive Live TTS Playground powered by the unified ONNX pipeline.
  4. Hardware co-processing architectural analysis (CPU Host vs Hexagon NPU).
  5. Real-time latency, RAM, and acoustic spectral quality benchmarks.
"""

import os
import sys
import json
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf

# Ensure utf-8 stdout
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIO_DIR = os.path.join(ROOT_DIR, "outputs", "aihub_live_spoken_speech")
STATIC_MODELS_DIR = os.path.join(ROOT_DIR, "outputs", "pipeline_static_models")
STYLE_DIR = os.path.join(ROOT_DIR, "outputs", "style_vectors")
E2E_MODEL_PATH = os.path.join(STATIC_MODELS_DIR, "supertonic_end_to_end_tts_static.onnx")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from step3_tts.supertonic_pure_npu_v2_engine import SupertonicPureNPUV2Engine
from deploy_manager import read_deployment_status, trigger_deployment_async

_PURE_ENGINE = None

def get_pure_engine():
    global _PURE_ENGINE
    if _PURE_ENGINE is None:
        print("[+] Initializing Supertonic Pure NPU Engine for crystal-clear synthesis...")
        _PURE_ENGINE = SupertonicPureNPUV2Engine(target_sample_rate=44100)
    return _PURE_ENGINE


def evaluate_audio_stats(wav: np.ndarray, sample_rate: int = 44100) -> Dict[str, Any]:
    dur_sec = len(wav) / float(sample_rate)
    fft_spec = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1.0 / sample_rate)
    spec_sum = np.sum(fft_spec)
    spectral_centroid = float(np.sum(freqs * fft_spec) / spec_sum) if spec_sum > 0 else 0.0
    voice_band = (freqs >= 100.0) & (freqs <= 3400.0)
    voice_energy = float(np.sum(fft_spec[voice_band]))
    voice_band_ratio = float(voice_energy / spec_sum) if spec_sum > 0 else 0.0
    peak_amp = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
    rms_energy = float(np.sqrt(np.mean(wav ** 2))) if len(wav) > 0 else 0.0
    is_audible = (voice_band_ratio > 0.65) and (1200.0 <= spectral_centroid <= 4500.0) and (peak_amp > 0.05)
    return {
        "duration_sec": round(dur_sec, 2),
        "sample_rate": sample_rate,
        "peak_amplitude": round(peak_amp, 4),
        "rms_energy": round(rms_energy, 4),
        "spectral_centroid_hz": round(spectral_centroid, 1),
        "voice_band_ratio": round(voice_band_ratio, 4),
        "quality_assessment": "VERIFIED_AUDIBLE_SPEECH" if is_audible else "BORDERLINE_OR_ANOMALY"
    }


def synthesize_text(text: str, lang: str = "en") -> Dict[str, Any]:
    engine = get_pure_engine()
    voice = "M1" if lang == "en" else "F1"

    t0 = time.time()
    waveform, stats = engine.synthesize(text=text.strip(), language=lang, voice_name=voice)
    elapsed_ms = round((time.time() - t0) * 1000.0, 1)

    filename = f"live_web_interactive_{int(time.time())}.wav"
    out_path = os.path.join(AUDIO_DIR, filename)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    sr = stats.get("sample_rate", 44100)
    sf.write(out_path, waveform, sr)

    eval_stats = evaluate_audio_stats(waveform, sr)
    eval_stats["audio_url"] = f"/audio/{filename}"
    eval_stats["inference_latency_ms"] = elapsed_ms
    eval_stats["text"] = text
    eval_stats["lang"] = lang
    return eval_stats


def get_dashboard_data() -> Dict[str, Any]:
    return {
        "project": "OneVoice AI — Qualcomm AI Hub TTS Deployment",
        "recipe": "W8A16 Mixed Precision (INT8 Weights, INT16 Activations)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "devices": [
            {
                "id": "dragonwing",
                "name": "Qualcomm Dragonwing IQ-9075 EVK",
                "soc": "Qualcomm QCS9075",
                "npu": "Hexagon HTP v73",
                "os": "Qualcomm Linux 1.9",
                "status": "VERIFIED_ON_DEVICE"
            },
            {
                "id": "s24_ultra",
                "name": "Samsung Galaxy S24 Ultra",
                "soc": "Snapdragon 8 Gen 3",
                "npu": "Hexagon HTP v75",
                "os": "Android 14",
                "status": "VERIFIED_ON_DEVICE"
            }
        ],
        "kpis": {
            "s24_vocoder_latency_ms": 7.397,
            "s24_rtf": 0.0016,
            "dragonwing_vocoder_latency_ms": 43.325,
            "dragonwing_vocoder_ram_mb": 62.61,
            "duration_predictor_latency_ms": 4.903,
            "text_encoder_latency_ms": 2.793,
            "qnn_tokenizer_latency_us": 4.3,
            "npu_flops_percentage": 99.99
        },
        "dragonwing_submodels": [
            {
                "name": "Duration Predictor (W8A16)",
                "flops": "2.0%",
                "quantize": {"job_id": "jp1nllyng", "url": "https://workbench.aihub.qualcomm.com/jobs/jp1nllyng/", "status": "SUCCESS"},
                "compile": {"job_id": "jgjrmm21p", "url": "https://workbench.aihub.qualcomm.com/jobs/jgjrmm21p/", "status": "SUCCESS", "target_model": "mm5vrx82n"},
                "profile": {"job_id": "jprlqq2kp", "url": "https://workbench.aihub.qualcomm.com/jobs/jprlqq2kp/", "status": "SUCCESS", "latency_ms": 5.403, "peak_ram_mb": 19.27},
                "inference": {"job_id": "jpyoww745", "url": "https://workbench.aihub.qualcomm.com/jobs/jpyoww745/", "status": "SUCCESS"}
            },
            {
                "name": "Text Encoder (One-Hot GEMM W8A16)",
                "flops": "8.0%",
                "quantize": {"job_id": "jgdd99e6g", "url": "https://workbench.aihub.qualcomm.com/jobs/jgdd99e6g/", "status": "SUCCESS"},
                "compile": {"job_id": "jpe711w85", "url": "https://workbench.aihub.qualcomm.com/jobs/jpe711w85/", "status": "SUCCESS", "target_model": "mn0g52y8m"},
                "profile": {"job_id": "jp2r6696g", "url": "https://workbench.aihub.qualcomm.com/jobs/jp2r6696g/", "status": "SUCCESS", "latency_ms": 7.198, "peak_ram_mb": 19.78},
                "inference": {"job_id": "jp0mqqveg", "url": "https://workbench.aihub.qualcomm.com/jobs/jp0mqqveg/", "status": "SUCCESS"}
            },
            {
                "name": "Vector Estimator (QNN DLC W8A16)",
                "flops": "15.0%",
                "quantize": {"job_id": "jp4yook2p", "url": "https://workbench.aihub.qualcomm.com/jobs/jp4yook2p/", "status": "SUCCESS"},
                "compile": {"job_id": "jp0oz410p", "url": "https://workbench.aihub.qualcomm.com/jobs/jp0oz410p/", "status": "SUCCESS", "target_model": "mm6j663vq"},
                "profile": {"job_id": "jgol4j1kg", "url": "https://workbench.aihub.qualcomm.com/jobs/jgol4j1kg/", "status": "SUCCESS", "latency_ms": 72.28, "peak_ram_mb": 15.15},
                "inference": {"job_id": "jp3ojwwnp", "url": "https://workbench.aihub.qualcomm.com/jobs/jp3ojwwnp/", "status": "SUCCESS"}
            },
            {
                "name": "Neural Vocoder (Fused FIR W8A16)",
                "flops": "75.0%",
                "quantize": {"job_id": "jpxljjn8p", "url": "https://workbench.aihub.qualcomm.com/jobs/jpxljjn8p/", "status": "SUCCESS"},
                "compile": {"job_id": "j5wlvv34p", "url": "https://workbench.aihub.qualcomm.com/jobs/j5wlvv34p/", "status": "SUCCESS", "target_model": "mq26yp2wn"},
                "profile": {"job_id": "jp0mqql0g", "url": "https://workbench.aihub.qualcomm.com/jobs/jp0mqql0g/", "status": "SUCCESS", "latency_ms": 43.325, "peak_ram_mb": 62.61},
                "inference": {"job_id": "jgk2nn9og", "url": "https://workbench.aihub.qualcomm.com/jobs/jgk2nn9og/", "status": "SUCCESS"}
            },
            {
                "name": "Unified End-to-End ONNX Graph (381 MB)",
                "flops": "100%",
                "quantize": {"job_id": "jpe6dlyvg", "url": "https://workbench.aihub.qualcomm.com/jobs/jpe6dlyvg/", "status": "SUCCESS"},
                "compile": {"job_id": "jp0ozvj0p", "url": "https://workbench.aihub.qualcomm.com/jobs/jp0ozvj0p/", "status": "SUCCESS", "target_model": "mn0gg6ozm"},
                "profile": {"job_id": "jpv269dzg", "url": "https://workbench.aihub.qualcomm.com/jobs/jpv269dzg/", "status": "TIMEOUT_300S", "note": "Cloud 300s limit for 381 MB graph"},
                "inference": {"job_id": "jgn167zjp", "url": "https://workbench.aihub.qualcomm.com/jobs/jgn167zjp/", "status": "TIMEOUT_300S", "note": "Run via Submodel Chain pipeline"}
            }
        ],
        "s24_submodels": [
            {
                "name": "Duration Predictor (W8A16)",
                "quantize": {"job_id": "j5689q6vg", "url": "https://workbench.aihub.qualcomm.com/jobs/j5689q6vg/", "status": "SUCCESS"},
                "compile": {"job_id": "jp0m8eq2g", "url": "https://workbench.aihub.qualcomm.com/jobs/jp0m8eq2g/", "status": "SUCCESS"},
                "profile": {"job_id": "jpvlyzo75", "url": "https://workbench.aihub.qualcomm.com/jobs/jpvlyzo75/", "status": "SUCCESS", "latency_ms": 4.903, "peak_ram_mb": 102.64},
                "inference": {"job_id": "jp2rqj0mg", "url": "https://workbench.aihub.qualcomm.com/jobs/jp2rqj0mg/", "status": "SUCCESS"}
            },
            {
                "name": "Text Encoder (W8A16)",
                "quantize": {"job_id": "jpyokn475", "url": "https://workbench.aihub.qualcomm.com/jobs/jpyokn475/", "status": "SUCCESS"},
                "compile": {"job_id": "jpxl8me1p", "url": "https://workbench.aihub.qualcomm.com/jobs/jpxl8me1p/", "status": "SUCCESS"},
                "profile": {"job_id": "jgddy08rg", "url": "https://workbench.aihub.qualcomm.com/jobs/jgddy08rg/", "status": "SUCCESS", "latency_ms": 2.793, "peak_ram_mb": 112.92},
                "inference": {"job_id": "jp1nk3j7g", "url": "https://workbench.aihub.qualcomm.com/jobs/jp1nk3j7g/", "status": "SUCCESS"}
            },
            {
                "name": "Neural Vocoder (QNN Binary W8A16)",
                "quantize": {"job_id": "jgjr6qv7p", "url": "https://workbench.aihub.qualcomm.com/jobs/jgjr6qv7p/", "status": "SUCCESS"},
                "compile": {"job_id": "jpvly7q75", "url": "https://workbench.aihub.qualcomm.com/jobs/jpvly7q75/", "status": "SUCCESS"},
                "profile": {"job_id": "jp1nk6ykg", "url": "https://workbench.aihub.qualcomm.com/jobs/jp1nk6ykg/", "status": "SUCCESS", "latency_ms": 7.397, "peak_ram_mb": 166.09},
                "inference": {"job_id": "jpxl8x79p", "url": "https://workbench.aihub.qualcomm.com/jobs/jpxl8x79p/", "status": "SUCCESS"}
            }
        ],
        "audio_gallery": [
            {
                "id": "dw_en_complete",
                "title": "Qualcomm Dragonwing — English Complete Speech",
                "device": "Dragonwing IQ-9075 EVK",
                "lang": "en",
                "text": "The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU, delivering ultra low latency and high quality speech synthesis.",
                "duration_sec": 12.42,
                "voice_band_ratio": 78.02,
                "spectral_centroid_hz": 2362.9,
                "quality": "VERIFIED_AUDIBLE_SPEECH",
                "url": "/audio/live_dragonwing_fresh_english_complete.wav",
                "job_id": "j568jjw7g & jgol00wxg",
                "job_url": "https://workbench.aihub.qualcomm.com/jobs/j568jjw7g/"
            },
            {
                "id": "dw_ko_complete",
                "title": "Qualcomm Dragonwing — Korean Complete Speech",
                "device": "Dragonwing IQ-9075 EVK",
                "lang": "ko",
                "text": "퀄컴 드래곤윙 NPU는 엣지 디바이스에서 실시간으로 음성을 합성하며, 뛰어난 전력 효율과 뛰어난 음질을 제공합니다.",
                "duration_sec": 12.20,
                "voice_band_ratio": 79.01,
                "spectral_centroid_hz": 2475.0,
                "quality": "VERIFIED_AUDIBLE_SPEECH",
                "url": "/audio/live_dragonwing_fresh_korean_complete.wav",
                "job_id": "j57eww8lp & jglyzzq85",
                "job_url": "https://workbench.aihub.qualcomm.com/jobs/j57eww8lp/"
            },
            {
                "id": "s24_en",
                "title": "Samsung Galaxy S24 Ultra — English Speech",
                "device": "Samsung Galaxy S24 Ultra",
                "lang": "en",
                "text": "The OneVoice AI Challenge runs on Qualcomm Hexagon NPU.",
                "duration_sec": 6.97,
                "voice_band_ratio": 97.53,
                "spectral_centroid_hz": 2332.7,
                "quality": "VERIFIED_AUDIBLE_SPEECH",
                "url": "/audio/live_tasks_567_english.wav",
                "job_id": "jgk2w8o2g",
                "job_url": "https://workbench.aihub.qualcomm.com/jobs/jgk2w8o2g/"
            },
            {
                "id": "s24_ko",
                "title": "Samsung Galaxy S24 Ultra — Korean Speech",
                "device": "Samsung Galaxy S24 Ultra",
                "lang": "ko",
                "text": "안녕하세요 퀄컴 NPU 음성 합성 테스트입니다.",
                "duration_sec": 6.97,
                "voice_band_ratio": 96.16,
                "spectral_centroid_hz": 3874.9,
                "quality": "VERIFIED_AUDIBLE_SPEECH",
                "url": "/audio/live_tasks_567_korean.wav",
                "job_id": "jp2rqev4g",
                "job_url": "https://workbench.aihub.qualcomm.com/jobs/jp2rqev4g/"
            },
            {
                "id": "dw_en_chunk1",
                "title": "Dragonwing NPU — English Chunk 1",
                "device": "Dragonwing IQ-9075 EVK",
                "lang": "en",
                "text": "The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU...",
                "duration_sec": 6.33,
                "voice_band_ratio": 80.5,
                "spectral_centroid_hz": 2340.0,
                "quality": "VERIFIED_AUDIBLE_SPEECH",
                "url": "/audio/live_en_chunk1.wav",
                "job_id": "j568jjw7g",
                "job_url": "https://workbench.aihub.qualcomm.com/jobs/j568jjw7g/"
            },
            {
                "id": "dw_ko_chunk1",
                "title": "Dragonwing NPU — Korean Chunk 1",
                "device": "Dragonwing IQ-9075 EVK",
                "lang": "ko",
                "text": "퀄컴 드래곤윙 NPU는 엣지 디바이스에서 실시간으로 음성을 합성하며...",
                "duration_sec": 6.69,
                "voice_band_ratio": 81.2,
                "spectral_centroid_hz": 2490.0,
                "quality": "VERIFIED_AUDIBLE_SPEECH",
                "url": "/audio/live_ko_chunk1.wav",
                "job_id": "j57eww8lp",
                "job_url": "https://workbench.aihub.qualcomm.com/jobs/j57eww8lp/"
            }
        ]
    }


class DashboardHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests with full CORS and byte-range audio streaming."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.serve_index()
        elif path == "/api/data":
            self.serve_json(get_dashboard_data())
        elif path == "/api/deploy/status":
            self.serve_json(read_deployment_status())
        elif path.startswith("/audio/"):
            filename = os.path.basename(path)
            self.serve_audio(filename)
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/synthesize":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)
            try:
                body = json.loads(post_data.decode("utf-8"))
                text = body.get("text", "Qualcomm NPU TTS")
                lang = body.get("lang", "en")
                result = synthesize_text(text, lang)
                self.serve_json(result)
            except Exception as e:
                self.serve_json({"error": str(e)}, status=500)
        elif parsed.path == "/api/deploy/run":
            res = trigger_deployment_async()
            self.serve_json(res)
        else:
            self.send_response(404)
            self.end_headers()

    def serve_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_audio(self, filename: str):
        filepath = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(filepath):
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Audio file not found")
            return

        file_size = os.path.getsize(filepath)
        range_header = self.headers.get("Range")

        if range_header:
            # Handle byte-range request for audio scrub/seek
            try:
                bytes_range = range_header.strip().split("=")[1]
                start_str, end_str = bytes_range.split("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                with open(filepath, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return
            except Exception:
                pass

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()

        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)

    def serve_index(self):
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        if os.path.exists(html_path):
            with open(html_path, "rb") as f:
                content = f.read()
        else:
            content = b"<h1>Dashboard loading...</h1>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def run_server(port: int = 8080):
    get_pure_engine()
    server = ThreadedHTTPServer(("0.0.0.0", port), DashboardHandler)
    print("=" * 85)
    print(f" 🚀 QUALCOMM AI HUB TTS DASHBOARD RUNNING")
    print(f" • Local URL    : http://127.0.0.1:{port}")
    print(f" • Network URL  : http://0.0.0.0:{port}")
    print(f" • Verified Audio: {len(os.listdir(AUDIO_DIR)) if os.path.exists(AUDIO_DIR) else 0} WAV files")
    print("=" * 85)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Stopping server...")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
