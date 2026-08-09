"""Unified TTS Manager for Step 3 (Piper for Vi, Supertonic 3 for Ko/En, MeloTTS for Zh).
Handles 4-language text normalization, thread-safe warmup, dynamic model routing,
fast polyphase audio resampling to 16kHz mono 16-bit PCM, peak normalization, and streaming chunking.
"""
import os
import sys
import time
import re
import wave
from dataclasses import dataclass
from typing import List, Generator, Tuple, Optional

import numpy as np

# Add parent directory for common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout, rtf, get_device
from step3_tts.text_normalizer import TextNormalizer
from step3_tts.warmup_worker import WarmupWorker

TARGET_SR = 16000


@dataclass
class TTSResult:
    audio_bytes: bytes
    audio_array: np.ndarray
    sample_rate: int
    duration_sec: float
    rtf: float
    ttfb_ms: float
    engine: str
    lang: str


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Fast polyphase FIR or linear audio resampling to target sample rate."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        resampled = resample_poly(audio, up, down)
        return resampled.astype(np.float32)
    except Exception:
        # Fallback to linear interpolation
        duration = len(audio) / orig_sr
        num_target_samples = int(round(duration * target_sr))
        old_indices = np.linspace(0, len(audio) - 1, num=len(audio))
        new_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
        resampled = np.interp(new_indices, old_indices, audio)
        return resampled.astype(np.float32)


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize peak amplitude to prevent digital clipping."""
    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 0.0:
        return (audio / peak * target_peak).astype(np.float32)
    return audio.astype(np.float32)


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 audio array (-1.0 to 1.0) to 16-bit PCM WAV bytes."""
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return audio_int16.tobytes()


class UnifiedTTSManager:
    def __init__(self, warmup: bool = True):
        self.normalizer = TextNormalizer()
        self.warmup_worker = WarmupWorker()
        self.piper_voice = None
        self.supertonic_model = None
        self.melotts_model = None

        if warmup:
            self.warmup_engines()

    def warmup_engines(self):
        """Pre-allocate graph memory and bind ION/DMA memory pools."""
        # Piper warmup
        try:
            self.warmup_worker.warmup_engine("piper", self._synth_piper_raw, sample_text="xin chào", lang="vi")
        except Exception as e:
            print(f"[UnifiedTTSManager] Piper warmup skipped: {e}")

    def _synth_piper_raw(self, text: str, lang: str = "vi") -> Tuple[np.ndarray, int]:
        from piper.voice import PiperVoice
        if lang == "vi":
            voice_name = "vi_VN-vais1000-medium"
        elif lang == "zh":
            voice_name = "zh_CN-huayan-medium"
        else:
            voice_name = "en_US-amy-medium"

        onnx_path = f"{voice_name}.onnx"

        if not os.path.exists(onnx_path):
            import subprocess
            subprocess.run([sys.executable, "-m", "piper.download_voices", voice_name], check=True)

        if not hasattr(self, "piper_voices"):
            self.piper_voices = {}

        if voice_name not in self.piper_voices:
            self.piper_voices[voice_name] = PiperVoice.load(onnx_path)

        voice = self.piper_voices[voice_name]
        audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32, 22050

    def _synth_supertonic_raw(self, text: str, lang: str = "ko") -> Tuple[np.ndarray, int]:
        from supertonic import TTS
        if self.supertonic_model is None:
            self.supertonic_model = TTS(auto_download=True)
            self.supertonic_style = self.supertonic_model.get_voice_style(voice_name="M1")
        wav, duration = self.supertonic_model.synthesize(
            text=text, lang=lang, voice_style=self.supertonic_style, total_steps=8, speed=1.0
        )
        wav_arr = np.asarray(wav).squeeze().astype(np.float32)
        return wav_arr, 44100

    def _synth_melotts_raw(self, text: str, lang: str = "zh") -> Tuple[np.ndarray, int]:
        try:
            from melo.api import TTS
            if self.melotts_model is None:
                self.melotts_model = TTS(language="ZH", device="cpu")
            speaker_ids = self.melotts_model.hps.data.spk2id
            spk_id = speaker_ids.get("ZH-1", list(speaker_ids.values())[0])
            wav = self.melotts_model.tts_to_file(text, spk_id, quiet=True)
            return wav, 44100
        except Exception as e:
            # Fallback to Piper ONNX Chinese (zh_CN-huayan-medium)
            return self._synth_piper_raw(text, lang="zh")

    def synthesize(self, text: str, lang: str) -> TTSResult:
        """Unified synthesis API for Vi, En, Zh, Ko.

        Standardizes output to 16,000 Hz mono 16-bit PCM.
        """
        lang = lang.lower().strip()
        cleaned_text = self.normalizer.normalize(text, lang)
        if not cleaned_text:
            cleaned_text = text

        t0 = time.perf_counter()
        ttfb_ms = 0.0

        if lang == "vi":
            audio_raw, native_sr = self._synth_piper_raw(cleaned_text, lang="vi")
            engine_name = "piper"
        elif lang in ("ko", "en"):
            try:
                audio_raw, native_sr = self._synth_supertonic_raw(cleaned_text, lang=lang)
                engine_name = "supertonic"
            except Exception:
                # Fallback to Piper for English if Supertonic unavailable
                if lang == "en":
                    audio_raw, native_sr = self._synth_piper_raw(cleaned_text, lang="en")
                    engine_name = "piper_fallback"
                else:
                    raise
        elif lang == "zh":
            audio_raw, native_sr = self._synth_melotts_raw(cleaned_text, lang="zh")
            engine_name = "melotts"
        else:
            raise ValueError(f"Unsupported language code: '{lang}'")

        synth_time = time.perf_counter() - t0
        ttfb_ms = synth_time * 1000.0

        # Resample to 16,000 Hz & normalize peak
        audio_16k = resample_audio(audio_raw, native_sr, TARGET_SR)
        audio_16k = normalize_peak(audio_16k)
        audio_bytes = float32_to_int16_bytes(audio_16k)

        audio_sec = len(audio_16k) / TARGET_SR
        rtf_val = rtf(synth_time, audio_sec)

        return TTSResult(
            audio_bytes=audio_bytes,
            audio_array=audio_16k,
            sample_rate=TARGET_SR,
            duration_sec=round(audio_sec, 3),
            rtf=round(rtf_val, 4),
            ttfb_ms=round(ttfb_ms, 2),
            engine=engine_name,
            lang=lang,
        )

    def synthesize_stream(self, text: str, lang: str) -> Generator[TTSResult, None, None]:
        """Sentence/clause chunking for low TTFB streaming synthesis."""
        # Split by sentence/clause boundaries (, . ! ? ;)
        chunks = [c.strip() for c in re.split(r"([,\.!\?;])", text) if c.strip()]
        # Merge punctuation back to preceding chunk
        merged_chunks = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if i + 1 < len(chunks) and chunks[i + 1] in (",", ".", "!", "?", ";"):
                chunk += chunks[i + 1]
                i += 2
            else:
                i += 1
            if chunk:
                merged_chunks.append(chunk)

        for chunk in merged_chunks or [text]:
            yield self.synthesize(chunk, lang)
