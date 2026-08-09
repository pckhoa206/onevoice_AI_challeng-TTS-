# Step 3 — TTS (Speech Synthesis)

Full analysis, all 5 candidates tested with real RTF + round-trip WER/CER: [`../../step3.md`](../../step3.md)

**Picks:** Piper (Vietnamese) + Supertonic (Korean+English) + MeloTTS-ZH (Chinese).

## Architecture

The unified TTS pipeline consists of:
- `text_normalizer.py`: 4-language Text Normalization (Vi, En, Zh, Ko) handling numbers, dates, currency, acronyms (VNG, Qualcomm, USD), and special characters.
- `warmup_worker.py`: Thread-safe background initialization to pre-allocate NPU/RAM graph memory and eliminate cold-start delay.
- `tts_manager.py`: Central `UnifiedTTSManager` router providing standardized **16,000 Hz mono 16-bit PCM** output, fast polyphase FIR audio resampling (`scipy.signal.resample_poly`), peak audio normalization, and streaming clause chunking.
- `profile_qnn.py`: Remote profiling tool for submitting ONNX models to **Qualcomm AI Hub (`qai-hub`)** on Snapdragon NPU hardware.

## Setup

Each engine has its own install quirk — see `../../requirements.txt` for the full list. The two that need special handling:

```bash
# Piper: pip install piper-tts, then download each voice once before first run
python -m piper.download_voices vi_VN-vais1000-medium
python -m piper.download_voices en_US-amy-medium

# VieNeu-TTS (backup): mode="standard" needs neucodec + a torchao version that matches your torch build
pip install vieneu neucodec "torchao==0.9.0"

# MeloTTS is not on PyPI:
git clone https://github.com/myshell-ai/MeloTTS && cd MeloTTS && pip install -e .
```

## Run

```bash
# Run Unified TTS Manager integration tests (all 4 languages):
python test_unified_tts.py

# Run individual TTS engine tests:
python test_tts_piper.py        # Vi+En
python test_tts_supertonic.py   # Ko+En+Vi
python test_tts_melotts.py      # Zh+En
python test_tts_vieneu.py       # Vi (backup)

# Profile models on Qualcomm AI Hub (Qualcomm Snapdragon NPU):
python profile_qnn.py --model vi_VN-vais1000-medium.onnx --device "Snapdragon 8 Elite QCP"

# Round-trip quality eval: re-transcribes generated WAVs through ASR and scores WER/CER
python test_tts_eval_quality.py
```

Results -> `outputs/tts_unified_results.csv` and `outputs/tts_quality_results.csv`.

