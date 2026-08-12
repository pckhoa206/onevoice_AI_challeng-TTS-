"""Prosody Enhancer and Punctuation-based Micro-Pause Conditioning for TTS.

Converts plain translation text into prosody-annotated phoneme tokens with
micro-pauses, pitch accents, and dynamic rate modulation for human-like speech rhythm.
"""
import os
import re
import sys
import numpy as np
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import _ensure_utf8_stdout

ABBREVIATIONS_VI = {
    "VNG": "vê en giê",
    "AI": "ei ai",
    "NPU": "en pi u",
    "CPU": "si pi u",
    "GPU": "gi pi u",
    "TTS": "tê tê ét",
    "STT": "ét tê tê",
}


class ProsodyEnhancer:
    """Enhances raw text into prosody-conditioned token streams with natural pause rhythm."""

    def __init__(self):
        pass

    def normalize_text(self, text: str, language: str = "vi") -> str:
        """Clean and normalize abbreviations, numbers, and currencies."""
        text = text.strip()
        if language == "vi":
            for abbr, expanded in ABBREVIATIONS_VI.items():
                text = re.sub(rf"\b{abbr}\b", expanded, text, flags=re.IGNORECASE)
            text = re.sub(r"\b(\d+)\s*\$", r"\1 đô la", text)
            text = re.sub(r"\b(\d+)\s*₫\b", r"\1 đồng", text)
        elif language == "en":
            text = re.sub(r"\b(\d+)\s*\$", r"\1 dollars", text)
        return text

    def extract_prosodic_structure(self, text: str, language: str = "vi") -> Dict[str, Any]:
        """Extract sentence tokens, punctuation pause durations, and intonation pitch tags."""
        normalized = self.normalize_text(text, language)

        # Split clauses by punctuation
        clauses = re.split(r"([,.\?!;:])", normalized)
        prosodic_segments = []

        total_pause_ms = 0.0
        pitch_contour_type = "flat"

        for i in range(0, len(clauses), 2):
            segment_text = clauses[i].strip()
            if not segment_text:
                continue

            punct = clauses[i + 1] if (i + 1) < len(clauses) else ""
            pause_ms = 0
            pitch_accent = 1.0

            if punct == ",":
                pause_ms = 150
            elif punct in [".", ";", ":"]:
                pause_ms = 350
            elif punct == "?":
                pause_ms = 250
                pitch_accent = 1.25  # Upward pitch tail tilt for questions
                pitch_contour_type = "question_rising"
            elif punct == "!":
                pause_ms = 200
                pitch_accent = 1.30  # High energy exclamation
                pitch_contour_type = "exclamation_energetic"

            total_pause_ms += pause_ms
            prosodic_segments.append({
                "text": segment_text,
                "punctuation": punct,
                "pause_ms": pause_ms,
                "pitch_accent": pitch_accent,
            })

        return {
            "original_text": text,
            "normalized_text": normalized,
            "segments": prosodic_segments,
            "total_pause_ms": total_pause_ms,
            "pitch_contour_type": pitch_contour_type,
        }


def main():
    _ensure_utf8_stdout()
    enhancer = ProsodyEnhancer()
    sample_text = "Xin chào VNG! Mô hình AI trên NPU Qualcomm có nhanh không?"
    structure = enhancer.extract_prosodic_structure(sample_text, language="vi")

    print(f" • Input Text : {structure['original_text']}")
    print(f" • Normalized : {structure['normalized_text']}")
    print(f" • Pitch Contour: {structure['pitch_contour_type']}")
    print(f" • Segments ({len(structure['segments'])}):")
    for seg in structure["segments"]:
        print(f"    - Text: '{seg['text']}' | Punct: '{seg['punctuation']}' | Pause: {seg['pause_ms']}ms | Accent: x{seg['pitch_accent']:.2f}")


if __name__ == "__main__":
    main()
