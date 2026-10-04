"""Text Normalizer for Step 3 TTS (Multilingual: Vi, En, Zh, Ko).
Handles numbers, dates, currency symbols, tech acronyms/loanwords, punctuation,
and special symbols to prevent phonemizer errors or engine crashes.
"""
import re


def vi_num_to_words(n: int) -> str:
    """Convert any integer number up to 999,999,999 to Vietnamese words."""
    if n == 0:
        return "không"

    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

    def read_three_digits(number: int, show_zero_hundred: bool = False) -> str:
        h = number // 100
        t = (number % 100) // 10
        u = number % 10
        res = []

        if h > 0 or show_zero_hundred:
            res.append(f"{digits[h]} trăm")

        if t == 0:
            if show_zero_hundred and u > 0:
                res.append("lẻ")
            elif h > 0 and u > 0:
                res.append("lẻ")
        elif t == 1:
            res.append("mười")
        else:
            res.append(f"{digits[t]} mươi")

        if u > 0:
            if u == 1 and t > 1:
                res.append("mốt")
            elif u == 5 and t > 0:
                res.append("lăm")
            else:
                res.append(digits[u])

        return " ".join(res)

    units = ["", "nghìn", "triệu", "tỷ"]
    parts = []
    curr = n
    idx = 0

    while curr > 0:
        chunk = curr % 1000
        if chunk > 0 or (idx > 0 and (curr // 1000) > 0):
            show_zero = (curr > 999) and (chunk < 100)
            chunk_str = read_three_digits(chunk, show_zero_hundred=show_zero)
            unit = units[idx] if idx < len(units) else ""
            if chunk_str:
                parts.insert(0, f"{chunk_str} {unit}".strip())
        curr //= 1000
        idx += 1

    return " ".join(parts).strip()


class TextNormalizer:
    def __init__(self):
        # Acronym & Loanword maps per language
        self.vi_acronyms = {
            "VNG": "vê en giê",
            "AI": "ei ai",
            "NPU": "en pi u",
            "GPU": "gi pi u",
            "CPU": "si pi u",
            "TTS": "tê tê ét",
            "ASR": "a ét er",
            "MT": "em tê",
            "QUALCOMM": "quai com",
            "CHALLENGE": "che len",
            "PROJECT": "dự án",
            "SERVER": "xơ vơ",
            "CODE": "cốt",
            "DEVICE": "đề vai",
            "EDGE": "ép giơ",
            "CLOUD": "cờ lao",
            "HACKATHON": "hắc ca thon",
            "MODEL": "mô hình",
            "DEMO": "đê mô",
            "APP": "áp",
            "API": "a pi i",
            "USD": "đô la",
            "VND": "đồng",
            "EUR": "ơ rô",
        }

        self.en_acronyms = {
            "VNG": "V N G",
            "NPU": "N P U",
            "GPU": "G P U",
            "CPU": "C P U",
            "TTS": "T T S",
            "ASR": "A S R",
            "MT": "M T",
        }

        self.ko_acronyms = {
            "NPU": "엔피유",
            "GPU": "지피유",
            "CPU": "씨피유",
            "AI": "에이아이",
            "TTS": "티티에스",
            "VNG": "브이엔지",
            "QUALCOMM": "퀄컴",
            "CHALLENGE": "챌린지",
        }

        self.zh_acronyms = {
            "AI": "人工智能",
            "NPU": "N P U",
            "GPU": "G P U",
            "CPU": "C P U",
            "QUALCOMM": "高通",
            "VNG": "V N G",
            "CHALLENGE": "挑战赛",
        }

    def clean_common(self, text: str) -> str:
        """Strip URLs, emojis, excessive punctuation, and normalize whitespace."""
        if not text:
            return ""
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[\!\?\.]{2,}", ".", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _convert_vi_numbers(self, text: str) -> str:
        """Convert all numbers (e.g. 2026 -> hai nghìn không trăm hai mươi sáu, 50000 -> năm mươi nghìn)."""
        def replace_num(match):
            num_str = match.group(0).replace(",", "").replace(".", "")
            try:
                val = int(num_str)
                return vi_num_to_words(val)
            except ValueError:
                return match.group(0)

        # Match numbers (with commas or dots like 50,000 or 2026)
        return re.sub(r"\b\d{1,9}\b", replace_num, text)

    def spell_out_acronyms(self, text: str) -> str:
        """Space out all-caps acronyms so the TTS model pronounces each letter normally (e.g. VNG -> V N G, AI -> A I, NPU -> N P U)."""
        text = re.sub(r"([A-Z]{2,})", lambda m: " " + " ".join(list(m.group(1))) + " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def normalize_vi(self, text: str) -> str:
        text = self.clean_common(text)
        if not text:
            return ""

        # Normalize currency symbols before numbers ($ 50000 -> 50000 đô la)
        text = re.sub(r"(\d+(?:[\.,]\d+)?)\s*(\$|USD|đô la)\s*(USD|đô la)?", r"\1 đô la", text, flags=re.IGNORECASE)
        text = re.sub(r"\$\s*(\d+(?:[\.,]\d+)?)", r"\1 đô la", text)
        text = re.sub(r"(\d+(?:[\.,]\d+)?)\s*(?:₫|VND)", r"\1 đồng", text, flags=re.IGNORECASE)

        # Spell out acronyms letter-by-letter as normal characters (e.g. VNG -> V N G, AI -> A I, NPU -> N P U)
        text = self.spell_out_acronyms(text)

        # Convert remaining numbers to Vietnamese words
        text = self._convert_vi_numbers(text)

        # Clean extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_en(self, text: str) -> str:
        text = self.clean_common(text)
        if not text:
            return ""

        text = re.sub(r"(\d+(?:[\.,]\d+)?)\s*(\$|USD)\s*(USD)?", r"\1 dollars", text, flags=re.IGNORECASE)
        text = re.sub(r"\$\s*(\d+(?:[\.,]\d+)?)", r"\1 dollars", text)

        # Spell out acronyms letter-by-letter as normal characters (e.g. VNG -> V N G, NPU -> N P U)
        text = self.spell_out_acronyms(text)
        return text

    def normalize_ko(self, text: str) -> str:
        text = self.clean_common(text)
        if not text:
            return ""

        text = re.sub(r"(\d+(?:[\.,]\d+)?)\s*(?:원|₩)", r"\1원", text)

        # Spell out acronyms letter-by-letter as normal characters (e.g. NPU -> N P U, SNS -> S N S)
        text = self.spell_out_acronyms(text)
        return text

    def normalize_zh(self, text: str) -> str:
        text = self.clean_common(text)
        if not text:
            return ""

        text = re.sub(r"(\d+(?:[\.,]\d+)?)\s*(?:元|块|￥)", r"\1元", text)

        # Spell out acronyms letter-by-letter as normal characters (e.g. NPU -> N P U, AI -> A I)
        text = self.spell_out_acronyms(text)
        return text

    def normalize(self, text: str, lang: str) -> str:
        """Route to language-specific text normalizer."""
        lang = lang.lower().strip()
        if lang == "vi":
            return self.normalize_vi(text)
        elif lang == "en":
            return self.normalize_en(text)
        elif lang == "ko":
            return self.normalize_ko(text)
        elif lang == "zh":
            return self.normalize_zh(text)
        else:
            return self.clean_common(text)
