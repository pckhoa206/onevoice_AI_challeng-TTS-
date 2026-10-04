#include "supertonic_tokenizer_kernel.h"
#include "indexer_table.h"
#include <cstring>
#include <algorithm>
#include <cctype>

namespace supertonic {

TokenizerKernel::TokenizerKernel() {
    // Zero dynamic initialization required; G_INDEXER_TABLE is statically allocated in .rodata
}

std::vector<uint32_t> TokenizerKernel::utf8_to_codepoints(const char* str) {
    std::vector<uint32_t> codepoints;
    if (!str) return codepoints;

    size_t len = std::strlen(str);
    size_t i = 0;
    while (i < len) {
        uint8_t c = static_cast<uint8_t>(str[i]);
        uint32_t cp = 0;

        if (c < 0x80) {
            cp = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 < len) {
                cp = ((c & 0x1F) << 6) | (static_cast<uint8_t>(str[i + 1]) & 0x3F);
                i += 2;
            } else {
                break;
            }
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 < len) {
                cp = ((c & 0x0F) << 12) |
                     ((static_cast<uint8_t>(str[i + 1]) & 0x3F) << 6) |
                     (static_cast<uint8_t>(str[i + 2]) & 0x3F);
                i += 3;
            } else {
                break;
            }
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 < len) {
                cp = ((c & 0x07) << 18) |
                     ((static_cast<uint8_t>(str[i + 1]) & 0x3F) << 12) |
                     ((static_cast<uint8_t>(str[i + 2]) & 0x3F) << 6) |
                     (static_cast<uint8_t>(str[i + 3]) & 0x3F);
                i += 4;
            } else {
                break;
            }
        } else {
            // Invalid byte, skip
            i += 1;
            continue;
        }

        codepoints.push_back(cp);
    }
    return codepoints;
}

void TokenizerKernel::decompose_hangul(uint32_t cp, std::vector<uint32_t>& out_cps) {
    if (cp >= 0xAC00 && cp <= 0xD7A3) {
        uint32_t s_idx = cp - 0xAC00;
        uint32_t l_idx = s_idx / 588;
        uint32_t v_idx = (s_idx % 588) / 28;
        uint32_t t_idx = s_idx % 28;

        out_cps.push_back(0x1100 + l_idx);
        out_cps.push_back(0x1161 + v_idx);
        if (t_idx > 0) {
            out_cps.push_back(0x11A7 + t_idx);
        }
    } else {
        out_cps.push_back(cp);
    }
}

bool TokenizerKernel::is_ending_punct(uint32_t cp) {
    // [.!?;:',")]}…。」』】〉》›»]
    static const uint32_t PUNCTS[] = {
        '.', '!', '?', ';', ':', '\'', '"', ')', ']', '}',
        0x2026, 0x3002, 0x300D, 0x300F, 0x3011, 0x3009, 0x300B, 0x203A, 0x00BB
    };
    for (uint32_t p : PUNCTS) {
        if (cp == p) return true;
    }
    return false;
}

std::vector<uint32_t> TokenizerKernel::preprocess_codepoints(const std::vector<uint32_t>& raw_cps, const std::string& lang) {
    std::vector<uint32_t> step1;
    step1.reserve(raw_cps.size() * 2);

    // 1. Hangul decomposition & Symbol normalization & Special char removal
    for (size_t i = 0; i < raw_cps.size(); ++i) {
        uint32_t cp = raw_cps[i];

        // Remove special decorative symbols: ♥ (0x2665), ☆ (0x2606), ♡ (0x2661), © (0x00A9), \ (92)
        if (cp == 0x2665 || cp == 0x2606 || cp == 0x2661 || cp == 0x00A9 || cp == '\\') {
            continue;
        }

        // Expand @ to " at "
        if (cp == '@') {
            step1.push_back(' ');
            step1.push_back('a');
            step1.push_back('t');
            step1.push_back(' ');
            continue;
        }

        // Normalize symbols
        switch (cp) {
            case 0x2013: // –
            case 0x2011: // ‑
            case 0x2014: // —
                cp = '-';
                break;
            case 0x00AF: // ¯
            case '_':
            case '[':
            case ']':
            case '|':
            case '/':
            case '#':
            case 0x2192: // →
            case 0x2190: // ←
                cp = ' ';
                break;
            case 0x201C: // “
            case 0x201D: // ”
                cp = '"';
                break;
            case 0x2018: // ‘
            case 0x2019: // ’
            case 0x00B4: // ´
            case '`':
                cp = '\'';
                break;
            default:
                break;
        }

        // Hangul syllable decomposition for Korean
        if (cp >= 0xAC00 && cp <= 0xD7A3) {
            decompose_hangul(cp, step1);
        } else {
            step1.push_back(cp);
        }
    }

    // 2. Fix punctuation spacing: remove space before [, . ! ? ; : ']
    std::vector<uint32_t> step2;
    step2.reserve(step1.size());
    for (size_t i = 0; i < step1.size(); ++i) {
        if (step1[i] == ' ' && (i + 1 < step1.size())) {
            uint32_t next = step1[i + 1];
            if (next == ',' || next == '.' || next == '!' || next == '?' ||
                next == ';' || next == ':' || next == '\'') {
                // skip space
                continue;
            }
        }
        step2.push_back(step1[i]);
    }

    // 3. Remove duplicate consecutive quotes
    std::vector<uint32_t> step3;
    step3.reserve(step2.size());
    for (size_t i = 0; i < step2.size(); ++i) {
        uint32_t cp = step2[i];
        if ((cp == '"' || cp == '\'' || cp == '`') && !step3.empty() && step3.back() == cp) {
            // Skip duplicate quote
            continue;
        }
        step3.push_back(cp);
    }

    // 4. Clean whitespace: collapse multiple spaces and trim
    std::vector<uint32_t> step4;
    step4.reserve(step3.size());
    bool in_space = false;
    for (uint32_t cp : step3) {
        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') {
            if (!in_space && !step4.empty()) {
                step4.push_back(' ');
                in_space = true;
            }
        } else {
            step4.push_back(cp);
            in_space = false;
        }
    }
    // Trim trailing space
    while (!step4.empty() && step4.back() == ' ') {
        step4.pop_back();
    }

    // 5. Add period if needed
    if (!step4.empty() && !is_ending_punct(step4.back())) {
        step4.push_back('.');
    }

    // 6. Wrap with language tokens
    std::vector<uint32_t> final_cps;
    final_cps.reserve(step4.size() + 10);

    // Opening tag
    final_cps.push_back('<');
    for (char c : lang) {
        final_cps.push_back(static_cast<uint32_t>(c));
    }
    final_cps.push_back('>');

    // Content
    for (uint32_t cp : step4) {
        final_cps.push_back(cp);
    }

    // Closing tag
    final_cps.push_back('<');
    final_cps.push_back('/');
    for (char c : lang) {
        final_cps.push_back(static_cast<uint32_t>(c));
    }
    final_cps.push_back('>');

    return final_cps;
}

int TokenizerKernel::tokenize(const char* utf8_text, const char* lang, int64_t* out_ids, float* out_mask, int max_len) {
    if (!utf8_text || !out_ids || !out_mask || max_len <= 0) {
        return 0;
    }

    std::string lang_str = (lang && std::strlen(lang) > 0) ? lang : "en";

    // 1. Decode UTF-8 string to Unicode Code Points
    std::vector<uint32_t> raw_cps = utf8_to_codepoints(utf8_text);

    // 2. Preprocess & add language tags
    std::vector<uint32_t> processed_cps = preprocess_codepoints(raw_cps, lang_str);

    // 3. Map Code Points to Token IDs via G_INDEXER_TABLE (O(1) in Hexagon TCM/L1)
    std::vector<int64_t> token_ids;
    token_ids.reserve(processed_cps.size());

    for (uint32_t cp : processed_cps) {
        if (cp < SUPERTONIC_INDEXER_SIZE) {
            int16_t tok_id = G_INDEXER_TABLE[cp];
            if (tok_id != -1) {
                token_ids.push_back(static_cast<int64_t>(tok_id));
            }
        }
    }

    int valid_len = static_cast<int>(token_ids.size());
    int copy_len = std::min(valid_len, max_len);

    // 4. Fill output buffers with static shape & zero padding
    for (int i = 0; i < copy_len; ++i) {
        out_ids[i] = token_ids[i];
        out_mask[i] = 1.0f;
    }
    for (int i = copy_len; i < max_len; ++i) {
        out_ids[i] = 0;
        out_mask[i] = 0.0f;
    }

    return valid_len;
}

} // namespace supertonic
