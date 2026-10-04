#ifndef SUPERTONIC_TOKENIZER_KERNEL_H
#define SUPERTONIC_TOKENIZER_KERNEL_H

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

namespace supertonic {

/**
 * High-performance C++ Tokenizer Core for Qualcomm Hexagon HTP / DSP.
 * Supports English ('en') and Korean ('ko') with direct Hangul Jamo bit-shift decomposition.
 */
class TokenizerKernel {
public:
    TokenizerKernel();
    ~TokenizerKernel() = default;

    /**
     * Tokenizes UTF-8 input text for target language.
     * @param utf8_text  Input UTF-8 C-string.
     * @param lang       Target language code ("en", "ko").
     * @param out_ids    Output buffer of size max_len for token IDs (int64).
     * @param out_mask   Output buffer of size max_len for attention mask (float32).
     * @param max_len    Target sequence length (default 64 for Qualcomm NPU).
     * @return Number of valid tokens before padding.
     */
    int tokenize(const char* utf8_text, const char* lang, int64_t* out_ids, float* out_mask, int max_len = 64);

private:
    std::vector<uint32_t> utf8_to_codepoints(const char* utf8_str);
    std::vector<uint32_t> preprocess_codepoints(const std::vector<uint32_t>& raw_cps, const std::string& lang);
    void decompose_hangul(uint32_t cp, std::vector<uint32_t>& out_cps);
    bool is_ending_punct(uint32_t cp);
};

} // namespace supertonic

#endif // SUPERTONIC_TOKENIZER_KERNEL_H
