#include "core/translate/languages.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <utility>

namespace ee {

namespace {

constexpr std::array<std::pair<std::string_view, std::string_view>, 45> kNllb = {{
    {"en", "eng_Latn"}, {"hi", "hin_Deva"}, {"es", "spa_Latn"}, {"fr", "fra_Latn"}, {"de", "deu_Latn"},
    {"it", "ita_Latn"}, {"pt", "por_Latn"}, {"ru", "rus_Cyrl"}, {"ja", "jpn_Jpan"}, {"zh", "zho_Hans"},
    {"ko", "kor_Hang"}, {"ar", "arb_Arab"}, {"bn", "ben_Beng"}, {"ta", "tam_Taml"}, {"te", "tel_Telu"},
    {"mr", "mar_Deva"}, {"ur", "urd_Arab"}, {"gu", "guj_Gujr"}, {"pa", "pan_Guru"}, {"kn", "kan_Knda"},
    {"ml", "mal_Mlym"}, {"tr", "tur_Latn"}, {"vi", "vie_Latn"}, {"id", "ind_Latn"}, {"nl", "nld_Latn"},
    {"pl", "pol_Latn"}, {"uk", "ukr_Cyrl"}, {"sv", "swe_Latn"}, {"th", "tha_Thai"}, {"he", "heb_Hebr"},
    {"fa", "pes_Arab"}, {"sw", "swh_Latn"}, {"el", "ell_Grek"}, {"cs", "ces_Latn"}, {"ro", "ron_Latn"},
    {"hu", "hun_Latn"}, {"fi", "fin_Latn"}, {"da", "dan_Latn"}, {"no", "nob_Latn"}, {"ms", "zsm_Latn"},
    {"ne", "npi_Deva"}, {"si", "sin_Sinh"}, {"my", "mya_Mymr"}, {"km", "khm_Khmr"}, {"lo", "lao_Laoo"},
}};

constexpr std::array<std::string_view, 7> kNoSpaces = {"zh", "ja", "th", "my", "km", "lo", "bo"};

/// Length of the UTF-8 sequence starting with `lead` (1 for invalid bytes).
std::size_t utf8_length(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;
}

}  // namespace

std::optional<std::string_view> nllb_code(std::string_view iso) noexcept {
    for (const auto& [code, nllb] : kNllb) {
        if (code == iso) return nllb;
    }
    return std::nullopt;
}

bool uses_spaces(std::string_view iso) noexcept {
    for (std::string_view lang : kNoSpaces) {
        if (lang == iso) return false;
    }
    return true;
}

std::vector<std::string> split_words(std::string_view text, std::string_view language) {
    std::vector<std::string> out;
    if (!uses_spaces(language)) {
        for (std::size_t i = 0; i < text.size();) {
            const std::size_t n = std::min(utf8_length(static_cast<unsigned char>(text[i])), text.size() - i);
            if (!(n == 1 && std::isspace(static_cast<unsigned char>(text[i])) != 0)) out.emplace_back(text.substr(i, n));
            i += n;
        }
        return out;
    }
    std::string current;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            if (!current.empty()) out.push_back(std::move(current));
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) out.push_back(std::move(current));
    return out;
}

std::string join_text(const std::vector<std::string>& words, std::string_view language) {
    const std::string_view sep = uses_spaces(language) ? " " : "";
    std::string out;
    for (const std::string& w : words) {
        if (!out.empty()) out += sep;
        out += w;
    }
    return out;
}

}  // namespace ee
