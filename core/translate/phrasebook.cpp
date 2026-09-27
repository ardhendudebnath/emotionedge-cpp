#include "core/translate/phrasebook.hpp"

#include <yaml-cpp/yaml.h>

#include <cctype>

#include "core/asr/local_agreement.hpp"
#include "core/translate/control_tokens.hpp"
#include "core/translate/languages.hpp"

namespace ee {

namespace {
std::string key(std::string_view src, std::string_view tgt, std::string_view normalized) {
    return std::string(src) + "-" + std::string(tgt) + "|" + std::string(normalized);
}
}  // namespace

std::string PhrasebookTranslator::normalize(std::string_view text) {
    std::string out;
    bool space = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c == 0xE2 && i + 2 < text.size() && static_cast<unsigned char>(text[i + 1]) == 0x80 &&
            static_cast<unsigned char>(text[i + 2]) == 0x99) {  // ’ -> '
            out += '\'';
            i += 2;
            space = false;
            continue;
        }
        if (std::isspace(c) != 0) {
            space = !out.empty();
            continue;
        }
        if (space) out += ' ';
        space = false;
        out += static_cast<char>(c < 0x80 ? std::tolower(c) : c);
    }
    return out;
}

void PhrasebookTranslator::add(std::string_view source_language, std::string_view target_language,
                               std::string_view source, Entry entry) {
    std::map<std::string, std::string> links;
    for (const auto& [src_word, tgt_word] : entry.emphasis) links[LocalAgreement::normalize(src_word)] = tgt_word;
    entry.emphasis = std::move(links);
    entries_[key(source_language, target_language, normalize(source))] = std::move(entry);
}

PhrasebookTranslator PhrasebookTranslator::load(const std::filesystem::path& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        throw ConfigError("cannot read phrasebook '" + path.string() + "': " + e.what());
    }
    PhrasebookTranslator book;
    const auto pairs = root["pairs"];
    if (!pairs || !pairs.IsMap()) throw ConfigError("phrasebook needs a 'pairs' mapping");
    for (const auto& pair : pairs) {
        const std::string name = pair.first.as<std::string>();
        const auto dash = name.find('-');
        if (dash == std::string::npos) throw ConfigError("phrasebook pair '" + name + "' must be <src>-<tgt>");
        for (const auto& e : pair.second) {
            Entry entry;
            entry.target = e["tgt"].as<std::string>();
            if (const auto links = e["emphasis"]; links && links.IsMap()) {
                for (const auto& kv : links) entry.emphasis[kv.first.as<std::string>()] = kv.second.as<std::string>();
            }
            book.add(name.substr(0, dash), name.substr(dash + 1), e["src"].as<std::string>(), std::move(entry));
        }
    }
    return book;
}

TranslationResult PhrasebookTranslator::translate(const TranslationRequest& request) {
    const Markup source = parse_markup(strip_control_prefix(request.source), request.source_language);
    TranslationResult out;
    const auto it = entries_.find(key(request.source_language, request.target_language, normalize(source.plain)));
    if (it == entries_.end()) {
        // Visible pseudo-translation: same words, markup kept, so emphasis maps one-to-one.
        const std::vector<std::string> words = split_words(source.plain, request.source_language);
        out.text = "[" + std::string(request.target_language) + "] " +
                   apply_emphasis_markup(words, source.emphasis, request.source_language);
        return out;
    }

    const Entry& entry = it->second;
    std::vector<std::string> target_words = split_words(entry.target, request.target_language);
    const std::vector<std::string> source_words = split_words(source.plain, request.source_language);
    std::vector<std::uint16_t> target_emphasis;
    for (std::uint16_t s : source.emphasis) {
        if (s >= source_words.size()) continue;
        const auto link = entry.emphasis.find(LocalAgreement::normalize(source_words[s]));
        if (link == entry.emphasis.end()) continue;
        for (std::size_t t = 0; t < target_words.size(); ++t) {
            if (LocalAgreement::normalize(target_words[t]) == LocalAgreement::normalize(link->second)) {
                target_emphasis.push_back(static_cast<std::uint16_t>(t));
                break;
            }
        }
    }
    out.text = apply_emphasis_markup(target_words, target_emphasis, request.target_language);
    return out;
}

}  // namespace ee
