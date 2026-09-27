#include "core/translate/glossary.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>

#include "core/runtime/params.hpp"

namespace ee {

namespace {

bool is_word_char(unsigned char c) { return std::isalnum(c) != 0 || c == '_' || c >= 0x80; }

bool iequal_at(std::string_view text, std::size_t pos, std::string_view term) {
    if (pos + term.size() > text.size()) return false;
    for (std::size_t i = 0; i < term.size(); ++i) {
        const auto a = static_cast<unsigned char>(text[pos + i]);
        const auto b = static_cast<unsigned char>(term[i]);
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
}

}  // namespace

void Glossary::add(std::string source, std::string target) {
    if (source.empty()) return;
    terms_.push_back({std::move(source), std::move(target)});
    std::stable_sort(terms_.begin(), terms_.end(),
                     [](const Term& a, const Term& b) { return a.source.size() > b.source.size(); });
}

Glossary Glossary::load(const std::filesystem::path& path) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        throw ConfigError("cannot read glossary '" + path.string() + "': " + e.what());
    }
    Glossary g;
    if (!root.IsMap()) throw ConfigError("glossary must be a mapping of term: translation");
    for (const auto& kv : root) {
        g.add(kv.first.as<std::string>(), kv.second.IsNull() ? std::string() : kv.second.as<std::string>());
    }
    return g;
}

Glossary::Protected Glossary::protect(std::string_view text) const {
    Protected p;
    for (std::size_t i = 0; i < text.size();) {
        const bool at_boundary = i == 0 || !is_word_char(static_cast<unsigned char>(text[i - 1]));
        const Term* hit = nullptr;
        if (at_boundary) {
            for (const Term& t : terms_) {
                const std::size_t end = i + t.source.size();
                if (iequal_at(text, i, t.source) &&
                    (end == text.size() || !is_word_char(static_cast<unsigned char>(text[end])))) {
                    hit = &t;
                    break;
                }
            }
        }
        if (hit == nullptr) {
            p.text += text[i];
            ++i;
            continue;
        }
        p.text += "__T" + std::to_string(p.replacements.size()) + "__";
        p.replacements.push_back(hit->target.empty() ? std::string(text.substr(i, hit->source.size())) : hit->target);
        i += hit->source.size();
    }
    return p;
}

std::string Glossary::restore(std::string_view translated, const Protected& p, std::uint32_t* missing) const {
    std::string out(translated);
    std::uint32_t lost = 0;
    for (std::size_t i = p.replacements.size(); i-- > 0;) {
        const std::string placeholder = "__T" + std::to_string(i) + "__";
        const std::size_t pos = out.find(placeholder);
        if (pos == std::string::npos) {
            ++lost;
            continue;
        }
        out.replace(pos, placeholder.size(), p.replacements[i]);
    }
    if (missing != nullptr) *missing = lost;
    return out;
}

}  // namespace ee
