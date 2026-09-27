#include "core/runtime/stage_registry.hpp"

namespace ee {

void StageRegistry::add(std::string type, StageFactory factory) {
    factories_[std::move(type)] = std::move(factory);
}

bool StageRegistry::contains(std::string_view type) const {
    return factories_.find(type) != factories_.end();
}

std::unique_ptr<IStage> StageRegistry::create(std::string_view type) const {
    const auto it = factories_.find(type);
    if (it == factories_.end()) {
        std::string known;
        for (const auto& [name, _] : factories_) known += (known.empty() ? "" : ", ") + name;
        throw ConfigError("unknown stage type '" + std::string(type) + "' (known: " + known + ")");
    }
    return it->second();
}

std::vector<std::string> StageRegistry::types() const {
    std::vector<std::string> out;
    out.reserve(factories_.size());
    for (const auto& [name, _] : factories_) out.push_back(name);
    return out;
}

}  // namespace ee
