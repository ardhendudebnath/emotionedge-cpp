#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/runtime/stage.hpp"

namespace ee {

using StageFactory = std::function<std::unique_ptr<IStage>()>;

/// Maps stage `type` names from the pipeline YAML to factories. New engines plug in by
/// registering an IStage implementation under a new type name.
class StageRegistry {
public:
    /// Registers (or replaces) a factory.
    void add(std::string type, StageFactory factory);
    [[nodiscard]] bool contains(std::string_view type) const;
    /// Throws ConfigError for an unknown type.
    [[nodiscard]] std::unique_ptr<IStage> create(std::string_view type) const;
    [[nodiscard]] std::vector<std::string> types() const;

    template <typename Stage>
    void add(std::string type) {
        add(std::move(type), [] { return std::make_unique<Stage>(); });
    }

private:
    std::map<std::string, StageFactory, std::less<>> factories_;
};

}  // namespace ee
