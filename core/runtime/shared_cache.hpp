#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace ee {

/// Models loaded once per process and shared by every session that asks for the same one:
/// the streaming server's sessions, or the emotion and consistency stages of one session.
/// An entry lives while any caller holds it, so the last session to end frees its models.
///
/// `load` runs under the cache's lock: sessions built at the same time load a model once, and
/// the others wait for it instead of loading their own copy.
template <typename T>
class SharedCache {
public:
    template <typename Load>
    [[nodiscard]] std::shared_ptr<T> get(const std::string& key, Load&& load) {
        const std::lock_guard<std::mutex> lock(mu_);
        if (std::shared_ptr<T> existing = entries_[key].lock()) return existing;
        std::shared_ptr<T> loaded = load();
        entries_[key] = loaded;
        return loaded;
    }

private:
    std::mutex mu_;
    std::map<std::string, std::weak_ptr<T>> entries_;
};

}  // namespace ee
