// Models loaded once per process and shared by the sessions that ask for the same one.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/runtime/shared_cache.hpp"

namespace ee {
namespace {

struct Model {
    explicit Model(std::string n) : name(std::move(n)) {}
    std::string name;
};

TEST(SharedCache, LoadsEachModelOnceWhileItIsHeld) {
    SharedCache<Model> cache;
    int loads = 0;
    const auto load = [&](const std::string& name) {
        return [&loads, name] {
            ++loads;
            return std::make_shared<Model>(name);
        };
    };
    auto a = cache.get("whisper|gpu0", load("whisper"));
    auto b = cache.get("whisper|gpu0", load("whisper"));
    EXPECT_EQ(a, b);  // the second session gets the first one's model
    EXPECT_EQ(loads, 1);
    auto c = cache.get("whisper|cpu", load("whisper"));  // another device: another model
    EXPECT_NE(a, c);
    EXPECT_EQ(loads, 2);

    // The last holder gone, the model is freed; the next session loads it again.
    const std::weak_ptr<Model> watch = a;
    a.reset();
    b.reset();
    EXPECT_TRUE(watch.expired());
    auto d = cache.get("whisper|gpu0", load("whisper"));
    EXPECT_EQ(loads, 3);
    EXPECT_EQ(d->name, "whisper");
}

TEST(SharedCache, AFailedLoadLeavesNothingBehind) {
    SharedCache<Model> cache;
    EXPECT_THROW((void)cache.get("nllb", []() -> std::shared_ptr<Model> { throw std::runtime_error("missing file"); }),
                 std::runtime_error);
    auto m = cache.get("nllb", [] { return std::make_shared<Model>("nllb"); });
    EXPECT_EQ(m->name, "nllb");
}

TEST(SharedCache, SessionsBuiltAtOnceShareOneLoad) {
    SharedCache<Model> cache;
    std::atomic<int> loads{0};
    std::vector<std::shared_ptr<Model>> got(8);
    std::vector<std::thread> sessions;
    for (std::size_t i = 0; i < got.size(); ++i) {
        sessions.emplace_back([&, i] {
            got[i] = cache.get("nllb", [&] {
                ++loads;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));  // a slow load
                return std::make_shared<Model>("nllb");
            });
        });
    }
    for (std::thread& t : sessions) t.join();
    EXPECT_EQ(loads.load(), 1);
    for (const auto& m : got) EXPECT_EQ(m, got[0]);
}

}  // namespace
}  // namespace ee
