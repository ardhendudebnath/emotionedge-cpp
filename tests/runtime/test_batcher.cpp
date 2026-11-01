// Requests from several sessions run in batches on one executor (the shared translator).
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "core/runtime/batcher.hpp"

namespace ee {
namespace {

using IntBatcher = Batcher<int, int>;

TEST(Batcher, AnIdleExecutorRunsARequestAtOnce) {
    std::vector<std::size_t> sizes;
    IntBatcher b(
        [&](std::vector<int>& batch) {
            sizes.push_back(batch.size());
            std::vector<int> out;
            for (int x : batch) out.push_back(2 * x);
            return out;
        },
        8);
    EXPECT_EQ(b.submit(21), 42);
    EXPECT_EQ(b.submit(5), 10);
    EXPECT_EQ(sizes, (std::vector<std::size_t>{1, 1}));  // nothing waited to be batched
    EXPECT_EQ(b.batches(), 2u);
}

TEST(Batcher, RequestsThatArriveDuringABatchGoTogether) {
    std::mutex mu;
    std::condition_variable cv;
    bool release = false;
    std::vector<std::vector<int>> seen;
    IntBatcher b(
        [&](std::vector<int>& batch) {
            std::unique_lock<std::mutex> lock(mu);
            seen.push_back(batch);
            if (seen.size() == 1) cv.wait(lock, [&] { return release; });  // the first batch is slow
            std::vector<int> out;
            for (int x : batch) out.push_back(x + 100);
            return out;
        },
        8);
    std::vector<int> results(4, 0);
    std::vector<std::thread> sessions;
    EXPECT_FALSE(b.busy());
    sessions.emplace_back([&] { results[0] = b.submit(0); });
    while (b.batches() == 0) std::this_thread::yield();  // the first request is running
    EXPECT_TRUE(b.busy());  // a new request would wait (callers can skip optional work)
    for (int i = 1; i < 4; ++i) sessions.emplace_back([&, i] { results[i] = b.submit(i); });
    while (b.waiting() < 3) std::this_thread::yield();  // the others queue behind it
    {
        const std::lock_guard<std::mutex> lock(mu);
        release = true;
    }
    cv.notify_all();
    for (std::thread& t : sessions) t.join();
    while (b.busy()) std::this_thread::yield();  // idle once the last batch has delivered

    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0].size(), 1u);
    EXPECT_EQ(seen[1].size(), 3u);  // one batch for the three that waited
    EXPECT_EQ(results, (std::vector<int>{100, 101, 102, 103}));  // each gets its own result
    EXPECT_EQ(b.requests(), 4u);
}

TEST(Batcher, KeepsGroupsApartAndCapsTheBatch) {
    std::mutex mu;
    std::condition_variable cv;
    bool release = false;
    std::vector<std::vector<int>> seen;
    IntBatcher b(
        [&](std::vector<int>& batch) {
            std::unique_lock<std::mutex> lock(mu);
            seen.push_back(batch);
            if (seen.size() == 1) cv.wait(lock, [&] { return release; });
            return std::vector<int>(batch.size(), 0);
        },
        2, [](const int& x) { return x % 2; });  // odd and even requests never share a batch
    std::vector<std::thread> sessions;
    sessions.emplace_back([&] { (void)b.submit(0); });
    while (b.batches() == 0) std::this_thread::yield();
    std::size_t queued = 0;
    for (int x : {2, 4, 6, 1}) {
        sessions.emplace_back([&, x] { (void)b.submit(x); });
        ++queued;  // nothing leaves the queue while the first batch runs
        while (b.waiting() < queued) std::this_thread::yield();  // queued in this order
    }
    {
        const std::lock_guard<std::mutex> lock(mu);
        release = true;
    }
    cv.notify_all();
    for (std::thread& t : sessions) t.join();

    ASSERT_EQ(seen.size(), 4u);
    EXPECT_EQ(seen[1], (std::vector<int>{2, 4}));  // the cap
    for (const auto& batch : seen) {
        for (int x : batch) EXPECT_EQ(x % 2, batch.front() % 2);
    }
}

TEST(Batcher, EveryRequestOfAFailedBatchGetsTheError) {
    IntBatcher b([](std::vector<int>&) -> std::vector<int> { throw std::runtime_error("CUDA out of memory"); }, 4);
    EXPECT_THROW((void)b.submit(1), std::runtime_error);
    EXPECT_THROW((void)b.submit(2), std::runtime_error);  // and the batcher keeps working
}

}  // namespace
}  // namespace ee
