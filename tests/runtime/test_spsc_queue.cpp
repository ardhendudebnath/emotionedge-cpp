#include <gtest/gtest.h>

#include <cstdint>
#include <thread>

#include "core/runtime/frame.hpp"
#include "core/runtime/spsc_queue.hpp"

namespace ee {
namespace {

TEST(SpscQueue, RoundsCapacityUpToPowerOfTwo) {
    EXPECT_EQ(SpscQueue<int>(5).capacity(), 8u);
    EXPECT_EQ(SpscQueue<int>(8).capacity(), 8u);
    EXPECT_EQ(SpscQueue<int>(1).capacity(), 2u);
}

TEST(SpscQueue, DeliversInOrderAndRejectsWhenFull) {
    SpscQueue<int> q(4);
    for (int i = 0; i < 4; ++i) EXPECT_TRUE(q.try_push(i));
    EXPECT_FALSE(q.try_push(99));
    EXPECT_EQ(q.size(), 4u);

    int v = -1;
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(q.try_pop(v));
        EXPECT_EQ(v, i);
    }
    EXPECT_FALSE(q.try_pop(v));
    EXPECT_TRUE(q.empty());
}

TEST(SpscQueue, PeekLooksBehindTheFront) {
    SpscQueue<int> q(4);
    ASSERT_TRUE(q.try_push(1));
    ASSERT_TRUE(q.try_push(2));
    ASSERT_NE(q.peek(0), nullptr);
    ASSERT_NE(q.peek(1), nullptr);
    EXPECT_EQ(*q.peek(0), 1);
    EXPECT_EQ(*q.peek(1), 2);
    EXPECT_EQ(q.peek(2), nullptr);
}

TEST(SpscQueue, WrapsAroundManyTimes) {
    SpscQueue<int> q(4);
    int v = 0;
    for (int i = 0; i < 1000; ++i) {
        ASSERT_TRUE(q.try_push(i));
        ASSERT_TRUE(q.try_pop(v));
        ASSERT_EQ(v, i);
    }
}

TEST(SpscQueue, AcquirePublishFillsSlotsInPlace) {
    SpscQueue<Frame> q(2);
    Frame* slot = q.acquire();
    ASSERT_NE(slot, nullptr);
    slot->reset(FrameKind::Audio);
    slot->audio.assign(320, 0.25f);
    q.publish();

    Frame* front = q.front();
    ASSERT_NE(front, nullptr);
    EXPECT_EQ(front->kind, FrameKind::Audio);
    EXPECT_EQ(front->audio.size(), 320u);
    q.pop();
    EXPECT_EQ(q.front(), nullptr);
}

TEST(SpscQueue, ReusedSlotsKeepTheirBuffers) {
    // Copying a same-size frame into a slot that already held one must not reallocate:
    // this is what keeps steady-state audio traffic allocation-free.
    SpscQueue<Frame> q(2);
    q.for_each_slot([](Frame& f) { f.audio.reserve(480); });

    Frame src;
    src.reset(FrameKind::Audio);
    src.audio.assign(480, 0.5f);

    std::vector<const float*> first_pass;
    for (int i = 0; i < 2; ++i) {
        Frame* slot = q.acquire();
        ASSERT_NE(slot, nullptr);
        *slot = src;
        first_pass.push_back(slot->audio.data());
        q.publish();
    }
    for (int i = 0; i < 2; ++i) q.pop();
    for (int i = 0; i < 2; ++i) {
        Frame* slot = q.acquire();
        ASSERT_NE(slot, nullptr);
        *slot = src;
        EXPECT_EQ(slot->audio.data(), first_pass[static_cast<std::size_t>(i)]);
        q.publish();
    }
}

TEST(SpscQueue, ConcurrentProducerAndConsumerPreserveOrder) {
    SpscQueue<std::uint64_t> q(256);
    constexpr std::uint64_t kCount = 500'000;
    std::thread producer([&] {
        for (std::uint64_t i = 0; i < kCount;) {
            if (q.try_push(i)) {
                ++i;
            } else {
                std::this_thread::yield();
            }
        }
    });
    std::uint64_t expected = 0;
    std::uint64_t v = 0;
    bool in_order = true;
    while (expected < kCount) {
        if (q.try_pop(v)) {
            in_order = in_order && (v == expected);
            ++expected;
        } else {
            std::this_thread::yield();
        }
    }
    producer.join();
    EXPECT_TRUE(in_order);
    EXPECT_TRUE(q.empty());
}

}  // namespace
}  // namespace ee
