#pragma once

#include <chrono>
#include <cstdint>

namespace ee {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

[[nodiscard]] inline std::int64_t micros_between(TimePoint from, TimePoint to) {
    return std::chrono::duration_cast<std::chrono::microseconds>(to - from).count();
}

[[nodiscard]] inline double seconds_between(TimePoint from, TimePoint to) {
    return std::chrono::duration<double>(to - from).count();
}

[[nodiscard]] inline TimePoint add_seconds(TimePoint t, double seconds) {
    return t + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

}  // namespace ee
