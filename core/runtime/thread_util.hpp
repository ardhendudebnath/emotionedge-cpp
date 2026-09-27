#pragma once

#include <span>
#include <string_view>

#include "core/runtime/config.hpp"

namespace ee {

/// Best-effort helpers for the scheduler threads. Each returns false when the platform or
/// the process's privileges do not allow the request; callers carry on regardless.
bool set_current_thread_name(std::string_view name);
/// Pins to the listed CPUs that exist on this machine; false if none do or pinning fails.
bool pin_current_thread(std::span<const int> cores);
bool set_current_thread_priority(ThreadPriority priority);
[[nodiscard]] unsigned hardware_threads() noexcept;

}  // namespace ee
