#pragma once

#include <string_view>

#include "core/runtime/config.hpp"

namespace ee {

/// Best-effort helpers for the scheduler threads. Each returns false when the platform or
/// the process's privileges do not allow the request; callers carry on regardless.
bool set_current_thread_name(std::string_view name);
bool pin_current_thread(int core);
bool set_current_thread_priority(ThreadPriority priority);
[[nodiscard]] unsigned hardware_threads() noexcept;

}  // namespace ee
