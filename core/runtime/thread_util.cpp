#include "core/runtime/thread_util.hpp"

#include <string>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ee {

unsigned hardware_threads() noexcept {
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : n;
}

bool set_current_thread_name(std::string_view name) {
#if defined(_WIN32)
    const std::wstring wide(name.begin(), name.end());
    return SUCCEEDED(SetThreadDescription(GetCurrentThread(), wide.c_str()));
#elif defined(__APPLE__)
    const std::string n(name.substr(0, 63));
    return pthread_setname_np(n.c_str()) == 0;
#else
    const std::string n(name.substr(0, 15));  // Linux limit: 16 bytes including the terminator
    return pthread_setname_np(pthread_self(), n.c_str()) == 0;
#endif
}

bool pin_current_thread(std::span<const int> cores) {
    const unsigned n = hardware_threads();
#if defined(_WIN32)
    DWORD_PTR mask = 0;
    for (int c : cores) {
        if (c >= 0 && static_cast<unsigned>(c) < n && c < 64) mask |= DWORD_PTR{1} << c;
    }
    return mask != 0 && SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#elif defined(__APPLE__)
    (void)cores;
    (void)n;
    return false;  // macOS offers affinity hints only, not pinning
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    bool any = false;
    for (int c : cores) {
        if (c < 0 || static_cast<unsigned>(c) >= n || c >= CPU_SETSIZE) continue;
        CPU_SET(static_cast<unsigned>(c), &set);
        any = true;
    }
    return any && pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}

bool set_current_thread_priority(ThreadPriority priority) {
#if defined(_WIN32)
    int p = THREAD_PRIORITY_NORMAL;
    switch (priority) {
    case ThreadPriority::Low: p = THREAD_PRIORITY_BELOW_NORMAL; break;
    case ThreadPriority::Normal: p = THREAD_PRIORITY_NORMAL; break;
    case ThreadPriority::High: p = THREAD_PRIORITY_HIGHEST; break;
    case ThreadPriority::Realtime: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }
    return SetThreadPriority(GetCurrentThread(), p) != 0;
#elif defined(__APPLE__)
    (void)priority;
    return false;
#else
    if (priority == ThreadPriority::Realtime) {
        sched_param param{};
        param.sched_priority = 80;
        if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) == 0) return true;
        // Without CAP_SYS_NICE fall back to the strongest nice level we are allowed.
    }
    int nice_value = 0;
    switch (priority) {
    case ThreadPriority::Low: nice_value = 10; break;
    case ThreadPriority::Normal: return true;
    case ThreadPriority::High: nice_value = -5; break;
    case ThreadPriority::Realtime: nice_value = -10; break;
    }
    const auto tid = static_cast<id_t>(syscall(SYS_gettid));
    return setpriority(PRIO_PROCESS, tid, nice_value) == 0;
#endif
}

}  // namespace ee
