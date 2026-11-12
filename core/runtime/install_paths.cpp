#include "core/runtime/install_paths.hpp"

#include <string>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>

#include <cstdint>
#endif

namespace fs = std::filesystem;

namespace ee {

fs::path executable_path() {
#if defined(_WIN32)
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) return {};
        if (n < buf.size()) {
            buf.resize(n);
            return fs::path(buf);
        }
        buf.resize(buf.size() * 2);  // truncated: try a bigger buffer
    }
#elif defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    if (const auto end = buf.find('\0'); end != std::string::npos) buf.resize(end);
    std::error_code ec;
    const fs::path canonical = fs::weakly_canonical(buf, ec);
    return ec ? fs::path(buf) : canonical;
#else
    std::error_code ec;
    const fs::path self = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::path() : self;
#endif
}

fs::path installed_data_dir(const fs::path& executable, const fs::path& bin_dir, const fs::path& data_dir) {
    if (data_dir.is_absolute()) return data_dir;
    if (executable.empty() || bin_dir.is_absolute()) return {};
    fs::path prefix = executable.parent_path();
    for (const fs::path& part : bin_dir) {
        if (part.empty() || part == ".") continue;
        prefix = prefix.parent_path();
    }
    return (prefix / data_dir).lexically_normal();
}

fs::path first_existing(std::initializer_list<fs::path> candidates) {
    for (const fs::path& candidate : candidates) {
        std::error_code ec;
        if (!candidate.empty() && fs::exists(candidate, ec)) return candidate;
    }
    return {};
}

}  // namespace ee
