#pragma once

#include <chrono>
#include <optional>

namespace ee {

/// Whether a session's caption draft runs under multi-user load (translate `busy_drafts`,
/// `draft_refresh_ms`). Drafts only refresh the live captions, and the shared translator runs
/// one batch at a time: a draft started while other sessions' translations run or wait delays
/// their finals. So while it is busy a draft is skipped (the next partial transcript brings a
/// newer one), unless this session's captions have not moved for `refresh`: then it runs
/// anyway, so captions never look frozen. One gate per session.
class DraftGate {
public:
    using Clock = std::chrono::steady_clock;

    explicit DraftGate(bool skip_when_busy = true, std::chrono::milliseconds refresh = std::chrono::milliseconds(1000))
        : skip_when_busy_(skip_when_busy), refresh_(refresh) {}

    /// Whether a draft may run now, with the translator `busy`. Call ran() when it did.
    [[nodiscard]] bool allow(bool busy, Clock::time_point now) const noexcept {
        if (!busy || !skip_when_busy_) return true;
        return !last_.has_value() || now - *last_ >= refresh_;  // no draft yet: as if long ago
    }
    void ran(Clock::time_point now) noexcept { last_ = now; }

private:
    bool skip_when_busy_;
    std::chrono::milliseconds refresh_;
    std::optional<Clock::time_point> last_;  ///< this session's last draft that ran
};

}  // namespace ee
