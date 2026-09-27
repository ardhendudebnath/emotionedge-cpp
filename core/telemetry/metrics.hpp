#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ee::telemetry {

/// Lock-free log-linear histogram of non-negative integers (latencies in µs, scores ×1e4).
/// 32 sub-buckets per power of two bound the percentile error to about 3 %.
class Histogram {
public:
    static constexpr int kSubBits = 5;
    static constexpr std::size_t kSub = std::size_t{1} << kSubBits;
    static constexpr int kMaxExponent = 44;
    static constexpr std::size_t kBuckets = kSub + (kMaxExponent - kSubBits + 1) * kSub;

    struct Snapshot {
        std::uint64_t count = 0;
        std::uint64_t sum = 0;
        std::uint64_t min = 0;
        std::uint64_t max = 0;
        std::vector<std::uint64_t> buckets;

        [[nodiscard]] double mean() const { return count ? static_cast<double>(sum) / count : 0.0; }
        /// Value at quantile q in [0, 1] (bucket midpoint, clamped to the observed min/max).
        [[nodiscard]] std::uint64_t percentile(double q) const;
    };

    void record(std::uint64_t value) noexcept;
    [[nodiscard]] Snapshot snapshot() const;
    void reset() noexcept;

    [[nodiscard]] static std::size_t bucket_index(std::uint64_t value) noexcept;
    [[nodiscard]] static std::uint64_t bucket_lower(std::size_t index) noexcept;
    [[nodiscard]] static std::uint64_t bucket_upper(std::size_t index) noexcept;

private:
    std::array<std::atomic<std::uint64_t>, kBuckets> buckets_{};
    std::atomic<std::uint64_t> count_{0};
    std::atomic<std::uint64_t> sum_{0};
    std::atomic<std::uint64_t> min_{UINT64_MAX};
    std::atomic<std::uint64_t> max_{0};
};

class Counter {
public:
    void inc(std::uint64_t n = 1) noexcept { value_.fetch_add(n, std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t value() const noexcept { return value_.load(std::memory_order_relaxed); }

private:
    std::atomic<std::uint64_t> value_{0};
};

class Gauge {
public:
    void set(double v) noexcept { value_.store(v, std::memory_order_relaxed); }
    [[nodiscard]] double value() const noexcept { return value_.load(std::memory_order_relaxed); }

private:
    std::atomic<double> value_{0.0};
};

using Labels = std::vector<std::pair<std::string, std::string>>;

/// Owns every metric. Registration takes a lock (setup time only); updating a metric through
/// the returned reference is lock-free, which is what stages do on the hot path.
class MetricsRegistry {
public:
    Counter& counter(const std::string& name, const std::string& help, Labels labels = {});
    Gauge& gauge(const std::string& name, const std::string& help, Labels labels = {});
    /// Exported as a Prometheus summary; `scale` converts recorded integers to base units
    /// (1e-6 for µs -> seconds).
    Histogram& histogram(const std::string& name, const std::string& help, Labels labels = {},
                         double scale = 1e-6);

    /// Prometheus text exposition format (version 0.0.4).
    [[nodiscard]] std::string prometheus() const;

    struct HistogramView {
        std::string name;
        Labels labels;
        double scale;
        Histogram::Snapshot snapshot;
    };
    [[nodiscard]] std::vector<HistogramView> histograms() const;
    [[nodiscard]] const Counter* find_counter(const std::string& name, const Labels& labels = {}) const;

private:
    enum class Type { Counter, Gauge, Summary };
    struct Series {
        Labels labels;
        std::unique_ptr<Counter> counter;
        std::unique_ptr<Gauge> gauge;
        std::unique_ptr<Histogram> histogram;
    };
    struct Family {
        std::string help;
        Type type = Type::Counter;
        double scale = 1.0;
        std::deque<Series> series;
    };
    Series& series(const std::string& name, const std::string& help, Type type, Labels labels,
                   double scale);

    mutable std::mutex mutex_;
    std::map<std::string, Family> families_;
};

/// Writes `text` to `path` via a temporary file and rename, so readers never see partial files.
bool write_file_atomically(const std::string& path, const std::string& text);

}  // namespace ee::telemetry
