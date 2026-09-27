#include "core/telemetry/metrics.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

namespace ee::telemetry {

// ---- Histogram --------------------------------------------------------------------------------

std::size_t Histogram::bucket_index(std::uint64_t value) noexcept {
    if (value < kSub) return static_cast<std::size_t>(value);
    const int exponent = 63 - std::countl_zero(value);  // floor(log2(value)) >= kSubBits
    if (exponent > kMaxExponent) return kBuckets - 1;
    const std::uint64_t sub = (value >> (exponent - kSubBits)) - kSub;
    return kSub + static_cast<std::size_t>(exponent - kSubBits) * kSub + static_cast<std::size_t>(sub);
}

std::uint64_t Histogram::bucket_lower(std::size_t index) noexcept {
    if (index < kSub) return index;
    const std::size_t j = index - kSub;
    const int shift = static_cast<int>(j / kSub);
    const std::uint64_t sub = j % kSub;
    return (kSub + sub) << shift;
}

std::uint64_t Histogram::bucket_upper(std::size_t index) noexcept {
    if (index < kSub) return index + 1;
    const int shift = static_cast<int>((index - kSub) / kSub);
    return bucket_lower(index) + (std::uint64_t{1} << shift);
}

void Histogram::record(std::uint64_t value) noexcept {
    buckets_[bucket_index(value)].fetch_add(1, std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);
    sum_.fetch_add(value, std::memory_order_relaxed);
    std::uint64_t cur = min_.load(std::memory_order_relaxed);
    while (value < cur && !min_.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
    }
    cur = max_.load(std::memory_order_relaxed);
    while (value > cur && !max_.compare_exchange_weak(cur, value, std::memory_order_relaxed)) {
    }
}

Histogram::Snapshot Histogram::snapshot() const {
    Snapshot s;
    s.buckets.resize(kBuckets);
    for (std::size_t i = 0; i < kBuckets; ++i) s.buckets[i] = buckets_[i].load(std::memory_order_relaxed);
    s.count = count_.load(std::memory_order_relaxed);
    s.sum = sum_.load(std::memory_order_relaxed);
    s.min = s.count ? min_.load(std::memory_order_relaxed) : 0;
    s.max = max_.load(std::memory_order_relaxed);
    return s;
}

void Histogram::reset() noexcept {
    for (auto& b : buckets_) b.store(0, std::memory_order_relaxed);
    count_.store(0, std::memory_order_relaxed);
    sum_.store(0, std::memory_order_relaxed);
    min_.store(UINT64_MAX, std::memory_order_relaxed);
    max_.store(0, std::memory_order_relaxed);
}

std::uint64_t Histogram::Snapshot::percentile(double q) const {
    if (count == 0) return 0;
    q = std::clamp(q, 0.0, 1.0);
    const auto rank = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::ceil(q * count)));
    std::uint64_t seen = 0;
    for (std::size_t i = 0; i < buckets.size(); ++i) {
        seen += buckets[i];
        if (seen >= rank) {
            const std::uint64_t lo = bucket_lower(i);
            const std::uint64_t hi = bucket_upper(i);
            const std::uint64_t mid = lo + (hi - lo - 1) / 2;
            return std::clamp(mid, min, max);
        }
    }
    return max;
}

// ---- Registry ---------------------------------------------------------------------------------

namespace {

Labels sorted(Labels labels) {
    std::sort(labels.begin(), labels.end());
    return labels;
}

std::string escape_label(const std::string& v) {
    std::string out;
    out.reserve(v.size());
    for (char c : v) {
        if (c == '\\' || c == '"') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out += c;
        }
    }
    return out;
}

std::string format_labels(const Labels& labels, const char* extra_key = nullptr,
                          const char* extra_value = nullptr) {
    if (labels.empty() && extra_key == nullptr) return {};
    std::string out = "{";
    bool first = true;
    for (const auto& [k, v] : labels) {
        if (!first) out += ',';
        out += k + "=\"" + escape_label(v) + "\"";
        first = false;
    }
    if (extra_key != nullptr) {
        if (!first) out += ',';
        out += std::string(extra_key) + "=\"" + extra_value + "\"";
    }
    out += '}';
    return out;
}

std::string number(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9g", v);
    return buf;
}

}  // namespace

MetricsRegistry::Series& MetricsRegistry::series(const std::string& name, const std::string& help,
                                                 Type type, Labels labels, double scale) {
    labels = sorted(std::move(labels));
    std::lock_guard lock(mutex_);
    auto [it, inserted] = families_.try_emplace(name);
    Family& fam = it->second;
    if (inserted) {
        fam.help = help;
        fam.type = type;
        fam.scale = scale;
    }
    for (Series& s : fam.series) {
        if (s.labels == labels) return s;
    }
    Series& s = fam.series.emplace_back();
    s.labels = std::move(labels);
    switch (fam.type) {
    case Type::Counter: s.counter = std::make_unique<Counter>(); break;
    case Type::Gauge: s.gauge = std::make_unique<Gauge>(); break;
    case Type::Summary: s.histogram = std::make_unique<Histogram>(); break;
    }
    return s;
}

Counter& MetricsRegistry::counter(const std::string& name, const std::string& help, Labels labels) {
    Series& s = series(name, help, Type::Counter, std::move(labels), 1.0);
    if (!s.counter) throw std::logic_error("metric '" + name + "' is not a counter");
    return *s.counter;
}

Gauge& MetricsRegistry::gauge(const std::string& name, const std::string& help, Labels labels) {
    Series& s = series(name, help, Type::Gauge, std::move(labels), 1.0);
    if (!s.gauge) throw std::logic_error("metric '" + name + "' is not a gauge");
    return *s.gauge;
}

Histogram& MetricsRegistry::histogram(const std::string& name, const std::string& help,
                                      Labels labels, double scale) {
    Series& s = series(name, help, Type::Summary, std::move(labels), scale);
    if (!s.histogram) throw std::logic_error("metric '" + name + "' is not a summary");
    return *s.histogram;
}

std::string MetricsRegistry::prometheus() const {
    std::lock_guard lock(mutex_);
    std::ostringstream os;
    for (const auto& [name, fam] : families_) {
        const char* type = fam.type == Type::Counter ? "counter"
                           : fam.type == Type::Gauge ? "gauge"
                                                     : "summary";
        os << "# HELP " << name << ' ' << fam.help << '\n';
        os << "# TYPE " << name << ' ' << type << '\n';
        for (const Series& s : fam.series) {
            if (s.counter) {
                os << name << format_labels(s.labels) << ' ' << s.counter->value() << '\n';
            } else if (s.gauge) {
                os << name << format_labels(s.labels) << ' ' << number(s.gauge->value()) << '\n';
            } else if (s.histogram) {
                static constexpr std::pair<double, const char*> kQuantiles[] = {
                    {0.5, "0.5"}, {0.95, "0.95"}, {0.99, "0.99"}};
                const auto snap = s.histogram->snapshot();
                for (const auto& [q, qs] : kQuantiles) {
                    os << name << format_labels(s.labels, "quantile", qs) << ' '
                       << number(static_cast<double>(snap.percentile(q)) * fam.scale) << '\n';
                }
                os << name << "_sum" << format_labels(s.labels) << ' '
                   << number(static_cast<double>(snap.sum) * fam.scale) << '\n';
                os << name << "_count" << format_labels(s.labels) << ' ' << snap.count << '\n';
            }
        }
    }
    return os.str();
}

std::vector<MetricsRegistry::HistogramView> MetricsRegistry::histograms() const {
    std::lock_guard lock(mutex_);
    std::vector<HistogramView> out;
    for (const auto& [name, fam] : families_) {
        for (const Series& s : fam.series) {
            if (s.histogram) out.push_back({name, s.labels, fam.scale, s.histogram->snapshot()});
        }
    }
    return out;
}

const Counter* MetricsRegistry::find_counter(const std::string& name, const Labels& labels) const {
    const Labels key = sorted(labels);
    std::lock_guard lock(mutex_);
    const auto it = families_.find(name);
    if (it == families_.end()) return nullptr;
    for (const Series& s : it->second.series) {
        if (s.labels == key) return s.counter.get();
    }
    return nullptr;
}

bool write_file_atomically(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << text;
        if (!out) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

}  // namespace ee::telemetry
