#include "core/emotion/prosody_features.hpp"

#include <algorithm>
#include <cmath>

#include "core/audio/dsp.hpp"

namespace ee {

namespace {
PitchConfig pitch_config(int rate) {
    PitchConfig cfg;
    cfg.sample_rate = rate;
    cfg.frame = rate * 40 / 1000;
    cfg.hop = rate * 10 / 1000;
    return cfg;
}
}  // namespace

ProsodyTracker::ProsodyTracker(int sample_rate)
    : rate_(sample_rate), frame_(static_cast<std::size_t>(sample_rate * 40 / 1000)),
      hop_(static_cast<std::size_t>(sample_rate * 10 / 1000)), yin_(pitch_config(sample_rate)) {
    reset();
}

void ProsodyTracker::reset() {
    // Half a frame of leading zeros centers frame i on t = i · hop.
    buffer_.assign(frame_ / 2, 0.0f);
    consumed_ = 0;
    pushed_ = 0;
    frames_.clear();
}

void ProsodyTracker::push(std::span<const float> audio) {
    buffer_.insert(buffer_.end(), audio.begin(), audio.end());
    pushed_ += audio.size();
    analyze_ready(false);
}

void ProsodyTracker::flush() { analyze_ready(true); }

void ProsodyTracker::analyze_one(std::span<const float> frame) {
    const PitchFrame pf = yin_.analyze(frame);
    ProsodyFrame out;
    out.f0_hz = pf.voiced ? pf.f0_hz : 0.0f;
    out.voiced = pf.voiced;
    out.energy_db = rms_dbfs(frame);
    out.hf_db = out.energy_db > -70.0f ? hf_ratio_db(frame) : 0.0f;
    frames_.push_back(out);
}

void ProsodyTracker::analyze_ready(bool flushing) {
    const std::size_t total = consumed_ + buffer_.size();  // absolute, including the zero prefix
    while (true) {
        const std::size_t start = frames_.size() * hop_;  // absolute start == center in pushed-audio time
        if (start + frame_ <= total) {
            analyze_one(std::span<const float>(buffer_.data() + (start - consumed_), frame_));
        } else if (flushing && start < pushed_) {
            scratch_.assign(frame_, 0.0f);
            const std::size_t have = total > start ? total - start : 0;
            std::copy_n(buffer_.begin() + static_cast<std::ptrdiff_t>(start - consumed_), have, scratch_.begin());
            analyze_one(scratch_);
        } else {
            break;
        }
    }
    const std::size_t next_start = frames_.size() * hop_;
    if (next_start > consumed_) {
        const std::size_t drop = std::min(next_start - consumed_, buffer_.size());
        buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(drop));
        consumed_ += drop;
    }
}

ProsodySummary summarize(std::span<const ProsodyFrame> frames, float hop_s) {
    ProsodySummary s;
    if (frames.empty()) return s;
    float max_energy = -120.0f;
    for (const ProsodyFrame& f : frames) max_energy = std::max(max_energy, f.energy_db);
    const float active_threshold = std::max(max_energy - 30.0f, -60.0f);
    std::size_t first = frames.size();
    std::size_t last = 0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].energy_db >= active_threshold) {
            first = std::min(first, i);
            last = i;
        }
    }
    if (first >= frames.size()) return s;

    const std::size_t span = last - first + 1;
    s.duration_s = static_cast<float>(span) * hop_s;
    s.energy_peak_db = max_energy;

    double energy_sum = 0.0;
    double hf_sum = 0.0;
    std::vector<float> f0;
    for (std::size_t i = first; i <= last; ++i) {
        const ProsodyFrame& f = frames[i];
        if (f.energy_db < active_threshold) continue;
        ++s.active_frames;
        energy_sum += f.energy_db;
        if (f.voiced) {
            ++s.voiced_frames;
            hf_sum += f.hf_db;
            f0.push_back(f.f0_hz);
        }
    }
    s.energy_db = static_cast<float>(energy_sum / s.active_frames);
    s.pause_ratio = 1.0f - static_cast<float>(s.active_frames) / static_cast<float>(span);
    s.voiced_ratio = static_cast<float>(s.voiced_frames) / static_cast<float>(s.active_frames);

    if (!f0.empty()) {
        s.hf_db = static_cast<float>(hf_sum / static_cast<double>(f0.size()));
        std::vector<float> sorted = f0;
        std::sort(sorted.begin(), sorted.end());
        s.f0_median_hz = sorted[sorted.size() / 2];
        s.f0_range_st = semitones(sorted[sorted.size() * 9 / 10], sorted[sorted.size() / 10]);
    }

    // Frame-level jitter proxy: relative F0 change between consecutive voiced frames.
    double jitter_sum = 0.0;
    int jitter_n = 0;
    for (std::size_t i = first + 1; i <= last; ++i) {
        if (frames[i].voiced && frames[i - 1].voiced && frames[i - 1].f0_hz > 0.0f) {
            jitter_sum += std::abs(frames[i].f0_hz - frames[i - 1].f0_hz) / frames[i - 1].f0_hz;
            ++jitter_n;
        }
    }
    s.jitter = jitter_n > 0 ? static_cast<float>(jitter_sum / jitter_n) : 0.0f;

    // Final contour: least-squares F0 slope over the last 300 ms of voicing.
    if (s.f0_median_hz > 0.0f) {
        std::size_t last_voiced = last;
        while (last_voiced > first && !frames[last_voiced].voiced) --last_voiced;
        const auto window = static_cast<std::size_t>(0.3f / hop_s);
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        int n = 0;
        for (std::size_t i = last_voiced + 1 > window ? last_voiced + 1 - window : 0; i <= last_voiced; ++i) {
            if (!frames[i].voiced) continue;
            const double x = static_cast<double>(i) * hop_s;
            const double y = semitones(frames[i].f0_hz, s.f0_median_hz);
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            ++n;
        }
        const double denom = n * sxx - sx * sx;
        if (n >= 5 && std::abs(denom) > 1e-12) s.final_slope_st_s = static_cast<float>((n * sxy - sx * sy) / denom);
    }

    // Syllable nuclei: peaks of the 50 ms-smoothed envelope that rise and fall by >= 3 dB.
    std::vector<float> env(span);
    for (std::size_t i = 0; i < span; ++i) {
        double acc = 0.0;
        int n = 0;
        for (std::size_t k = (i >= 2 ? i - 2 : 0); k <= std::min(span - 1, i + 2); ++k) {
            acc += std::max(frames[first + k].energy_db, active_threshold - 10.0f);
            ++n;
        }
        env[i] = static_cast<float>(acc / n);
    }
    int peaks = 0;
    bool rising = true;
    float valley = env[0];
    float peak = env[0];
    for (float e : env) {
        if (rising) {
            peak = std::max(peak, e);
            if (peak - e >= 3.0f && peak - valley >= 3.0f) {
                ++peaks;
                rising = false;
                valley = e;
            }
        } else {
            valley = std::min(valley, e);
            if (e - valley >= 3.0f) {
                rising = true;
                peak = e;
            }
        }
    }
    if (rising && peak - valley >= 3.0f) ++peaks;  // the utterance ended on a nucleus
    s.syllable_rate = s.duration_s > 0.0f ? static_cast<float>(peaks) / s.duration_s : 0.0f;
    return s;
}

}  // namespace ee
