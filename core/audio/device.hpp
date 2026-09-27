#pragma once

// Live audio I/O (blueprint 1.1 "miniaudio" and T0 "audio I/O callback, real-time prio").
#if !defined(EE_HAVE_MINIAUDIO)
#error "core/audio/device.hpp needs a build with -DEE_WITH_MINIAUDIO=ON"
#endif

#include <memory>
#include <string>
#include <vector>

#include "core/audio/audio_io.hpp"

namespace ee {

struct AudioDeviceInfo {
    std::string name;
    bool is_default = false;
    bool capture = false;  ///< false = playback device
};

[[nodiscard]] std::vector<AudioDeviceInfo> list_audio_devices();

/// Opens a capture and/or a playback device (16 kHz mono f32 in, TTS rate mono f32 out; the
/// backend converts to the hardware format). The callbacks only move samples between the
/// hardware and the lock-free rings: no locks, no allocation, no logging.
class AudioDevice {
public:
    struct Config {
        int capture_rate = 16000;
        int playback_rate = 24000;
        std::string capture_device;   ///< substring of a device name; empty = system default
        std::string playback_device;
        unsigned period_ms = 10;
    };

    /// Either pointer may be null to open only one direction.
    AudioDevice(Config config, RingSource* capture, RingSink* playback);
    ~AudioDevice();
    AudioDevice(const AudioDevice&) = delete;
    AudioDevice& operator=(const AudioDevice&) = delete;

    void start();
    void stop();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace ee
