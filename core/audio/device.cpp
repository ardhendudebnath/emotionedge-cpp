#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DECODING
#define MA_NO_GENERATION
#include <miniaudio.h>

#include <stdexcept>

#include "core/audio/device.hpp"

namespace ee {

struct AudioDevice::Impl {
    ma_context context{};
    ma_device capture_device{};
    ma_device playback_device{};
    bool context_ready = false;
    bool capture_ready = false;
    bool playback_ready = false;
    RingSource* capture = nullptr;
    RingSink* playback = nullptr;
};

namespace {

void on_capture(ma_device* device, void* output, const void* input, ma_uint32 frames) {
    (void)output;
    auto* impl = static_cast<AudioDevice::Impl*>(device->pUserData);
    impl->capture->push(std::span<const float>(static_cast<const float*>(input), frames));
}

void on_playback(ma_device* device, void* output, const void* input, ma_uint32 frames) {
    (void)input;
    auto* impl = static_cast<AudioDevice::Impl*>(device->pUserData);
    impl->playback->pull(std::span<float>(static_cast<float*>(output), frames));
}

/// Finds a device whose name contains `wanted`; nullptr (= default device) if `wanted` is empty.
const ma_device_id* find_device(ma_context& context, const std::string& wanted, bool capture) {
    if (wanted.empty()) return nullptr;
    ma_device_info* playback = nullptr;
    ma_device_info* captures = nullptr;
    ma_uint32 playback_count = 0;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(&context, &playback, &playback_count, &captures, &capture_count) != MA_SUCCESS) {
        throw std::runtime_error("cannot enumerate audio devices");
    }
    ma_device_info* list = capture ? captures : playback;
    const ma_uint32 count = capture ? capture_count : playback_count;
    for (ma_uint32 i = 0; i < count; ++i) {
        if (std::string(list[i].name).find(wanted) != std::string::npos) return &list[i].id;
    }
    throw std::runtime_error("no " + std::string(capture ? "capture" : "playback") + " device matches '" + wanted + "'");
}

}  // namespace

std::vector<AudioDeviceInfo> list_audio_devices() {
    ma_context context;
    if (ma_context_init(nullptr, 0, nullptr, &context) != MA_SUCCESS) throw std::runtime_error("cannot initialize audio backend");
    std::vector<AudioDeviceInfo> out;
    ma_device_info* playback = nullptr;
    ma_device_info* captures = nullptr;
    ma_uint32 playback_count = 0;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(&context, &playback, &playback_count, &captures, &capture_count) == MA_SUCCESS) {
        for (ma_uint32 i = 0; i < capture_count; ++i) out.push_back({captures[i].name, captures[i].isDefault != 0, true});
        for (ma_uint32 i = 0; i < playback_count; ++i) out.push_back({playback[i].name, playback[i].isDefault != 0, false});
    }
    ma_context_uninit(&context);
    return out;
}

AudioDevice::AudioDevice(Config config, RingSource* capture, RingSink* playback) : impl_(std::make_unique<Impl>()) {
    impl_->capture = capture;
    impl_->playback = playback;
    if (ma_context_init(nullptr, 0, nullptr, &impl_->context) != MA_SUCCESS) {
        throw std::runtime_error("cannot initialize audio backend");
    }
    impl_->context_ready = true;

    if (capture != nullptr) {
        ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
        cfg.capture.format = ma_format_f32;
        cfg.capture.channels = 1;
        cfg.capture.pDeviceID = find_device(impl_->context, config.capture_device, true);
        cfg.sampleRate = static_cast<ma_uint32>(config.capture_rate);
        cfg.periodSizeInMilliseconds = config.period_ms;
        cfg.dataCallback = on_capture;
        cfg.pUserData = impl_.get();
        if (ma_device_init(&impl_->context, &cfg, &impl_->capture_device) != MA_SUCCESS) {
            throw std::runtime_error("cannot open capture device");
        }
        impl_->capture_ready = true;
    }
    if (playback != nullptr) {
        ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
        cfg.playback.format = ma_format_f32;
        cfg.playback.channels = 1;
        cfg.playback.pDeviceID = find_device(impl_->context, config.playback_device, false);
        cfg.sampleRate = static_cast<ma_uint32>(config.playback_rate);
        cfg.periodSizeInMilliseconds = config.period_ms;
        cfg.dataCallback = on_playback;
        cfg.pUserData = impl_.get();
        if (ma_device_init(&impl_->context, &cfg, &impl_->playback_device) != MA_SUCCESS) {
            throw std::runtime_error("cannot open playback device");
        }
        impl_->playback_ready = true;
    }
}

AudioDevice::~AudioDevice() {
    stop();
    if (impl_->capture_ready) ma_device_uninit(&impl_->capture_device);
    if (impl_->playback_ready) ma_device_uninit(&impl_->playback_device);
    if (impl_->context_ready) ma_context_uninit(&impl_->context);
}

void AudioDevice::start() {
    if (impl_->playback_ready && ma_device_start(&impl_->playback_device) != MA_SUCCESS) {
        throw std::runtime_error("cannot start playback device");
    }
    if (impl_->capture_ready && ma_device_start(&impl_->capture_device) != MA_SUCCESS) {
        throw std::runtime_error("cannot start capture device");
    }
}

void AudioDevice::stop() {
    if (impl_->capture_ready) ma_device_stop(&impl_->capture_device);
    if (impl_->playback_ready) ma_device_stop(&impl_->playback_device);
}

}  // namespace ee
