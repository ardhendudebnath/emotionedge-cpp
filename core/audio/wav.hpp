#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

namespace ee {

struct WavError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct WavData {
    int sample_rate = 0;
    int channels = 0;            ///< channels in the file; `samples` is always mono
    std::vector<float> samples;  ///< mono, [-1, 1]
};

enum class WavFormat { Pcm16, Float32 };

/// Decodes RIFF/WAVE: PCM 8/16/24/32-bit, IEEE float 32/64, and WAVE_FORMAT_EXTENSIBLE.
/// Multi-channel audio is mixed down to mono. Throws WavError.
[[nodiscard]] WavData decode_wav(std::span<const std::uint8_t> bytes);
[[nodiscard]] WavData read_wav(const std::filesystem::path& path);

[[nodiscard]] std::vector<std::uint8_t> encode_wav(std::span<const float> mono, int sample_rate,
                                                   WavFormat format = WavFormat::Pcm16);
void write_wav(const std::filesystem::path& path, std::span<const float> mono, int sample_rate,
               WavFormat format = WavFormat::Pcm16);

}  // namespace ee
