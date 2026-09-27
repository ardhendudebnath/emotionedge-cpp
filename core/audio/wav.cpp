#include "core/audio/wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

namespace ee {

namespace {

std::uint16_t le16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

std::uint32_t le32(const std::uint8_t* p) {
    return std::uint32_t{p[0]} | (std::uint32_t{p[1]} << 8) | (std::uint32_t{p[2]} << 16) |
           (std::uint32_t{p[3]} << 24);
}

void put16(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xff));
    out.push_back(static_cast<std::uint8_t>(v >> 8));
}

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
}

void put_tag(std::vector<std::uint8_t>& out, const char* tag) { out.insert(out.end(), tag, tag + 4); }

constexpr std::uint16_t kPcm = 1;
constexpr std::uint16_t kFloat = 3;
constexpr std::uint16_t kExtensible = 0xFFFE;

float decode_sample(const std::uint8_t* p, std::uint16_t format, std::uint16_t bits) {
    if (format == kPcm) {
        switch (bits) {
        case 8: return (static_cast<float>(p[0]) - 128.0f) / 128.0f;
        case 16: return static_cast<float>(static_cast<std::int16_t>(le16(p))) / 32768.0f;
        case 24: {
            std::int32_t v = static_cast<std::int32_t>(p[0] | (p[1] << 8) | (p[2] << 16));
            if ((v & 0x800000) != 0) v -= 0x1000000;
            return static_cast<float>(v) / 8388608.0f;
        }
        case 32: return static_cast<float>(static_cast<double>(static_cast<std::int32_t>(le32(p))) / 2147483648.0);
        default: break;
        }
    } else if (format == kFloat) {
        if (bits == 32) {
            const std::uint32_t raw = le32(p);
            float f = 0.0f;
            std::memcpy(&f, &raw, sizeof f);
            return f;
        }
        if (bits == 64) {
            std::uint64_t raw = 0;
            for (int i = 0; i < 8; ++i) raw |= std::uint64_t{p[i]} << (8 * i);
            double d = 0.0;
            std::memcpy(&d, &raw, sizeof d);
            return static_cast<float>(d);
        }
    }
    throw WavError("unsupported WAV encoding (format " + std::to_string(format) + ", " +
                   std::to_string(bits) + " bits)");
}

}  // namespace

WavData decode_wav(std::span<const std::uint8_t> b) {
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) != 0 || std::memcmp(b.data() + 8, "WAVE", 4) != 0) {
        throw WavError("not a RIFF/WAVE file");
    }
    std::uint16_t format = 0;
    std::uint16_t channels = 0;
    std::uint16_t bits = 0;
    std::uint32_t rate = 0;
    bool have_fmt = false;
    const std::uint8_t* data = nullptr;
    std::size_t data_size = 0;

    std::size_t pos = 12;
    while (pos + 8 <= b.size()) {
        const std::uint8_t* chunk = b.data() + pos;
        const std::uint32_t size = le32(chunk + 4);
        const std::size_t body = pos + 8;
        const std::size_t avail = b.size() - body;
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (size < 16 || avail < 16) throw WavError("truncated fmt chunk");
            const std::uint8_t* f = b.data() + body;
            format = le16(f);
            channels = le16(f + 2);
            rate = le32(f + 4);
            bits = le16(f + 14);
            if (format == kExtensible) {
                if (size < 40 || avail < 26) throw WavError("truncated WAVE_FORMAT_EXTENSIBLE header");
                format = le16(f + 24);  // first two bytes of the SubFormat GUID
            }
            have_fmt = true;
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data = b.data() + body;
            data_size = std::min<std::size_t>(size, avail);  // tolerate streamed/truncated files
            if (have_fmt) break;
        }
        if (size > avail) break;
        pos = body + size + (size & 1u);
    }
    if (!have_fmt || data == nullptr) throw WavError("WAV file lacks a fmt or data chunk");
    if (channels == 0 || rate == 0 || bits == 0 || bits % 8 != 0) throw WavError("invalid WAV fmt chunk");

    const std::size_t sample_bytes = bits / 8u;
    const std::size_t frame_bytes = sample_bytes * channels;
    const std::size_t frames = data_size / frame_bytes;

    WavData out;
    out.sample_rate = static_cast<int>(rate);
    out.channels = channels;
    out.samples.resize(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (std::size_t c = 0; c < channels; ++c) {
            sum += decode_sample(data + i * frame_bytes + c * sample_bytes, format, bits);
        }
        out.samples[i] = sum / static_cast<float>(channels);
    }
    return out;
}

WavData read_wav(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw WavError("cannot open '" + path.string() + "'");
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return decode_wav(bytes);
}

std::vector<std::uint8_t> encode_wav(std::span<const float> mono, int sample_rate, WavFormat format) {
    const std::uint16_t bits = format == WavFormat::Pcm16 ? 16 : 32;
    const std::uint32_t bytes_per_sample = bits / 8u;
    const auto data_size = static_cast<std::uint32_t>(mono.size() * bytes_per_sample);

    std::vector<std::uint8_t> out;
    out.reserve(44 + data_size);
    put_tag(out, "RIFF");
    put32(out, 36 + data_size);
    put_tag(out, "WAVE");
    put_tag(out, "fmt ");
    put32(out, 16);
    put16(out, format == WavFormat::Pcm16 ? kPcm : kFloat);
    put16(out, 1);
    put32(out, static_cast<std::uint32_t>(sample_rate));
    put32(out, static_cast<std::uint32_t>(sample_rate) * bytes_per_sample);
    put16(out, static_cast<std::uint16_t>(bytes_per_sample));
    put16(out, bits);
    put_tag(out, "data");
    put32(out, data_size);
    for (float s : mono) {
        if (format == WavFormat::Pcm16) {
            const float clamped = std::clamp(s, -1.0f, 1.0f);
            put16(out, static_cast<std::uint16_t>(static_cast<std::int16_t>(std::lround(clamped * 32767.0f))));
        } else {
            std::uint32_t raw = 0;
            std::memcpy(&raw, &s, sizeof raw);
            put32(out, raw);
        }
    }
    return out;
}

void write_wav(const std::filesystem::path& path, std::span<const float> mono, int sample_rate, WavFormat format) {
    const auto bytes = encode_wav(mono, sample_rate, format);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw WavError("cannot write '" + path.string() + "'");
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw WavError("write failed for '" + path.string() + "'");
}

}  // namespace ee
