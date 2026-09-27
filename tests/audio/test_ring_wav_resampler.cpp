#include <gtest/gtest.h>

#include <filesystem>
#include <thread>

#include "core/audio/resampler.hpp"
#include "core/audio/ring_buffer.hpp"
#include "core/audio/wav.hpp"
#include "support/signals.hpp"

namespace ee {
namespace {

// ---- SpscRing ---------------------------------------------------------------------------------

TEST(SpscRing, WrapsAndReportsPartialWrites) {
    SpscRing<float> ring(8);
    const std::vector<float> a{1, 2, 3, 4, 5, 6};
    EXPECT_EQ(ring.write(a), 6u);
    std::vector<float> out(4);
    EXPECT_EQ(ring.read(out), 4u);
    EXPECT_EQ(out, (std::vector<float>{1, 2, 3, 4}));
    const std::vector<float> b{7, 8, 9, 10, 11, 12, 13};
    EXPECT_EQ(ring.write(b), 6u);  // only 6 free slots
    EXPECT_EQ(ring.read_available(), 8u);
    EXPECT_EQ(ring.skip(3), 3u);
    std::vector<float> rest(8);
    EXPECT_EQ(ring.read(rest), 5u);
    EXPECT_EQ(std::vector<float>(rest.begin(), rest.begin() + 5), (std::vector<float>{8, 9, 10, 11, 12}));
}

TEST(SpscRing, ConcurrentStreamingIsLossless) {
    SpscRing<float> ring(1024);
    constexpr std::size_t kTotal = 400'000;
    std::thread producer([&] {
        std::vector<float> block(96);
        std::size_t next = 0;
        while (next < kTotal) {
            const std::size_t n = std::min(block.size(), kTotal - next);
            for (std::size_t i = 0; i < n; ++i) block[i] = static_cast<float>((next + i) % 100000);
            std::size_t written = 0;
            while (written < n) {
                written += ring.write(std::span<const float>(block.data() + written, n - written));
                if (written < n) std::this_thread::yield();
            }
            next += n;
        }
    });
    std::vector<float> block(128);
    std::size_t seen = 0;
    bool ok = true;
    while (seen < kTotal) {
        const std::size_t n = ring.read(block);
        for (std::size_t i = 0; i < n; ++i) ok = ok && block[i] == static_cast<float>((seen + i) % 100000);
        seen += n;
        if (n == 0) std::this_thread::yield();
    }
    producer.join();
    EXPECT_TRUE(ok);
}

// ---- WAV --------------------------------------------------------------------------------------

TEST(Wav, Pcm16RoundTripWithinQuantization) {
    const auto signal = test::sine(440.0, 0.1, 16000, 0.8f);
    const WavData back = decode_wav(encode_wav(signal, 16000, WavFormat::Pcm16));
    EXPECT_EQ(back.sample_rate, 16000);
    EXPECT_EQ(back.channels, 1);
    ASSERT_EQ(back.samples.size(), signal.size());
    for (std::size_t i = 0; i < signal.size(); ++i) EXPECT_NEAR(back.samples[i], signal[i], 1.0f / 16000.0f);
}

TEST(Wav, Float32RoundTripIsExact) {
    const auto signal = test::sine(1000.0, 0.05, 24000, 0.9f);
    const WavData back = decode_wav(encode_wav(signal, 24000, WavFormat::Float32));
    EXPECT_EQ(back.samples, signal);
}

TEST(Wav, DecodesStereo24BitAndDownmixes) {
    // Hand-built 24-bit stereo file: L = +0.5, R = -0.25 -> mono 0.125.
    std::vector<std::uint8_t> bytes = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E',
                                       'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 2, 0,
                                       0x80, 0x3E, 0, 0, 0, 0, 0, 0, 6, 0, 24, 0,
                                       'd', 'a', 't', 'a', 6, 0, 0, 0,
                                       0x00, 0x00, 0x40, 0x00, 0x00, 0xE0};
    const WavData wav = decode_wav(bytes);
    EXPECT_EQ(wav.channels, 2);
    EXPECT_EQ(wav.sample_rate, 16000);
    ASSERT_EQ(wav.samples.size(), 1u);
    EXPECT_NEAR(wav.samples[0], 0.125f, 1e-6f);
}

TEST(Wav, RejectsGarbage) {
    const std::vector<std::uint8_t> junk{'n', 'o', 'p', 'e'};
    EXPECT_THROW((void)decode_wav(junk), WavError);
    EXPECT_THROW((void)read_wav("/definitely/not/here.wav"), WavError);
}

TEST(Wav, WritesAndReadsFiles) {
    const auto path = std::filesystem::temp_directory_path() / "ee_wav_test.wav";
    const auto signal = test::sine(300.0, 0.02, 8000, 0.5f);
    write_wav(path, signal, 8000, WavFormat::Float32);
    EXPECT_EQ(read_wav(path).samples, signal);
    std::filesystem::remove(path);
}

// ---- Resampler --------------------------------------------------------------------------------

struct RateCase {
    int in;
    int out;
};

class ResamplerRates : public ::testing::TestWithParam<RateCase> {};

TEST_P(ResamplerRates, PreservesToneFrequencyAndLevel) {
    const auto [in_rate, out_rate] = GetParam();
    const auto input = test::sine(1000.0, 0.5, in_rate, 0.5f);
    const auto output = Resampler::convert(input, in_rate, out_rate);
    const auto expected = static_cast<std::size_t>(std::ceil(static_cast<double>(input.size()) * out_rate / in_rate));
    EXPECT_EQ(output.size(), expected);
    const std::size_t margin = static_cast<std::size_t>(out_rate / 20);  // skip filter edges
    EXPECT_NEAR(test::zero_crossing_hz(output, out_rate, margin, output.size() - margin), 1000.0, 2.0);
    EXPECT_NEAR(test::rms(output, margin, output.size() - margin), 0.5 / std::sqrt(2.0), 0.005);
}

INSTANTIATE_TEST_SUITE_P(CommonRates, ResamplerRates,
                         ::testing::Values(RateCase{48000, 16000}, RateCase{44100, 16000}, RateCase{16000, 24000},
                                           RateCase{22050, 24000}, RateCase{24000, 16000}, RateCase{24000, 48000}));

TEST(Resampler, RejectsContentAboveTheNewNyquist) {
    const auto input = test::sine(11000.0, 0.5, 48000, 0.5f);
    const auto output = Resampler::convert(input, 48000, 16000);
    const std::size_t margin = 800;
    EXPECT_LT(test::rms(output, margin, output.size() - margin), 0.5 * 1e-3);  // > 60 dB down
}

TEST(Resampler, StreamingMatchesOneShot) {
    const auto input = test::sine(700.0, 0.3, 22050, 0.4f);
    const auto whole = Resampler::convert(input, 22050, 24000);
    Resampler r(22050, 24000);
    std::vector<float> streamed;
    std::size_t pos = 0;
    std::size_t chunk = 1;
    while (pos < input.size()) {
        const std::size_t n = std::min(chunk, input.size() - pos);
        r.process(std::span<const float>(input.data() + pos, n), streamed);
        pos += n;
        chunk = chunk * 3 % 997 + 1;  // irregular chunk sizes
    }
    r.flush(streamed);
    ASSERT_EQ(streamed.size(), whole.size());
    for (std::size_t i = 0; i < whole.size(); ++i) ASSERT_FLOAT_EQ(streamed[i], whole[i]) << "sample " << i;
}

TEST(Resampler, SameRateIsPassThrough) {
    const auto input = test::sine(440.0, 0.01, 16000);
    EXPECT_EQ(Resampler::convert(input, 16000, 16000), input);
}

}  // namespace
}  // namespace ee
