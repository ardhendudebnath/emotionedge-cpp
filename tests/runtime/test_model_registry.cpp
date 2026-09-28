#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "core/runtime/model_registry.hpp"
#include "core/runtime/sha256.hpp"

#if defined(EE_HAVE_ONNXRUNTIME)
#include "core/runtime/onnx.hpp"
#endif

namespace ee {
namespace {

namespace fs = std::filesystem;

class ModelRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = fs::temp_directory_path() /
               ("ee_models_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "_" +
                ::testing::UnitTest::GetInstance()->current_test_info()->name());
        fs::create_directories(dir_ / "asr");
        write(dir_ / "asr" / "tiny.bin", "whisper-weights");
        write(dir_ / "mt.bin", "nllb-weights");
    }
    void TearDown() override { fs::remove_all(dir_); }

    static void write(const fs::path& p, const std::string& text) {
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        out << text;
    }

    ModelRegistry registry(const std::string& asr_hash) const {
        const std::string manifest = R"({
          "schema": 1,
          "models": [
            {"id": "asr.tiny", "task": "asr", "format": "ggml", "path": "asr/tiny.bin",
             "sha256": ")" + asr_hash + R"(", "bytes": 15, "languages": ["*"]},
            {"id": "mt.small", "task": "mt", "format": "ct2", "path": "mt.bin", "languages": ["en", "hi"]},
            {"id": "mt.other", "task": "mt", "format": "ct2", "path": "missing.bin", "languages": ["en", "es"]}
          ],
          "device_profiles": {"cpu": {"execution_providers": ["cpu"], "threads": 4, "precision": "int8"}},
          "language_packs": {"en-hi": {"asr": "asr.tiny", "mt": "mt.small"}}
        })";
        return ModelRegistry::parse(manifest, dir_);
    }

    fs::path dir_;
};

TEST_F(ModelRegistryTest, VerifiesHashesAndReportsProblems) {
    const auto reg = registry(sha256_hex("whisper-weights"));
    ASSERT_EQ(reg.models().size(), 3u);

    EXPECT_EQ(reg.verify(*reg.find("asr.tiny")).status, VerifyStatus::Ok);
    const VerifyResult unpinned = reg.verify(*reg.find("mt.small"));
    EXPECT_EQ(unpinned.status, VerifyStatus::Unpinned);
    EXPECT_EQ(unpinned.actual_sha256, sha256_hex("nllb-weights"));
    EXPECT_EQ(reg.verify(*reg.find("mt.other")).status, VerifyStatus::Missing);

    const auto wrong = registry(sha256_hex("something else"));
    EXPECT_EQ(wrong.verify(*wrong.find("asr.tiny"), false).status, VerifyStatus::HashMismatch);

    write(dir_ / "asr" / "tiny.bin", "truncated");
    EXPECT_EQ(reg.verify(*reg.find("asr.tiny"), false).status, VerifyStatus::SizeMismatch);
}

TEST_F(ModelRegistryTest, WritesAndReusesAVerificationStamp) {
    const auto reg = registry(sha256_hex("whisper-weights"));
    EXPECT_EQ(reg.verify(*reg.find("asr.tiny")).status, VerifyStatus::Ok);
    EXPECT_TRUE(fs::exists(dir_ / "asr" / "tiny.bin.sha256"));
    EXPECT_EQ(reg.verify(*reg.find("asr.tiny")).status, VerifyStatus::Ok);
}

TEST_F(ModelRegistryTest, ResolvesThroughLanguagePacksThenLanguages) {
    const auto reg = registry("");
    EXPECT_EQ(reg.resolve("mt", "en", "hi")->id, "mt.small");
    EXPECT_EQ(reg.resolve("mt", "en", "es")->id, "mt.other");
    EXPECT_EQ(reg.resolve("asr", "fr", "de")->id, "asr.tiny");  // "*" covers every language
    EXPECT_EQ(reg.resolve("tts", "en", "hi"), nullptr);
    ASSERT_NE(reg.device_profile("cpu"), nullptr);
    EXPECT_EQ(reg.device_profile("cpu")->threads, 4);
    EXPECT_EQ(reg.device_profile("npu"), nullptr);
}

TEST_F(ModelRegistryTest, ResolvesStageModelParameters) {
    const auto reg = registry(sha256_hex("whisper-weights"));
    Params by_id;
    by_id.set("model_id", "asr.tiny");
    EXPECT_EQ(fs::path(resolve_model_path(by_id, &reg)), (dir_ / "asr" / "tiny.bin").lexically_normal());

    Params by_path;
    by_path.set("model", "/abs/model.onnx");
    EXPECT_EQ(resolve_model_path(by_path, nullptr), "/abs/model.onnx");
    EXPECT_EQ(resolve_model_path(Params{}, nullptr), "");

    Params missing;
    missing.set("model_id", "mt.other");
    EXPECT_THROW((void)resolve_model_path(missing, &reg), ConfigError);
    EXPECT_THROW((void)resolve_model_path(by_id, nullptr), ConfigError);
}

TEST(ModelRegistry, RejectsMalformedManifests) {
    EXPECT_THROW((void)ModelRegistry::parse("[]", "."), ConfigError);
    EXPECT_THROW((void)ModelRegistry::parse("{bad json", "."), ConfigError);
    EXPECT_THROW((void)ModelRegistry::parse(R"({"models": [{"id": "x"}]})", "."), ConfigError);
    EXPECT_THROW((void)ModelRegistry::parse(
                     R"({"models": [{"id": "x", "path": "a"}, {"id": "x", "path": "b"}]})", "."),
                 ConfigError);
    EXPECT_THROW((void)ModelRegistry::parse(R"({"language_packs": {"en-hi": {"mt": "nope"}}})", "."),
                 ConfigError);
}

#if defined(EE_HAVE_ONNXRUNTIME)
// Phase 4: one stage can run its models on different devices (emotion2vec+ on the GPU,
// DistilRoBERTa's INT8 on the CPU); the manifest's profile supplies the providers and the
// stage's own threads win; `auto` becomes what this ORT build has.
TEST(OnnxSessionConfig, ResolvesDevicesPerModel) {
    const auto reg = ModelRegistry::parse(R"({"device_profiles": {
        "cpu": {"execution_providers": ["cpu"], "threads": 4},
        "cuda": {"execution_providers": ["cuda", "cpu"], "threads": 2}}})", ".");
    Params p;
    p.set("device", "cpu");
    p.set("acoustic_device", "cuda");
    p.set("threads", "6");
    const onnx::SessionConfig acoustic = onnx::session_config(p, &reg, "acoustic_");
    EXPECT_EQ(acoustic.providers, (std::vector<std::string>{"cuda", "cpu"}));
    EXPECT_EQ(acoustic.intra_threads, 6);
    EXPECT_EQ(onnx::session_config(p, &reg, "lexical_").providers, std::vector<std::string>{"cpu"});

    Params profile_threads;
    profile_threads.set("device", "cuda");
    EXPECT_EQ(onnx::session_config(profile_threads, &reg).intra_threads, 2);

    Params automatic;
    automatic.set("device", "auto");
    EXPECT_EQ(onnx::session_config(automatic, &reg).providers.front(),
              onnx::provider_available("cuda") ? "cuda" : "cpu");
    EXPECT_TRUE(onnx::provider_available("cpu"));

    Params unprofiled;  // no manifest profile: the device names the provider
    unprofiled.set("device", "tensorrt");
    EXPECT_EQ(onnx::session_config(unprofiled, nullptr).providers, (std::vector<std::string>{"tensorrt", "cpu"}));
}
#endif

TEST(HotSwap, PublishesNewEnginesAtomically) {
    HotSwap<int> slot;
    EXPECT_EQ(slot.acquire(), nullptr);
    slot.publish(std::make_shared<int>(1));
    const auto held = slot.acquire();
    slot.publish(std::make_shared<int>(2));
    EXPECT_EQ(*held, 1);  // a stage keeps its engine until its next acquire()
    EXPECT_EQ(*slot.acquire(), 2);
}

}  // namespace
}  // namespace ee
