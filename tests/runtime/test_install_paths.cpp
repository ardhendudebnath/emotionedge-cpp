#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "core/runtime/install_paths.hpp"

namespace ee {
namespace {

namespace fs = std::filesystem;

TEST(InstallPaths, DataDirSitsBesideTheBinDir) {
    const fs::path exe = fs::path("/opt/ee/bin/emotionedge").make_preferred();
    EXPECT_EQ(installed_data_dir(exe, "bin", "share/emotionedge"), fs::path("/opt/ee/share/emotionedge").make_preferred());
    // A deeper bin dir climbs one level per component; a trailing slash or "." adds none.
    const fs::path deep = fs::path("/opt/ee/x/bin/emotionedge").make_preferred();
    EXPECT_EQ(installed_data_dir(deep, "x/bin/", "share/emotionedge"), fs::path("/opt/ee/share/emotionedge").make_preferred());
    EXPECT_EQ(installed_data_dir(exe, "./bin", "share/emotionedge"), fs::path("/opt/ee/share/emotionedge").make_preferred());
}

TEST(InstallPaths, UnknownPrefixGivesNoDataDir) {
    const fs::path abs_data = fs::current_path() / "data";
    EXPECT_EQ(installed_data_dir("/opt/ee/bin/emotionedge", "bin", abs_data), abs_data);  // absolute: as is
    EXPECT_TRUE(installed_data_dir("", "bin", "share/emotionedge").empty());
    EXPECT_TRUE(installed_data_dir("/opt/ee/bin/emotionedge", fs::current_path() / "bin", "share/emotionedge").empty());
}

TEST(InstallPaths, ExecutablePathIsThisTestBinary) {
    const fs::path exe = executable_path();
    ASSERT_FALSE(exe.empty());
    EXPECT_TRUE(fs::exists(exe));
    EXPECT_NE(exe.filename().string().find("ee_tests"), std::string::npos) << exe;
}

TEST(InstallPaths, FirstExistingSkipsMissingAndEmpty) {
    const fs::path dir = fs::temp_directory_path() / ("ee_install_paths_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    fs::create_directories(dir);
    std::ofstream(dir / "b.yaml") << "x";
    EXPECT_EQ(first_existing({"", dir / "a.yaml", dir / "b.yaml", dir}), dir / "b.yaml");
    EXPECT_TRUE(first_existing({dir / "a.yaml", ""}).empty());
    fs::remove_all(dir);
}

}  // namespace
}  // namespace ee
