#pragma once
// Where an installed copy keeps its data. `cmake --install` puts the apps in <prefix>/bin and
// what they read (configs, the model manifest, the desktop font) in <prefix>/share/emotionedge
// (cmake/Install.cmake). The apps find that directory from their own location, so an install
// works without the source or build tree.
#include <filesystem>
#include <initializer_list>

namespace ee {

/// The running executable's path, or empty where the platform can't say.
std::filesystem::path executable_path();

/// The data directory of an install whose executable is at `executable`, given the install's
/// relative bin and data directories (CMAKE_INSTALL_BINDIR, CMAKE_INSTALL_DATADIR/emotionedge):
/// /opt/ee/bin/emotionedge with "bin" and "share/emotionedge" gives /opt/ee/share/emotionedge.
/// An absolute `data_dir` is returned as is. With an absolute `bin_dir` the prefix is unknown,
/// and so is an empty `executable`: both give an empty path.
std::filesystem::path installed_data_dir(const std::filesystem::path& executable,
                                         const std::filesystem::path& bin_dir,
                                         const std::filesystem::path& data_dir);

/// The first of `candidates` that exists, or an empty path. Empty candidates are skipped.
std::filesystem::path first_existing(std::initializer_list<std::filesystem::path> candidates);

}  // namespace ee
