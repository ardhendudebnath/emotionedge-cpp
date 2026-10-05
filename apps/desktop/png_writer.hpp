#pragma once

#include <filesystem>
#include <vector>

namespace ee::desktop {

/// Writes 8-bit RGBA pixels (top row first) as a PNG. Deflate "stored" blocks: no zlib needed,
/// and the files are large, which is fine for screenshots.
bool write_png(const std::filesystem::path& path, const std::vector<unsigned char>& rgba, int width, int height);

}  // namespace ee::desktop
