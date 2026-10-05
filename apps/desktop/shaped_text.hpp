#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <imgui.h>

namespace ee::desktop {

/// Draws text that needs shaping, which Dear ImGui does not do: Devanagari reorders vowel signs
/// and joins consonants into conjuncts. HarfBuzz shapes each string into positioned glyphs. ImGui's
/// bundled stb_truetype rasterizes them by glyph index into an atlas texture, which ImGui's
/// renderer backend uploads (a user texture).
class ShapedText {
public:
    ShapedText(const std::filesystem::path& font_file, float pixel_height);
    ~ShapedText();
    ShapedText(const ShapedText&) = delete;
    ShapedText& operator=(const ShapedText&) = delete;

    /// Draws `utf8` at the cursor, wrapped at spaces to `wrap_width`, tinted `color`, and moves
    /// the cursor below it.
    void draw(const std::string& utf8, float wrap_width, ImU32 color);
    [[nodiscard]] float line_height() const noexcept;
    /// Call after the renderer backend has shut down (it owns the GPU copy of the atlas).
    void release();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ee::desktop
