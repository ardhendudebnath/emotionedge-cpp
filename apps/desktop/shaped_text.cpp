#include "apps/desktop/shaped_text.hpp"

#include <hb.h>
#include <imstb_truetype.h>

#include <imgui_internal.h>  // RegisterUserTexture, ImTextureDataQueueUpload

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ee::desktop {

struct ShapedText::Impl {
    struct Glyph {
        bool empty = true;  ///< nothing to draw (a space), or the atlas is full
        int x = 0, y = 0, w = 0, h = 0;  ///< in the atlas
        int x0 = 0, y0 = 0;  ///< bitmap offset from the pen, baseline-relative
    };

    std::vector<unsigned char> font_data;
    stbtt_fontinfo info{};
    hb_blob_t* blob = nullptr;
    hb_face_t* face = nullptr;
    hb_font_t* font = nullptr;
    hb_buffer_t* buffer = nullptr;
    float scale = 0.0f;  ///< pixels per font unit (HarfBuzz positions are in font units)
    float ascent = 0.0f;
    float line_height = 0.0f;
    ImTextureData atlas;
    bool registered = false;
    int pen_x = 1, pen_y = 1, row_h = 0;
    std::unordered_map<unsigned, Glyph> glyphs;

    /// The glyph's place in the atlas, rasterizing it on first use.
    const Glyph& glyph(unsigned index) {
        auto [it, inserted] = glyphs.try_emplace(index);
        Glyph& g = it->second;
        if (!inserted) return g;
        int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        stbtt_GetGlyphBitmapBox(&info, static_cast<int>(index), scale, scale, &x0, &y0, &x1, &y1);
        const int w = x1 - x0, h = y1 - y0;
        if (w <= 0 || h <= 0) return g;
        if (pen_x + w + 1 > atlas.Width) {
            pen_x = 1;
            pen_y += row_h + 1;
            row_h = 0;
        }
        if (pen_y + h + 1 > atlas.Height) return g;
        std::vector<unsigned char> coverage(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
        stbtt_MakeGlyphBitmap(&info, coverage.data(), w, h, w, scale, scale, static_cast<int>(index));
        for (int y = 0; y < h; ++y) {
            auto* row = static_cast<unsigned char*>(atlas.GetPixelsAt(pen_x, pen_y + y));
            for (int x = 0; x < w; ++x) row[4 * x + 3] = coverage[static_cast<std::size_t>(y * w + x)];
        }
        ImTextureDataQueueUpload(&atlas, pen_x, pen_y, w, h);
        g = {false, pen_x, pen_y, w, h, x0, y0};
        pen_x += w + 1;
        row_h = std::max(row_h, h);
        return g;
    }
};

ShapedText::ShapedText(const std::filesystem::path& font_file, float pixel_height) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    std::ifstream in(font_file, std::ios::binary);
    if (!in) throw std::runtime_error("font not found: " + font_file.string());
    m.font_data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (stbtt_InitFont(&m.info, m.font_data.data(), stbtt_GetFontOffsetForIndex(m.font_data.data(), 0)) == 0) {
        throw std::runtime_error("not a usable TrueType font: " + font_file.string());
    }
    m.scale = stbtt_ScaleForPixelHeight(&m.info, pixel_height);
    int ascent = 0, descent = 0, gap = 0;
    stbtt_GetFontVMetrics(&m.info, &ascent, &descent, &gap);
    m.ascent = static_cast<float>(ascent) * m.scale;
    m.line_height = static_cast<float>(ascent - descent + gap) * m.scale;

    m.blob = hb_blob_create(reinterpret_cast<const char*>(m.font_data.data()), static_cast<unsigned>(m.font_data.size()),
                            HB_MEMORY_MODE_READONLY, nullptr, nullptr);
    m.face = hb_face_create(m.blob, 0);
    m.font = hb_font_create(m.face);  // scale = units per em: positions come back in font units
    m.buffer = hb_buffer_create();

    // White everywhere, coverage in alpha: the draw color tints it.
    m.atlas.Create(ImTextureFormat_RGBA32, 1024, 1024);
    auto* p = static_cast<unsigned char*>(m.atlas.GetPixels());
    for (int i = 0; i < m.atlas.Width * m.atlas.Height; ++i) {
        p[4 * i] = p[4 * i + 1] = p[4 * i + 2] = 255;
        p[4 * i + 3] = 0;
    }
    ImGui::RegisterUserTexture(&m.atlas);
    m.registered = true;
}

ShapedText::~ShapedText() {
    release();
    Impl& m = *impl_;
    hb_buffer_destroy(m.buffer);
    hb_font_destroy(m.font);
    hb_face_destroy(m.face);
    hb_blob_destroy(m.blob);
}

void ShapedText::release() {
    if (impl_->registered && ImGui::GetCurrentContext() != nullptr) ImGui::UnregisterUserTexture(&impl_->atlas);
    impl_->registered = false;
}

float ShapedText::line_height() const noexcept { return impl_->line_height; }

void ShapedText::draw(const std::string& utf8, float wrap_width, ImU32 color) {
    Impl& m = *impl_;
    hb_buffer_clear_contents(m.buffer);
    hb_buffer_add_utf8(m.buffer, utf8.data(), static_cast<int>(utf8.size()), 0, static_cast<int>(utf8.size()));
    hb_buffer_guess_segment_properties(m.buffer);
    hb_shape(m.font, m.buffer, nullptr, 0);
    unsigned int n = 0;
    const hb_glyph_info_t* infos = hb_buffer_get_glyph_infos(m.buffer, &n);
    const hb_glyph_position_t* pos = hb_buffer_get_glyph_positions(m.buffer, &n);

    // Lay the glyphs out on lines, breaking after the last space once a line is too wide.
    struct Placed {
        unsigned glyph;
        float pen;  ///< where this glyph's advance starts
        float dx, dy;
        int line;
    };
    std::vector<Placed> placed;
    placed.reserve(n);
    float pen = 0.0f, widest = 0.0f;
    int line = 0;
    std::size_t line_start = 0;
    std::ptrdiff_t last_space = -1;
    for (unsigned i = 0; i < n; ++i) {
        placed.push_back({infos[i].codepoint, pen, static_cast<float>(pos[i].x_offset) * m.scale,
                          -static_cast<float>(pos[i].y_offset) * m.scale, line});
        pen += static_cast<float>(pos[i].x_advance) * m.scale;
        if (infos[i].cluster < utf8.size() && utf8[infos[i].cluster] == ' ') {
            last_space = static_cast<std::ptrdiff_t>(placed.size()) - 1;
        } else if (wrap_width > 0.0f && pen > wrap_width && last_space >= static_cast<std::ptrdiff_t>(line_start) &&
                   static_cast<std::size_t>(last_space) + 1 < placed.size()) {
            const auto first = static_cast<std::size_t>(last_space) + 1;
            widest = std::max(widest, placed[first - 1].pen);
            const float origin = placed[first].pen;
            ++line;
            for (std::size_t k = first; k < placed.size(); ++k) {
                placed[k].pen -= origin;
                placed[k].line = line;
            }
            pen -= origin;
            line_start = first;
            last_space = -1;
        }
    }
    widest = std::max(widest, pen);

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImTextureRef tex = m.atlas.GetTexRef();
    const float tw = static_cast<float>(m.atlas.Width), th = static_cast<float>(m.atlas.Height);
    for (const Placed& p : placed) {
        const Impl::Glyph& g = m.glyph(p.glyph);
        if (g.empty) continue;
        const float x = std::round(origin.x + p.pen + p.dx + static_cast<float>(g.x0));
        const float y = std::round(origin.y + static_cast<float>(p.line) * m.line_height + m.ascent + p.dy +
                                   static_cast<float>(g.y0));
        draw_list->AddImage(tex, ImVec2(x, y), ImVec2(x + static_cast<float>(g.w), y + static_cast<float>(g.h)),
                            ImVec2(static_cast<float>(g.x) / tw, static_cast<float>(g.y) / th),
                            ImVec2(static_cast<float>(g.x + g.w) / tw, static_cast<float>(g.y + g.h) / th), color);
    }
    ImGui::Dummy(ImVec2(std::max(1.0f, widest), static_cast<float>(line + 1) * m.line_height));
}

}  // namespace ee::desktop
