// stb_truetype as ImGui bundles it, compiled once with external linkage for ShapedText, which
// rasterizes glyphs by index (ImGui's own copy is private to imgui_draw.cpp). Third-party code:
// built without our warning flags.
#define STB_TRUETYPE_IMPLEMENTATION
#include <imstb_truetype.h>
