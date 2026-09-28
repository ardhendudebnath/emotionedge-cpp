#pragma once

#include <string>
#include <string_view>

namespace ee {

/// Removes emoji and pictographic symbols before text reaches espeak-ng. espeak-ng 1.51 overflows
/// a buffer on some emoji sequences, e.g. "❤️" (U+2764 U+FE0F), and crashes the process (seen on
/// NLLB output from Reddit-style text). Speech translation has nothing to say for them anyway.
/// Removed: U+2600–27BF (symbols, dingbats), U+2B00–2BFF, U+FE00–FE0F (variation selectors),
/// U+1F000–1FAFF (emoji, flags), U+E0000–E007F (emoji tags), and U+200D only between two
/// removed symbols (emoji ZWJ sequences). A ZWJ inside Indic words stays. Whitespace left
/// behind is collapsed.
[[nodiscard]] std::string remove_emoji(std::string_view text);

}  // namespace ee
