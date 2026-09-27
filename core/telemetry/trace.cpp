#include "core/telemetry/trace.hpp"

namespace ee::telemetry {

namespace {
void write_json_string(std::ofstream& out, std::string_view s) {
    out << '"';
    for (char c : s) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        default:
            if (static_cast<unsigned char>(c) >= 0x20) out << c;
        }
    }
    out << '"';
}
}  // namespace

TraceWriter::TraceWriter(const std::string& path) : out_(path, std::ios::binary | std::ios::trunc) {
    if (out_) out_ << "[\n";
}

TraceWriter::~TraceWriter() { close(); }

void TraceWriter::close() {
    if (closed_ || !out_) return;
    out_ << "\n]\n";
    out_.flush();
    closed_ = true;
}

void TraceWriter::separator() {
    if (!first_) out_ << ",\n";
    first_ = false;
}

void TraceWriter::write(const TraceEvent& e) {
    if (closed_ || !out_) return;
    separator();
    out_ << "{\"name\":";
    write_json_string(out_, e.name);
    out_ << ",\"cat\":";
    write_json_string(out_, e.category);
    out_ << ",\"ph\":\"X\",\"ts\":" << e.ts_us << ",\"dur\":" << e.dur_us << ",\"pid\":1,\"tid\":"
         << e.tid << ",\"args\":{\"utterance\":" << e.utterance << "}}";
}

void TraceWriter::thread_name(std::uint32_t tid, std::string_view name) {
    if (closed_ || !out_) return;
    separator();
    out_ << "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << tid << ",\"args\":{\"name\":";
    write_json_string(out_, name);
    out_ << "}}";
}

}  // namespace ee::telemetry
