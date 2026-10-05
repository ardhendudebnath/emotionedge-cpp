// emotionedge-desktop: the desktop app (blueprint 5.3 "CLI / desktop"). One window with live
// captions: what was said, the emotion read from the voice, the translation as it is drafted and
// finalized, and the emotion consistency of the speech that played. Plus the latency budget.
//
//   emotionedge-desktop [--config FILE] [--set KEY=VALUE]... [--input speech.wav] [--mute]
//                       [--capture-device NAME] [--playback-device NAME] [--font FILE]
//                       [--screenshot out.png [--after SECONDS]]
//
// --screenshot renders a hidden window, saves one frame (by default 1 s after the session
// ends) and exits: how the UI is checked without a person at the screen.
#include <GLFW/glfw3.h>  // GLFW_INCLUDE_NONE comes with ee::imgui: no system GL headers
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "apps/desktop/desktop_session.hpp"
#include "apps/desktop/png_writer.hpp"
#include "apps/desktop/shaped_text.hpp"
#include "core/audio/device.hpp"
#include "core/pipeline/live_view.hpp"

namespace fs = std::filesystem;
using namespace ee;
using namespace ee::desktop;

namespace {

#if defined(_WIN32)
#define EE_GL_API __stdcall
#else
#define EE_GL_API
#endif

/// The few OpenGL 1.x calls the app makes itself; ImGui's backend loads its own.
struct Gl {
    void(EE_GL_API* viewport)(int, int, int, int) = nullptr;
    void(EE_GL_API* clear_color)(float, float, float, float) = nullptr;
    void(EE_GL_API* clear)(unsigned) = nullptr;
    void(EE_GL_API* read_pixels)(int, int, int, int, unsigned, unsigned, void*) = nullptr;
    static constexpr unsigned kColorBufferBit = 0x4000, kRgba = 0x1908, kUnsignedByte = 0x1401;

    void load() {
        viewport = reinterpret_cast<decltype(viewport)>(glfwGetProcAddress("glViewport"));
        clear_color = reinterpret_cast<decltype(clear_color)>(glfwGetProcAddress("glClearColor"));
        clear = reinterpret_cast<decltype(clear)>(glfwGetProcAddress("glClear"));
        read_pixels = reinterpret_cast<decltype(read_pixels)>(glfwGetProcAddress("glReadPixels"));
        if (!viewport || !clear_color || !clear || !read_pixels) throw std::runtime_error("OpenGL 1.x entry points missing");
    }
};

struct Options {
    DesktopSession::Settings settings;
    fs::path font = EE_DESKTOP_FONT;
    fs::path screenshot;
    double after_s = -1.0;
};

Options parse(int argc, char** argv) {
    Options o;
    o.settings.config = fs::exists("config/pipeline.engines.yaml") ? "config/pipeline.engines.yaml" : "config/pipeline.yaml";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--config") o.settings.config = value();
        else if (a == "--input") o.settings.input_wav = value();
        else if (a == "--mute") o.settings.mute = true;
        else if (a == "--capture-device") o.settings.capture_device = value();
        else if (a == "--playback-device") o.settings.playback_device = value();
        else if (a == "--font") o.font = value();
        else if (a == "--screenshot") o.screenshot = value();
        else if (a == "--after") o.after_s = std::stod(value());
        else if (a == "--set") {
            const std::string kv = value();
            const auto eq = kv.find('=');
            if (eq == std::string::npos) throw std::runtime_error("--set expects KEY=VALUE");
            o.settings.overrides.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        } else {
            throw std::runtime_error("unknown argument '" + a + "'");
        }
    }
    return o;
}

ImU32 emotion_color(const std::string& label) {
    if (label == "anger") return IM_COL32(205, 64, 56, 255);
    if (label == "joy") return IM_COL32(214, 150, 30, 255);
    if (label == "sadness") return IM_COL32(64, 112, 200, 255);
    if (label == "fear") return IM_COL32(140, 84, 190, 255);
    if (label == "surprise") return IM_COL32(32, 156, 156, 255);
    if (label == "calm") return IM_COL32(80, 156, 100, 255);
    return IM_COL32(110, 110, 118, 255);  // neutral
}

void chip(const std::string& label, ImU32 color) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const ImVec2 text = ImGui::CalcTextSize(label.c_str());
    const ImVec2 size(text.x + 16.0f, text.y + 4.0f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + size.x, p.y + size.y), color, size.y * 0.5f);
    dl->AddText(ImVec2(p.x + 8.0f, p.y + 2.0f), IM_COL32_WHITE, label.c_str());
    ImGui::Dummy(size);
}

/// p95 against the budget: within it, up to 1.5x over, or beyond.
ImVec4 budget_color(double ms, double budget) {
    if (ms <= budget) return ImVec4(0.42f, 0.78f, 0.47f, 1.0f);
    if (ms <= 1.5 * budget) return ImVec4(0.92f, 0.70f, 0.25f, 1.0f);
    return ImVec4(0.93f, 0.40f, 0.36f, 1.0f);
}

struct Ui {
    Ui(DesktopSession& s, LiveView& v, ShapedText& h, Options& o) : session(s), view(v), hindi(h), options(o) {
        try {
            for (const AudioDeviceInfo& d : list_audio_devices()) (d.capture ? captures : playbacks).push_back(d);
        } catch (const std::exception& e) {  // no audio here: files still play, muted
            device_error = e.what();
        }
        std::snprintf(wav, sizeof wav, "%s", options.settings.input_wav.string().c_str());
    }

    DesktopSession& session;
    LiveView& view;
    ShapedText& hindi;
    Options& options;
    std::vector<AudioDeviceInfo> captures, playbacks;
    int capture = -1, playback = -1;  ///< -1: the system default
    std::string device_error;
    char wav[512] = {};
    std::uint64_t seen_version = 0;

    void device_combo(const char* label, const std::vector<AudioDeviceInfo>& devices, int& selected) {
        ImGui::SetNextItemWidth(220.0f);
        const char* preview = selected < 0 ? "System default" : devices[static_cast<std::size_t>(selected)].name.c_str();
        if (ImGui::BeginCombo(label, preview)) {
            if (ImGui::Selectable("System default", selected < 0)) selected = -1;
            for (std::size_t i = 0; i < devices.size(); ++i) {
                if (ImGui::Selectable(devices[i].name.c_str(), selected == static_cast<int>(i))) selected = static_cast<int>(i);
            }
            ImGui::EndCombo();
        }
    }

    void start(bool from_file) {
        DesktopSession::Settings s = options.settings;
        s.capture_device = capture < 0 ? "" : captures[static_cast<std::size_t>(capture)].name;
        s.playback_device = playback < 0 ? "" : playbacks[static_cast<std::size_t>(playback)].name;
        s.input_wav = from_file ? fs::path(wav) : fs::path();
        session.start(std::move(s));
    }

    void controls() {
        ImGui::TextUnformatted("EmotionEdge");
        ImGui::SameLine();
        ImGui::TextDisabled("live speech translation   %s", options.settings.config.filename().string().c_str());
        const bool running = session.running();
        ImGui::BeginDisabled(running);
        device_combo("Microphone", captures, capture);
        ImGui::SameLine();
        device_combo("Speaker", playbacks, playback);
        ImGui::SameLine();
        if (ImGui::Button("Listen")) start(false);
        if (!device_error.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%s)", device_error.c_str());
        }
        ImGui::SetNextItemWidth(360.0f);
        ImGui::InputTextWithHint("##wav", "speech.wav", wav, sizeof wav);
        ImGui::SameLine();
        if (ImGui::Button("Play file") && wav[0] != '\0') start(true);
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::BeginDisabled(!running);
        if (ImGui::Button("Stop")) session.stop();
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("%s", session.status().c_str());
    }

    void captions() {
        const float width = ImGui::GetContentRegionAvail().x - 8.0f;
        const ImVec4 dim = ImGui::GetStyle().Colors[ImGuiCol_TextDisabled];
        const auto utterances = view.snapshot();
        if (utterances.empty()) ImGui::TextDisabled("Captions appear here: press Listen, or play a WAV file.");
        for (const LiveView::Utterance& u : utterances) {
            ImGui::PushID(static_cast<int>(u.id));
            ImGui::PushStyleColor(ImGuiCol_Text, u.source_final ? ImGui::GetStyle().Colors[ImGuiCol_Text] : dim);
            ImGui::TextWrapped("%s", u.source.empty() ? "..." : u.source.c_str());
            ImGui::PopStyleColor();
            if (!u.emotion.empty()) {
                chip(u.emotion, emotion_color(u.emotion));
                ImGui::SameLine();
                ImGui::TextDisabled("V %+.2f   A %+.2f   D %+.2f   confidence %.2f", static_cast<double>(u.valence),
                                    static_cast<double>(u.arousal), static_cast<double>(u.dominance),
                                    static_cast<double>(u.confidence));
            }
            if (!u.translation.empty()) {
                hindi.draw(u.translation, width, u.translation_final ? IM_COL32(245, 245, 245, 255) : IM_COL32(150, 150, 158, 255));
            }
            if (!u.ecs.empty()) {
                float mean = 0.0f;
                for (const float e : u.ecs) mean += e;
                mean /= static_cast<float>(u.ecs.size());
                std::string heard;
                for (const std::string& h : u.heard) heard += (heard.empty() ? "" : ", ") + h;
                ImGui::TextDisabled("ECS %.2f   the translation sounded: %s", static_cast<double>(mean), heard.c_str());
            }
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::PopID();
        }
        // Follow new captions while scrolled to the bottom.
        const std::uint64_t version = view.version();
        if (version != seen_version && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f) ImGui::SetScrollHereY(1.0f);
        seen_version = version;
    }

    void latency() {
        const DesktopSession::Stats st = session.stats();
        ImGui::SeparatorText("Latency after the speaker stops (ms)");
        if (ImGui::BeginTable("latency", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthStretch, 3.0f);
            ImGui::TableSetupColumn("p50");
            ImGui::TableSetupColumn("p95");
            ImGui::TableSetupColumn("budget");
            ImGui::TableHeadersRow();
            const auto row = [](const DesktopSession::LatencyRow& r, bool bold) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (bold) ImGui::TextUnformatted(r.name.c_str());
                else ImGui::TextDisabled("%s", r.name.c_str());
                ImGui::TableNextColumn();
                if (r.count == 0) {
                    ImGui::TextDisabled("-");
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("-");
                } else {
                    ImGui::Text("%.0f", r.p50_ms);
                    ImGui::TableNextColumn();
                    ImGui::TextColored(budget_color(r.p95_ms, r.budget_ms), "%.0f", r.p95_ms);
                }
                ImGui::TableNextColumn();
                ImGui::TextDisabled("%.0f", r.budget_ms);
            };
            for (const auto& r : st.rows) row(r, false);
            row(st.end_to_end, true);
            ImGui::EndTable();
        }
        ImGui::Spacing();
        ImGui::Text("Translation queued to play: %.1f s", st.playout_queued_s);
        ImGui::Text("Audio dropouts: %llu samples", static_cast<unsigned long long>(st.dropout_samples));
        ImGui::Text("Utterances: %zu", view.snapshot().size());
        ImGui::SeparatorText("Emotion");
        const std::vector<std::string> labels = {"anger", "joy", "sadness", "fear", "surprise", "calm", "neutral"};
        const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        for (std::size_t i = 0; i < labels.size(); ++i) {
            chip(labels[i], emotion_color(labels[i]));
            if (i + 1 < labels.size()) {  // wrap when the next chip would not fit
                const float next = ImGui::CalcTextSize(labels[i + 1].c_str()).x + 16.0f;
                if (ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + next < right) ImGui::SameLine();
            }
        }
    }

    void frame() {
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("EmotionEdge", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
        controls();
        ImGui::Separator();
        const float panel = std::max(320.0f, ImGui::GetContentRegionAvail().x * 0.32f);
        ImGui::BeginChild("captions", ImVec2(ImGui::GetContentRegionAvail().x - panel - 8.0f, 0.0f));
        captions();
        ImGui::EndChild();
        ImGui::SameLine();
        ImGui::BeginChild("panel", ImVec2(0.0f, 0.0f));
        latency();
        ImGui::EndChild();
        ImGui::End();
    }
};

int run(Options options) {
    glfwSetErrorCallback([](int code, const char* text) { std::fprintf(stderr, "glfw error %d: %s\n", code, text); });
    if (glfwInit() == GLFW_FALSE) throw std::runtime_error("cannot initialize GLFW (no display?)");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    const bool shooting = !options.screenshot.empty();
    if (shooting) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(1360, 820, "EmotionEdge", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        throw std::runtime_error("cannot open a window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    Gl gl;
    gl.load();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 6.0f;
    style.WindowPadding = ImVec2(14.0f, 12.0f);
    style.ItemSpacing = ImVec2(10.0f, 7.0f);
    if (io.Fonts->AddFontFromFileTTF(options.font.string().c_str(), 18.0f) == nullptr) io.Fonts->AddFontDefault();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    int exit_code = 0;
    // The atlas stays registered until the renderer backend has shut down (it owns the GPU copy).
    ShapedText hindi(options.font, 30.0f);
    {
        LiveView view;
        DesktopSession session(view);  // ends before the view it reports to
        Ui ui(session, view, hindi, options);
        if (!options.settings.input_wav.empty()) ui.start(true);

        const auto t0 = std::chrono::steady_clock::now();
        double finished_at = -1.0;
        while (glfwWindowShouldClose(window) == GLFW_FALSE) {
            glfwWaitEventsTimeout(1.0 / 30.0);
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ui.frame();
            ImGui::Render();
            int w = 0, h = 0;
            glfwGetFramebufferSize(window, &w, &h);
            gl.viewport(0, 0, w, h);
            gl.clear_color(0.08f, 0.08f, 0.09f, 1.0f);
            gl.clear(Gl::kColorBufferBit);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            if (shooting) {
                const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (finished_at < 0.0 && !session.running()) finished_at = t;
                const bool due = options.after_s > 0.0 ? t >= options.after_s : finished_at >= 0.0 && t >= finished_at + 1.0;
                if (due || t > 600.0) {
                    std::vector<unsigned char> pixels(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
                    gl.read_pixels(0, 0, w, h, Gl::kRgba, Gl::kUnsignedByte, pixels.data());
                    std::vector<unsigned char> flipped(pixels.size());
                    const std::size_t stride = static_cast<std::size_t>(w) * 4;
                    for (int y = 0; y < h; ++y) {
                        std::memcpy(flipped.data() + static_cast<std::size_t>(y) * stride,
                                    pixels.data() + static_cast<std::size_t>(h - 1 - y) * stride, stride);
                    }
                    if (!write_png(options.screenshot, flipped, w, h)) exit_code = 1;
                    std::printf("%s %s (%dx%d) at %.1f s: %s\n", exit_code == 0 ? "wrote" : "could not write",
                                options.screenshot.string().c_str(), w, h, t, session.status().c_str());
                    break;
                }
            }
            glfwSwapBuffers(window);
        }
        session.stop();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    hindi.release();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return exit_code;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
