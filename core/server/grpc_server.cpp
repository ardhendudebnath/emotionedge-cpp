#include "core/server/grpc_server.hpp"

#include <grpcpp/grpcpp.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/runtime/log.hpp"
#include "emotionedge/v1/translator.grpc.pb.h"

namespace ee {

namespace {

using json = nlohmann::json;
namespace pb = emotionedge::v1;
using Stream = grpc::ServerReaderWriter<pb::TranslateResponse, pb::TranslateRequest>;

void fill_point(pb::EmotionPoint* point, const json& e) {
    point->set_label(e.value("label", ""));
    point->set_valence(e.value("valence", 0.0f));
    point->set_arousal(e.value("arousal", 0.0f));
    point->set_dominance(e.value("dominance", 0.0f));
}

void add_words(const json& e, const char* key, google::protobuf::RepeatedPtrField<std::string>* out) {
    for (const std::string& w : e.value(key, std::vector<std::string>{})) out->Add(std::string(w));
}

/// Writes a stream's results as typed responses. The stream's pump thread is the only writer;
/// the call's handler thread only reads, and gRPC allows one of each at a time.
class GrpcChannel final : public OutputChannel {
public:
    explicit GrpcChannel(Stream& stream) : stream_(stream) {}

    void ready(const Ready& r) override {
        pb::TranslateResponse m;
        pb::Ready* x = m.mutable_ready();
        x->set_input_rate(r.input_rate);
        x->set_output_rate(r.output_rate);
        x->set_source_language(r.source_language);
        x->set_target_language(r.target_language);
        write(m);
    }

    /// The recorder's JSON event, as its typed message.
    void event(std::string_view text) override {
        const json e = json::parse(text, nullptr, false);
        if (!e.is_object()) return;
        const std::string type = e.value("type", "");
        const auto utterance = e.value("utterance", std::uint64_t{0});
        pb::TranslateResponse m;
        if (type == "transcript") {
            pb::Transcript* x = m.mutable_transcript();
            x->set_utterance(utterance);
            x->set_is_final(e.value("final", false));
            x->set_text(e.value("text", ""));
            x->set_language(e.value("language", ""));
            x->set_start_seconds(e.value("start", 0.0));
            x->set_end_seconds(e.value("end", 0.0));
            x->set_stable_words(e.value("stable_words", 0u));
        } else if (type == "emotion") {
            pb::Emotion* x = m.mutable_emotion();
            x->set_utterance(utterance);
            fill_point(x->mutable_point(), e);
            x->set_confidence(e.value("confidence", 0.0f));
            add_words(e, "emphasis", x->mutable_emphasis());
        } else if (type == "translation") {
            pb::Translation* x = m.mutable_translation();
            x->set_utterance(utterance);
            x->set_is_final(e.value("final", false));
            x->set_text(e.value("text", ""));
            x->set_language(e.value("language", ""));
            add_words(e, "emphasis", x->mutable_emphasis());
        } else if (type == "prosody") {
            pb::Prosody* x = m.mutable_prosody();
            x->set_utterance(utterance);
            x->set_pitch_pct(e.value("pitch_pct", 0.0f));
            x->set_range_pct(e.value("range_pct", 0.0f));
            x->set_rate_pct(e.value("rate_pct", 0.0f));
            x->set_energy_db(e.value("energy_db", 0.0f));
            x->set_pause_ms(e.value("pause_ms", 0.0f));
            x->set_accent(e.value("accent", 0.0f));
            x->set_final_fall(e.value("final_fall", 0.0f));
        } else if (type == "consistency") {
            pb::Consistency* x = m.mutable_consistency();
            x->set_utterance(utterance);
            x->set_ecs(e.value("ecs", 0.0f));
            if (e.contains("heard")) fill_point(x->mutable_heard(), e["heard"]);
        } else if (type == "playout") {
            pb::Playout* x = m.mutable_playout();
            x->set_utterance(utterance);
            x->set_start_seconds(e.value("start", 0.0));
            x->set_end_seconds(e.value("end", 0.0));
        } else {
            return;
        }
        write(m);
    }

    void audio(std::span<const std::int16_t> pcm) override {
        std::string bytes(pcm.size() * 2, '\0');
        for (std::size_t i = 0; i < pcm.size(); ++i) {
            const auto s = static_cast<std::uint16_t>(pcm[i]);
            bytes[2 * i] = static_cast<char>(s & 0xFFu);
            bytes[2 * i + 1] = static_cast<char>(s >> 8);
        }
        pb::TranslateResponse m;
        m.set_audio(std::move(bytes));
        write(m);
    }

    void done(std::string_view session_json) override {
        pb::TranslateResponse m;
        m.mutable_done()->set_session_json(std::string(session_json));
        write(m);
    }

    void error(std::string_view message) override {
        pb::TranslateResponse m;
        m.mutable_error()->set_message(std::string(message));
        write(m);
    }

private:
    void write(const pb::TranslateResponse& m) { stream_.Write(m); }  // false once the client is gone

    Stream& stream_;
};

grpc::Status refused_status(const SessionPool::Refused& refused) {
    switch (refused.reason()) {
    case SessionPool::Refused::Reason::Busy: return {grpc::StatusCode::RESOURCE_EXHAUSTED, refused.what()};
    case SessionPool::Refused::Reason::BadRate: return {grpc::StatusCode::INVALID_ARGUMENT, refused.what()};
    case SessionPool::Refused::Reason::Failed: break;
    }
    return {grpc::StatusCode::INTERNAL, refused.what()};
}

class TranslatorService final : public pb::Translator::Service {
public:
    explicit TranslatorService(SessionPool& pool) : pool_(pool) {}

    grpc::Status Translate(grpc::ServerContext* context, Stream* stream) override {
        pb::TranslateRequest request;
        bool pending = stream->Read(&request);
        int rate = 16000;
        if (pending && request.request_case() == pb::TranslateRequest::kConfig) {
            if (request.config().sample_rate() != 0) rate = request.config().sample_rate();
            pending = false;  // consumed
        }
        GrpcChannel channel(*stream);
        std::shared_ptr<SessionStream> session;
        try {
            session = pool_.open(channel, rate);
        } catch (const SessionPool::Refused& refused) {
            return refused_status(refused);
        }
        log::info("grpc: session opened (", rate, " Hz in) for ", context->peer());

        bool ended = false;
        const auto handle = [&](const pb::TranslateRequest& r) {
            if (r.request_case() == pb::TranslateRequest::kAudio) {
                session->audio(r.audio());
            } else if (r.request_case() == pb::TranslateRequest::kEnd) {
                session->end();
                ended = true;
            }
        };
        if (pending) handle(request);
        while (!ended && stream->Read(&request)) handle(request);
        if (context->IsCancelled()) {
            pool_.close(session);
            log::info("grpc: session cancelled by the client");
            return grpc::Status::CANCELLED;
        }
        if (!ended) session->end();  // the client closed its side without EndOfSpeech
        session->wait();             // Done or Error went out (or the pool stopped)
        pool_.close(session);
        log::info("grpc: session closed");
        return grpc::Status::OK;
    }

private:
    SessionPool& pool_;
};

}  // namespace

struct GrpcServer::Impl {
    Impl(SessionPool& p, std::string h, int pt) : service(p), host(std::move(h)), port(pt) {}

    TranslatorService service;
    std::string host;
    int port;
    int bound = 0;
    std::unique_ptr<grpc::Server> server;
};

GrpcServer::GrpcServer(SessionPool& pool, std::string host, int port)
    : impl_(std::make_unique<Impl>(pool, std::move(host), port)) {}

GrpcServer::~GrpcServer() { stop(); }

void GrpcServer::start() {
    grpc::ServerBuilder builder;
    const std::string address = impl_->host + ":" + std::to_string(impl_->port);
    builder.AddListeningPort(address, grpc::InsecureServerCredentials(), &impl_->bound);
    builder.RegisterService(&impl_->service);
    impl_->server = builder.BuildAndStart();
    if (!impl_->server || impl_->bound == 0) {
        impl_->server.reset();
        throw std::runtime_error("cannot listen on " + address + " (gRPC)");
    }
}

void GrpcServer::stop() {
    if (!impl_->server) return;
    // Calls still running after the deadline are cancelled.
    impl_->server->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
    impl_->server->Wait();
    impl_->server.reset();
}

int GrpcServer::port() const noexcept { return impl_->bound; }

}  // namespace ee
