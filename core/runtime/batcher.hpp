#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ee {

/// Requests from several threads (the streaming server's sessions) run in batches on one
/// executor. A request that arrives while a batch runs waits for it, then goes in the next batch
/// with every other request that arrived meanwhile: up to max_batch, of the same group. Nothing
/// waits when the executor is idle, so a lone caller pays only a thread hop.
///
/// For a GPU model, a batch of several sequences costs about what one does, where the same
/// requests run side by side would each take several times as long.
template <typename Request, typename Result>
class Batcher {
public:
    /// Runs one batch; returns one result per request, in order. May throw: every request of the
    /// batch then gets the exception.
    using Run = std::function<std::vector<Result>(std::vector<Request>& batch)>;
    /// Requests of different groups (e.g. beam sizes) never share a batch.
    using Group = std::function<int(const Request&)>;

    Batcher(Run run, std::size_t max_batch, Group group = {})
        : run_(std::move(run)), group_(std::move(group)), max_batch_(max_batch == 0 ? 1 : max_batch),
          worker_([this] { loop(); }) {}
    ~Batcher() {
        {
            const std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }
    Batcher(const Batcher&) = delete;
    Batcher& operator=(const Batcher&) = delete;

    /// Blocks until the request's batch has run; rethrows its exception.
    Result submit(Request request) {
        Job job{std::move(request), {}};
        std::future<Result> done = job.promise.get_future();
        {
            const std::lock_guard<std::mutex> lock(mu_);
            if (stop_) throw std::runtime_error("batcher stopped");
            pending_.push_back(&job);
        }
        cv_.notify_one();
        return done.get();
    }

    /// Requests waiting for the running batch to finish.
    [[nodiscard]] std::size_t waiting() const {
        const std::lock_guard<std::mutex> lock(mu_);
        return pending_.size();
    }
    /// Batches run so far, and the requests they held (requests / batches = mean batch size).
    [[nodiscard]] std::size_t batches() const {
        const std::lock_guard<std::mutex> lock(mu_);
        return batches_;
    }
    [[nodiscard]] std::size_t requests() const {
        const std::lock_guard<std::mutex> lock(mu_);
        return requests_;
    }

private:
    struct Job {
        Request request;
        std::promise<Result> promise;
    };

    void loop() {
        std::vector<Job*> jobs;
        std::vector<Request> batch;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return stop_ || !pending_.empty(); });
                if (pending_.empty()) return;  // stopping, nothing left
                // Everything that is waiting, up to max_batch, of the first request's group.
                const int group = group_ ? group_(pending_.front()->request) : 0;
                for (auto it = pending_.begin(); it != pending_.end() && jobs.size() < max_batch_;) {
                    if (!group_ || group_((*it)->request) == group) {
                        jobs.push_back(*it);
                        it = pending_.erase(it);
                    } else {
                        ++it;
                    }
                }
                ++batches_;
                requests_ += jobs.size();
            }
            batch.clear();
            for (Job* j : jobs) batch.push_back(std::move(j->request));
            try {
                std::vector<Result> results = run_(batch);
                if (results.size() != jobs.size()) throw std::logic_error("batcher: one result per request expected");
                for (std::size_t i = 0; i < jobs.size(); ++i) jobs[i]->promise.set_value(std::move(results[i]));
            } catch (...) {
                for (Job* j : jobs) j->promise.set_exception(std::current_exception());
            }
            jobs.clear();
        }
    }

    Run run_;
    Group group_;
    std::size_t max_batch_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job*> pending_;
    bool stop_ = false;
    std::size_t batches_ = 0;
    std::size_t requests_ = 0;
    std::thread worker_;  ///< last: started once everything above exists
};

}  // namespace ee
