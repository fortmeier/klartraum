// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Interactive chat with Qwen3.6-27B (or another qwen35 GGUF model) on
// Klartraum's Vulkan GGUF runtime.
//
//   scripts/qwen_gguf/download.py fetches the model; then, from the repo root:
//   ./build/examples/qwen_chat_example [--model FILE.gguf] [options]
//
// The conversation stays in the network's caches: each turn only runs the
// new tokens. Commands: /reset (new conversation), /think on|off,
// /stats, /quit. See examples/README.md for the options.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gguf/gguf_network.hpp"
#include "klartraum/gguf/gguf_tokenizer.hpp"
#include "klartraum/headless_frontend.hpp"

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string model;
    std::string system;
    std::string prompt; // non-interactive: answer this prompt and exit
    uint32_t context = 4096;
    uint32_t chunk = 16;
    uint32_t maxReply = 2048;
    bool think = false;
    bool greedy = false;
    float temperature = -1.0f; // < 0: the model's recommendation for the thinking mode
    float topP = -1.0f;
    uint32_t topK = 20;
    uint32_t seed = 0;
    bool quiet = false;
    bool profile = false;
};

void usage() {
    std::cout << "Usage: qwen_chat_example [options]\n"
                 "  --model FILE        GGUF model (default: data/gguf/qwen3.6-27b/*.gguf)\n"
                 "  --system TEXT       system prompt\n"
                 "  --prompt TEXT       answer one prompt and exit\n"
                 "  --think             start with thinking enabled (/think on|off switches)\n"
                 "  --context N         context length in tokens (default 4096)\n"
                 "  --chunk N           prompt tokens per submission, 1..16 (default 16)\n"
                 "  --max-reply N       most tokens per reply (default 2048)\n"
                 "  --temperature T     sampling temperature (default 0.7, 1.0 with thinking)\n"
                 "  --top-p P           nucleus sampling (default 0.8, 0.95 with thinking)\n"
                 "  --top-k K           top-k sampling (default 20)\n"
                 "  --greedy            always take the most likely token\n"
                 "  --seed N            random seed (default: time)\n"
                 "  --quiet             no loading and speed messages\n"
                 "  --profile           print the mean GPU time per layer kind after each reply\n";
}

Options parseOptions(int argc, char** argv) {
    Options options;
    options.seed = uint32_t(std::chrono::steady_clock::now().time_since_epoch().count());
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--model")
            options.model = value();
        else if (arg == "--system")
            options.system = value();
        else if (arg == "--prompt")
            options.prompt = value();
        else if (arg == "--think")
            options.think = true;
        else if (arg == "--context")
            options.context = uint32_t(std::stoul(value()));
        else if (arg == "--chunk")
            options.chunk = uint32_t(std::stoul(value()));
        else if (arg == "--max-reply")
            options.maxReply = uint32_t(std::stoul(value()));
        else if (arg == "--temperature")
            options.temperature = std::stof(value());
        else if (arg == "--top-p")
            options.topP = std::stof(value());
        else if (arg == "--top-k")
            options.topK = uint32_t(std::stoul(value()));
        else if (arg == "--greedy")
            options.greedy = true;
        else if (arg == "--seed")
            options.seed = uint32_t(std::stoul(value()));
        else if (arg == "--quiet")
            options.quiet = true;
        else if (arg == "--profile")
            options.profile = true;
        else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option " + arg);
        }
    }
    if (options.model.empty()) {
        const fs::path directory = "data/gguf/qwen3.6-27b";
        if (fs::exists(directory)) {
            for (const auto& entry : fs::directory_iterator(directory)) {
                if (entry.path().extension() == ".gguf")
                    options.model = entry.path().string();
            }
        }
        if (options.model.empty()) {
            throw std::runtime_error("No model found; run scripts/qwen_gguf/download.py or pass --model");
        }
    }
    return options;
}

double secondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Picks the next token from logits with temperature, top-k and top-p.
class Sampler {
public:
    Sampler(uint32_t seed)
        : rng(seed) {}

    int32_t sample(const std::vector<float>& logits, bool greedy, float temperature, uint32_t topK, float topP) {
        if (greedy || temperature <= 0.0f) {
            return int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        const uint32_t k = std::min<uint32_t>(std::max<uint32_t>(topK, 1), uint32_t(logits.size()));
        candidates.resize(logits.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        candidates.resize(k);
        std::vector<double> weights(k);
        double total = 0.0;
        for (uint32_t i = 0; i < k; ++i) {
            weights[i] = std::exp(double(logits[candidates[i]] - logits[candidates[0]]) / temperature);
            total += weights[i];
        }
        // Keep the smallest prefix whose probability reaches topP.
        double kept = 0.0;
        uint32_t count = 0;
        while (count < k && kept < topP * total)
            kept += weights[count++];
        std::discrete_distribution<uint32_t> choose(weights.begin(), weights.begin() + count);
        return candidates[choose(rng)];
    }

private:
    std::mt19937 rng;
    std::vector<int32_t> candidates;
};

// Prints token bytes as they arrive, holding back incomplete UTF-8 sequences.
class Printer {
public:
    void write(const std::string& bytes) {
        pending += bytes;
        size_t complete = pending.size();
        // Find the start of a trailing, incomplete multi-byte sequence.
        for (size_t back = 1; back <= std::min<size_t>(3, pending.size()); ++back) {
            const unsigned char byte = pending[pending.size() - back];
            if ((byte & 0xc0) == 0x80)
                continue; // continuation byte
            const size_t length = byte < 0x80 ? 1 : (byte >> 5) == 0x6 ? 2 : (byte >> 4) == 0xe ? 3 : 4;
            if (length > back)
                complete = pending.size() - back;
            break;
        }
        std::cout << pending.substr(0, complete) << std::flush;
        pending.erase(0, complete);
    }
    void flush() {
        std::cout << pending << std::flush;
        pending.clear();
    }

private:
    std::string pending;
};

class Chat {
public:
    Chat(klartraum::VulkanContext& context, const Options& options)
        : context(context),
          options(options),
          sampler(options.seed) {
        const auto started = std::chrono::steady_clock::now();
        klartraum::GgufNetworkOptions networkOptions;
        networkOptions.maxTokens = options.chunk;
        networkOptions.contextLength = options.context;
        network = context.create<klartraum::GgufNetwork>(options.model, networkOptions);
        tokenizer = std::make_unique<klartraum::GgufTokenizer>(network->getFile());
        graph = std::make_unique<klartraum::ComputeGraph>(context, 1);
        if (options.profile)
            graph->enableProfiling();
        graph->compileFrom(network);
        imStart = tokenizer->tokenId("<|im_start|>");
        imEnd = tokenizer->tokenId("<|im_end|>");
        endOfText = tokenizer->tokenId("<|endoftext|>");
        thinkStart = tokenizer->tokenId("<think>");
        thinkEnd = tokenizer->tokenId("</think>");
        if (!options.quiet) {
            std::cout << "Loaded " << fs::path(options.model).filename().string() << " ("
                      << (network->getWeightBytes() >> 20) << " MiB on the GPU) in " << secondsSince(started) << " s\n";
        }
        think = options.think;
        reset();
    }

    void reset() {
        network->resetState();
        position = 0;
        pending.clear();
        if (!options.system.empty()) {
            appendSpecial(imStart);
            appendText("system\n" + options.system);
            appendSpecial(imEnd);
            appendText("\n");
        }
    }

    void setThink(bool enabled) { think = enabled; }
    bool thinking() const { return think; }

    void printStats() const {
        std::cout << "context " << position << " / " << options.context << " tokens; last turn: prompt " << promptTokens
                  << " tokens at " << promptRate << " tok/s, reply " << replyTokens << " tokens at " << replyRate
                  << " tok/s\n";
    }

    // Adds a user turn and streams the reply.
    void respond(const std::string& message) {
        appendSpecial(imStart);
        appendText("user\n" + message);
        appendSpecial(imEnd);
        appendText("\n");
        appendSpecial(imStart);
        appendText("assistant\n");
        appendSpecial(thinkStart);
        appendText(think ? "\n" : "\n\n");
        if (!think) {
            appendSpecial(thinkEnd);
            appendText("\n\n");
        }

        const float temperature = options.temperature >= 0.0f ? options.temperature : (think ? 1.0f : 0.7f);
        const float topP = options.topP >= 0.0f ? options.topP : (think ? 0.95f : 0.8f);
        Printer printer;
        bool inThinking = think;
        if (inThinking)
            std::cout << "\033[2m";

        auto started = std::chrono::steady_clock::now();
        promptTokens = uint32_t(pending.size());
        std::vector<float> logits = run(); // prompt
        promptRate = promptTokens / secondsSince(started);
        started = std::chrono::steady_clock::now();
        replyTokens = 0;
        while (true) {
            const int32_t token = sampler.sample(logits, options.greedy, temperature, options.topK, topP);
            ++replyTokens;
            if (token == imEnd || token == endOfText) {
                pending.push_back(imEnd); // closes the turn in the context
                appendText("\n");
                break;
            }
            if (token == thinkEnd && inThinking) {
                printer.flush();
                std::cout << "\033[0m" << std::flush;
                inThinking = false;
            } else if (token != thinkStart) {
                printer.write(tokenizer->tokenBytes(token));
            }
            pending.push_back(token);
            if (replyTokens >= options.maxReply || position + pending.size() >= options.context) {
                printer.flush();
                std::cout << (inThinking ? "\033[0m" : "") << "\n[reply stopped: "
                          << (replyTokens >= options.maxReply ? "--max-reply reached" : "context full, /reset") << "]";
                pending.push_back(imEnd);
                appendText("\n");
                break;
            }
            logits = run();
        }
        printer.flush();
        if (inThinking)
            std::cout << "\033[0m";
        std::cout << std::endl;
        replyRate = replyTokens / secondsSince(started);
        if (!options.quiet) {
            std::cout << "\033[2m[" << promptTokens << " prompt tokens at " << int(promptRate) << " tok/s, "
                      << replyTokens << " reply tokens at " << std::round(replyRate * 10.0) / 10.0 << " tok/s]\033[0m"
                      << std::endl;
        }
        if (options.profile)
            printProfile();
    }

    // Mean GPU milliseconds per submission and layer kind (layer numbers removed).
    void printProfile() const {
        std::map<std::string, double> kinds;
        double total = 0.0;
        for (const auto& [name, milliseconds] : graph->getProfilingResults()) {
            std::string kind = name;
            if (kind.rfind("blk.", 0) == 0)
                kind = kind.substr(kind.find('.', 4) + 1);
            kinds[kind] += milliseconds;
            total += milliseconds;
        }
        std::vector<std::pair<std::string, double>> sorted(kinds.begin(), kinds.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::cout << "GPU time per submission: " << total << " ms\n";
        for (const auto& [kind, milliseconds] : sorted) {
            if (milliseconds >= 0.01 * total)
                std::cout << "  " << kind << ": " << milliseconds << " ms\n";
        }
    }

private:
    klartraum::VulkanContext& context;
    const Options& options;
    std::shared_ptr<klartraum::GgufNetwork> network;
    std::unique_ptr<klartraum::GgufTokenizer> tokenizer;
    std::unique_ptr<klartraum::ComputeGraph> graph;
    Sampler sampler;
    int32_t imStart, imEnd, endOfText, thinkStart, thinkEnd;
    bool think = false;
    uint32_t position = 0;
    std::vector<int32_t> pending; // tokens not yet in the caches
    uint32_t promptTokens = 0, replyTokens = 0;
    double promptRate = 0.0, replyRate = 0.0;

    void appendSpecial(int32_t token) { pending.push_back(token); }
    void appendText(const std::string& text) {
        const auto ids = tokenizer->encode(text);
        pending.insert(pending.end(), ids.begin(), ids.end());
    }

    // Runs the pending tokens in chunks; returns the logits after the last one.
    std::vector<float> run() {
        if (position + pending.size() > options.context) {
            throw std::runtime_error("The conversation exceeds the context of " + std::to_string(options.context) +
                                     " tokens; use /reset");
        }
        for (size_t start = 0; start < pending.size(); start += options.chunk) {
            const size_t end = std::min(pending.size(), start + options.chunk);
            network->setTokens({pending.begin() + long(start), pending.begin() + long(end)}, position);
            graph->submitAndWait(context.getGraphicsQueue(), 0);
            position += uint32_t(end - start);
        }
        pending.clear();
        return network->readLogits();
    }
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (!options.quiet) {
            std::cout << "Qwen3.6 is a third-party model (Apache 2.0, by Alibaba Cloud's Qwen team); Klartraum\n"
                         "neither ships nor endorses it. You are responsible for how you use it and its output.\n";
        }
        klartraum::HeadlessFrontend frontend;
        auto& context = frontend.getKlartraumEngine().getVulkanContext();
        Chat chat(context, options);

        if (!options.prompt.empty()) {
            chat.respond(options.prompt);
            return 0;
        }
        std::cout << "Commands: /reset, /think on|off, /stats, /quit. Thinking is " << (chat.thinking() ? "on" : "off")
                  << ".\n";
        std::string line;
        while (true) {
            std::cout << "\n> " << std::flush;
            if (!std::getline(std::cin, line))
                break;
            if (line.empty())
                continue;
            if (line == "/quit" || line == "/exit")
                break;
            if (line == "/reset") {
                chat.reset();
                std::cout << "[new conversation]\n";
            } else if (line == "/think on" || line == "/think off") {
                chat.setThink(line == "/think on");
                std::cout << "[thinking " << (chat.thinking() ? "on" : "off") << "]\n";
            } else if (line == "/stats") {
                chat.printStats();
            } else {
                chat.respond(line);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
