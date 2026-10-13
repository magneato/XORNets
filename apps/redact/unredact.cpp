#include "unredact.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb/stb_image_write.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <print>
#include <sstream>
#include <unistd.h>

#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>

namespace xn::unredact {

auto detect_redacted_rectangles(const redact::ImageRGB& img, int min_area, int dark_thresh)
    -> std::vector<redact::Box> {
    const int w = img.width;
    const int h = img.height;
    if (w <= 0 || h <= 0) return {};

    // 1. Multi-scale pyramid: downsample 4x using box averaging
    const int sw = std::max(1, w / 4);
    const int sh = std::max(1, h / 4);
    std::vector<std::uint8_t> small(sw * sh, 0);

    for (int sy = 0; sy < sh; ++sy) {
        for (int sx = 0; sx < sw; ++sx) {
            int lum_sum = 0;
            for (int dy = 0; dy < 4; ++dy) {
                for (int dx = 0; dx < 4; ++dx) {
                    const int px = std::min(w - 1, sx * 4 + dx);
                    const int py = std::min(h - 1, sy * 4 + dy);
                    const int idx = (py * w + px) * 3;
                    lum_sum += (img.data[idx] * 299 + img.data[idx + 1] * 587 + img.data[idx + 2] * 114) / 1000;
                }
            }
            small[sy * sw + sx] = static_cast<std::uint8_t>(lum_sum / 16);
        }
    }

    // 2. Binary mask of dark pixels (scaled back up)
    std::vector<std::vector<std::uint8_t>> binary(h, std::vector<std::uint8_t>(w, 0));
    for (int y = 0; y < h; ++y) {
        const int sy = std::min(sh - 1, y / 4);
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(sw - 1, x / 4);
            if (small[sy * sw + sx] < dark_thresh) {
                binary[y][x] = 1;
            }
        }
    }

    // 3. Scanline in/out edge transitions
    struct Span { int y; int x0; int x1; };
    std::vector<Span> spans;
    const int min_span_w = 16;

    for (int y = 0; y < h; ++y) {
        int in_x = -1;
        for (int x = 0; x < w; ++x) {
            if (binary[y][x] == 1) {
                if (in_x < 0) in_x = x;
            } else {
                if (in_x >= 0) {
                    if (x - in_x >= min_span_w) spans.push_back(Span{y, in_x, x});
                    in_x = -1;
                }
            }
        }
        if (in_x >= 0 && w - in_x >= min_span_w) spans.push_back(Span{y, in_x, w});
    }

    // 4. Group matching consecutive spans into boxes
    struct BoxTracker { redact::Box b; int rows; };
    std::vector<BoxTracker> active;
    std::vector<redact::Box> completed;

    for (const auto& sp : spans) {
        bool matched = false;
        for (auto& trk : active) {
            if (trk.b.y1 == sp.y && std::abs(trk.b.x0 - sp.x0) <= 8 && std::abs(trk.b.x1 - sp.x1) <= 8) {
                trk.b.x0 = std::min(trk.b.x0, sp.x0);
                trk.b.y1 = sp.y + 1;
                trk.b.x1 = std::max(trk.b.x1, sp.x1);
                trk.rows++;
                matched = true;
                break;
            }
        }
        if (!matched) {
            active.push_back(BoxTracker{redact::Box{sp.x0, sp.y, sp.x1, sp.y + 1}, 1});
        }
    }

    for (const auto& trk : active) {
        if (trk.rows >= 8) completed.push_back(trk.b);
    }

    // 5. Filter boxes by area and verify solid black fill in original image
    std::vector<redact::Box> filtered;
    for (const auto& b : completed) {
        const int area = b.area();
        if (area < min_area || area > (w * h * 0.8f)) continue;

        // Check dark pixel density
        int dark_count = 0;
        int total = 0;
        for (int y = b.y0; y < b.y1; ++y) {
            for (int x = b.x0; x < b.x1; ++x) {
                const int idx = (y * w + x) * 3;
                const int lum = (img.data[idx] * 299 + img.data[idx + 1] * 587 + img.data[idx + 2] * 114) / 1000;
                if (lum < 50) ++dark_count;
                ++total;
            }
        }
        if (total > 0 && (static_cast<float>(dark_count) / total) >= 0.80f) {
            filtered.push_back(b);
        }
    }

    // Deduplicate / merge overlapping boxes
    std::vector<redact::Box> merged;
    for (const auto& b : filtered) {
        bool overlap = false;
        for (auto& mb : merged) {
            const int ix0 = std::max(b.x0, mb.x0);
            const int iy0 = std::max(b.y0, mb.y0);
            const int ix1 = std::min(b.x1, mb.x1);
            const int iy1 = std::min(b.y1, mb.y1);
            if (ix1 > ix0 && iy1 > iy0) {
                mb.x0 = std::min(mb.x0, b.x0);
                mb.y0 = std::min(mb.y0, b.y0);
                mb.x1 = std::max(mb.x1, b.x1);
                mb.y1 = std::max(mb.y1, b.y1);
                overlap = true;
                break;
            }
        }
        if (!overlap) merged.push_back(b);
    }

    std::sort(merged.begin(), merged.end(), [](const redact::Box& a, const redact::Box& b) {
        return a.y0 < b.y0;
    });

    return merged;
}

auto extract_surrounding_context(const redact::ImageRGB& img, const redact::Box& bbox, int margin)
    -> std::string {
    const int x0 = std::max(0, bbox.x0 - margin);
    const int y0 = std::max(0, bbox.y0 - margin);
    const int x1 = std::min(img.width, bbox.x1 + margin);
    const int y1 = std::min(img.height, bbox.y1 + margin);

    const auto crop = img.crop(redact::Box{x0, y0, x1, y1});
    const std::string tmp_file = "/tmp/unredact_ctx_" + std::to_string(getpid()) + ".png";
    crop.save_png(tmp_file);

    const std::string cmd = "tesseract \"" + tmp_file + "\" stdout 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    std::string text;
    if (pipe) {
        char buf[256];
        while (fgets(buf, sizeof(buf), pipe)) text += buf;
        pclose(pipe);
    }
    std::remove(tmp_file.c_str());

    // Clean whitespace
    std::string clean;
    for (char c : text) {
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
        if (!clean.empty() && clean.back() == ' ' && c == ' ') continue;
        clean += c;
    }
    return clean;
}

auto load_corpus_phrases(const std::filesystem::path& corpus_path) -> std::vector<std::string> {
    std::ifstream f(corpus_path);
    if (!f.is_open()) return {};

    std::vector<std::string> phrases;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
            line.pop_back();
        if (line.size() >= 3) {
            phrases.push_back(line);
        }
    }
    return phrases;
}

auto score_candidate(std::string_view cand, std::string_view ctx) -> float {
    if (cand.empty()) return -100.0f;
    float score = 0.0f;
    int letters = 0;
    for (char c : cand) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == ' ') ++letters;
    }
    score += (static_cast<float>(letters) / cand.size()) * 20.0f;

    // Word count bonus
    int words = 1;
    for (char c : cand) if (c == ' ') ++words;
    score += std::min(words * 2.0f, 10.0f);

    // Context coherence check
    if (!ctx.empty()) {
        std::string cand_lower;
        for (char c : cand) cand_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        std::string ctx_lower;
        for (char c : ctx) ctx_lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (ctx_lower.find(cand_lower) != std::string::npos) score += 30.0f;
    }

    return score;
}

auto query_local_mcp_or_server(std::string_view prompt, std::string_view ctx, int port)
    -> std::optional<std::string> {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return std::nullopt;

    timeval tv{1, 0}; // 1 sec timeout
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_in saddr{};
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &saddr.sin_addr);

    if (connect(sock, reinterpret_cast<sockaddr*>(&saddr), sizeof(saddr)) < 0) {
        close(sock);
        return std::nullopt;
    }

    // Build JSON HTTP request
    std::string body = "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"" +
                       std::string(prompt) + " Context: " + std::string(ctx) + "\"}]}";
    std::string req = "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + std::to_string(port) +
                      "\r\nContent-Type: application/json\r\nContent-Length: " +
                      std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;

    send(sock, req.data(), req.size(), 0);
    char buf[1024];
    std::string resp;
    ssize_t r = 0;
    while ((r = recv(sock, buf, sizeof(buf) - 1, 0)) > 0) {
        buf[r] = '\0';
        resp += buf;
    }
    close(sock);

    auto f = resp.find("\"content\":");
    if (f != std::string::npos) {
        auto start = resp.find('"', f + 10);
        auto end = resp.find('"', start + 1);
        if (start != std::string::npos && end != std::string::npos) {
            return resp.substr(start + 1, end - start - 1);
        }
    }
    return std::nullopt;
}

void restore_redacted_region(redact::ImageRGB& img, const redact::Box& bbox, std::string_view text) {
    // 1. Inpaint white background into the rectangle
    img.fill_rect(bbox, 255, 255, 255);

    // 2. Draw recovered text into the box
    const int mid_y = bbox.y0 + std::max(1, (bbox.height() - 8) / 2);
    img.draw_text(bbox.x0 + 4, mid_y, text, 20, 20, 30);
}

} // namespace xn::unredact

auto main(int argc, char** argv) -> int {
    if (argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") {
        std::println(
            "unredact — Visual recovery and inpainting tool (C++26)\n\n"
            "Usage: unredact -o <unredacted.png> <redacted.png> [options]\n\n"
            "Options:\n"
            "  -o, --output FILE      Destination path for restored image (required)\n"
            "  --corpus FILE          Path to local text corpus\n"
            "  --dataset-meta FILE    Metadata JSON from redact tool for verification\n"
            "  --local-port PORT      Local MCP/LLM server port (default: 8080)\n"
            "  --verbose              Print detailed execution stages\n");
        return 0;
    }

    std::filesystem::path input_path;
    std::filesystem::path output_path;
    std::filesystem::path corpus_path;
    std::filesystem::path meta_path;
    int port = 8080;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if ((a == "-o" || a == "--output") && i + 1 < argc) output_path = argv[++i];
        else if (a == "--corpus" && i + 1 < argc) corpus_path = argv[++i];
        else if (a == "--dataset-meta" && i + 1 < argc) meta_path = argv[++i];
        else if (a == "--local-port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else if (!a.starts_with("-")) input_path = argv[i];
    }

    if (input_path.empty() || output_path.empty()) {
        std::println(stderr, "unredact: input and -o output paths are required. See --help");
        return 1;
    }

    auto img_opt = xn::redact::ImageRGB::load(input_path);
    if (!img_opt) {
        std::println(stderr, "unredact: failed to load image '{}'", input_path.string());
        return 1;
    }
    auto img = *img_opt;

    std::println("◈ unredact: analyzing {} ({}x{})", input_path.string(), img.width, img.height);
    const auto boxes = xn::unredact::detect_redacted_rectangles(img);
    std::println("  ✓ detected {} redacted block(s)", boxes.size());

    // Auto-discover corpus if not explicitly given
    if (corpus_path.empty()) {
        std::array<std::filesystem::path, 3> candidates{
            "/home/magneato/dev/bit88/corpus/corpus.txt",
            "/home/magneato/dev/bit88/bit/corpus/alice.txt",
            "/home/magneato/dev/bit/corpus/alice.txt"
        };
        for (const auto& c : candidates) {
            if (std::filesystem::exists(c)) {
                corpus_path = c;
                break;
            }
        }
    }

    std::vector<std::string> corpus_phrases;
    if (!corpus_path.empty() && std::filesystem::exists(corpus_path)) {
        corpus_phrases = xn::unredact::load_corpus_phrases(corpus_path);
        std::println("  ✓ loaded local corpus from {} ({} phrases)", corpus_path.filename().string(), corpus_phrases.size());
    }

    // Process and inpaint each redacted box
    for (size_t idx = 0; idx < boxes.size(); ++idx) {
        const auto& b = boxes[idx];
        const int est_chars = std::max(2, b.width() / 8);
        const auto ctx = xn::unredact::extract_surrounding_context(img, b);

        std::string chosen = "[REDACTED]";

        // Deliberate with local MCP endpoint if active
        auto local_pred = xn::unredact::query_local_mcp_or_server(
            "Predict redacted phrase of " + std::to_string(est_chars) + " chars", ctx, port);
        if (local_pred) {
            chosen = *local_pred;
        } else if (!corpus_phrases.empty()) {
            float best_score = -1000.0f;
            for (const auto& phrase : corpus_phrases) {
                if (std::abs(static_cast<int>(phrase.size()) - est_chars) <= std::max(4, est_chars / 3)) {
                    float s = xn::unredact::score_candidate(phrase, ctx);
                    if (s > best_score) {
                        best_score = s;
                        chosen = phrase;
                    }
                }
            }
        }

        if (verbose) {
            std::println("    · block #{}: box [{}, {}, {}, {}] ctx=\"{}\"",
                         idx + 1, b.x0, b.y0, b.x1, b.y1, ctx);
        }
        std::println("  ✓ block #{}: restored -> \"{}\"", idx + 1, chosen);
        xn::unredact::restore_redacted_region(img, b, chosen);
    }

    if (img.save_png(output_path)) {
        std::println("◈ unredact: complete -> {}", output_path.string());
    } else {
        std::println(stderr, "unredact: failed to save to {}", output_path.string());
        return 1;
    }

    return 0;
}
