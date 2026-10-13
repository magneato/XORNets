#pragma once

#include "redact.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <optional>

namespace xn::unredact {

struct RedactedBlock {
    redact::Box bbox{};
    int est_chars = 0;
    std::string recovered_text;
};

// Block Detection via multi-scale pyramid + in/out edge scan
auto detect_redacted_rectangles(const redact::ImageRGB& img, int min_area = 200, int dark_thresh = 40)
    -> std::vector<redact::Box>;

// OCR Context
auto extract_surrounding_context(const redact::ImageRGB& img, const redact::Box& bbox, int margin = 40)
    -> std::string;

// Local Corpus
auto load_corpus_phrases(const std::filesystem::path& corpus_path) -> std::vector<std::string>;
auto score_candidate(std::string_view cand, std::string_view ctx) -> float;

// Local Deliberation (strictly local, port 8080)
auto query_local_mcp_or_server(std::string_view prompt, std::string_view ctx, int port = 8080)
    -> std::optional<std::string>;

// Reconstruction
void restore_redacted_region(redact::ImageRGB& img, const redact::Box& bbox, std::string_view text);

} // namespace xn::unredact
