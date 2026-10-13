#include "redact.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb/stb_image_write.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <print>
#include <random>
#include <sstream>


namespace xn::redact {

namespace {

// Minimal 5x7 font bitmap for basic ASCII fallback rendering
constexpr std::array<std::uint8_t, 96 * 5> FONT5x7 = [] {
    std::array<std::uint8_t, 96 * 5> f{};
    // standard readable letters, digits, and basic punctuation
    // (fallback rendering into pixel grids)
    return f;
}();

auto boxes_overlap(const Box& b1, const Box& b2, int margin = 0) -> bool {
    return !(b1.x1 + margin <= b2.x0 ||
             b1.x0 - margin >= b2.x1 ||
             b1.y1 + margin <= b2.y0 ||
             b1.y0 - margin >= b2.y1);
}

} // namespace

auto run_ocr_tsv(const std::filesystem::path& image_path) -> std::vector<TextItem> {
    const std::string cmd = "tesseract \"" + image_path.string() + "\" stdout tsv 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {};

    std::vector<TextItem> items;
    char buffer[1024];
    bool header = true;

    while (fgets(buffer, sizeof(buffer), pipe)) {
        if (header) {
            header = false;
            continue;
        }

        // Columns: level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text
        std::stringstream ss(buffer);
        std::string token;
        std::vector<std::string> cols;
        while (std::getline(ss, token, '\t')) {
            cols.push_back(token);
        }

        if (cols.size() >= 12) {
            std::string text = cols[11];
            while (!text.empty() && (text.back() == '\r' || text.back() == '\n' || text.back() == ' '))
                text.pop_back();

            if (!text.empty()) {
                try {
                    TextItem it;
                    it.level     = std::stoi(cols[0]);
                    it.block_num = std::stoi(cols[2]);
                    it.line_num  = std::stoi(cols[4]);
                    it.word_num  = std::stoi(cols[5]);
                    const int left   = std::stoi(cols[6]);
                    const int top    = std::stoi(cols[7]);
                    const int width  = std::stoi(cols[8]);
                    const int height = std::stoi(cols[9]);
                    it.bbox = Box{left, top, left + width, top + height};
                    it.conf = std::stof(cols[10]);
                    it.text = std::move(text);
                    items.push_back(std::move(it));
                } catch (...) {
                    // skip malformed row
                }
            }
        }
    }
    pclose(pipe);
    return items;
}

auto merge_word_boxes(const std::vector<TextItem>& items, int max_x_gap, int max_y_gap) -> std::vector<TextBlock> {
    if (items.empty()) return {};

    auto sorted = items;
    std::sort(sorted.begin(), sorted.end(), [&](const TextItem& a, const TextItem& b) {
        if (std::abs(a.bbox.y0 - b.bbox.y0) <= max_y_gap)
            return a.bbox.x0 < b.bbox.x0;
        return a.bbox.y0 < b.bbox.y0;
    });

    std::vector<TextBlock> blocks;
    TextBlock cur;
    cur.bbox = sorted[0].bbox;
    cur.text = sorted[0].text;
    cur.avg_conf = sorted[0].conf;
    int count = 1;

    for (size_t i = 1; i < sorted.size(); ++i) {
        const auto& w = sorted[i];
        const int x_dist = std::max(0, std::max(cur.bbox.x0, w.bbox.x0) - std::min(cur.bbox.x1, w.bbox.x1));
        const int y_dist = std::max(0, std::max(cur.bbox.y0, w.bbox.y0) - std::min(cur.bbox.y1, w.bbox.y1));

        if (x_dist <= max_x_gap && y_dist <= max_y_gap) {
            cur.bbox.x0 = std::min(cur.bbox.x0, w.bbox.x0);
            cur.bbox.y0 = std::min(cur.bbox.y0, w.bbox.y0);
            cur.bbox.x1 = std::max(cur.bbox.x1, w.bbox.x1);
            cur.bbox.y1 = std::max(cur.bbox.y1, w.bbox.y1);
            cur.text += " " + w.text;
            cur.avg_conf += w.conf;
            ++count;
        } else {
            cur.avg_conf /= count;
            blocks.push_back(std::move(cur));
            cur.bbox = w.bbox;
            cur.text = w.text;
            cur.avg_conf = w.conf;
            count = 1;
        }
    }
    cur.avg_conf /= count;
    blocks.push_back(std::move(cur));

    return blocks;
}

auto shrinkwrap_bbox(const ImageRGB& img, const Box& bbox, int padding) -> Box {
    const int x0 = std::clamp(bbox.x0, 0, img.width);
    const int y0 = std::clamp(bbox.y0, 0, img.height);
    const int x1 = std::clamp(bbox.x1, 0, img.width);
    const int y1 = std::clamp(bbox.y1, 0, img.height);

    if (x1 <= x0 || y1 <= y0) return bbox;

    int min_x = img.width, min_y = img.height, max_x = -1, max_y = -1;

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const int idx = (y * img.width + x) * 3;
            // Grayscale luminance
            const int lum = (img.data[idx] * 299 + img.data[idx + 1] * 587 + img.data[idx + 2] * 114) / 1000;
            if (lum < 180) { // ink pixel
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }
    }

    if (max_x >= min_x && max_y >= min_y) {
        return Box{
            std::max(0, min_x - padding),
            std::max(0, min_y - padding),
            std::min(img.width, max_x + 1 + padding),
            std::min(img.height, max_y + 1 + padding)
        };
    }

    return Box{
        std::max(0, x0 - padding),
        std::max(0, y0 - padding),
        std::min(img.width, x1 + padding),
        std::min(img.height, y1 + padding)
    };
}

auto generate_counterfactuals(int img_w, int img_h, const std::vector<Box>& text_boxes, int num_samples)
    -> std::vector<CounterfactualItem> {
    std::mt19937_64 rng(1337);
    std::vector<CounterfactualItem> items;
    items.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        for (int attempt = 0; attempt < 250; ++attempt) {
            const int bw = 40 + static_cast<int>(rng() % 100);
            const int bh = 18 + static_cast<int>(rng() % 24);
            if (bw >= img_w || bh >= img_h) continue;

            const int bx0 = static_cast<int>(rng() % (img_w - bw));
            const int by0 = static_cast<int>(rng() % (img_h - bh));
            const Box cand{bx0, by0, bx0 + bw, by0 + bh};

            bool collision = false;
            for (const auto& tb : text_boxes) {
                if (boxes_overlap(cand, tb, 10)) { collision = true; break; }
            }
            if (!collision) {
                for (const auto& cf : items) {
                    if (boxes_overlap(cand, cf.bbox, 6)) { collision = true; break; }
                }
            }

            if (!collision) {
                items.push_back(CounterfactualItem{cand, 1.0f});
                break;
            }
        }
    }

    return items;
}

auto process_image(const std::filesystem::path& img_path,
                   const std::filesystem::path& out_dir,
                   int padding,
                   int num_cf,
                   float min_conf,
                   bool verbose) -> std::optional<RedactResult> {
    auto img_opt = ImageRGB::load(img_path);
    if (!img_opt) return std::nullopt;
    auto img = *img_opt;

    const auto ocr_items = run_ocr_tsv(img_path);
    std::vector<TextItem> valid_items;
    for (const auto& it : ocr_items) {
        if (it.conf >= min_conf) valid_items.push_back(it);
    }

    const auto raw_blocks = merge_word_boxes(valid_items);
    std::vector<TextBlock> shrinkwrapped;
    shrinkwrapped.reserve(raw_blocks.size());

    std::vector<Box> all_boxes;
    for (const auto& blk : raw_blocks) {
        const auto s_box = shrinkwrap_bbox(img, blk.bbox, padding);
        shrinkwrapped.push_back(TextBlock{s_box, blk.text, blk.avg_conf});
        all_boxes.push_back(s_box);
    }

    const auto counterfactuals = generate_counterfactuals(img.width, img.height, all_boxes, num_cf);

    const auto stem = img_path.stem().string();
    const auto sub_dir = out_dir / stem;
    std::filesystem::create_directories(sub_dir / "patches");
    std::filesystem::create_directories(sub_dir / "steps");

    // Save original image
    img.save_png(sub_dir / "original.png");

    auto current_img = img;
    std::ofstream meta(sub_dir / "metadata.json");
    meta << "{\n  \"source_image\": \"" << img_path.string() << "\",\n";
    meta << "  \"dimensions\": [" << img.width << ", " << img.height << "],\n";
    meta << "  \"text_redactions\": [\n";

    for (size_t i = 0; i < shrinkwrapped.size(); ++i) {
        const auto& blk = shrinkwrapped[i];
        // 1. Save unredacted patch behind
        const auto patch = img.crop(blk.bbox);
        char patch_name[128];
        std::snprintf(patch_name, sizeof(patch_name), "patch_%03zu_%d_%d_%d_%d.png",
                      i, blk.bbox.x0, blk.bbox.y0, blk.bbox.x1, blk.bbox.y1);
        patch.save_png(sub_dir / "patches" / patch_name);

        // 2. Black out on current image
        current_img.fill_rect(blk.bbox, 0, 0, 0);

        // 3. Save step image
        char step_name[64];
        std::snprintf(step_name, sizeof(step_name), "step_%03zu_redacted.png", i + 1);
        current_img.save_png(sub_dir / "steps" / step_name);

        meta << "    {\n";
        meta << "      \"id\": " << i << ",\n";
        meta << "      \"bbox\": [" << blk.bbox.x0 << ", " << blk.bbox.y0 << ", "
             << blk.bbox.x1 << ", " << blk.bbox.y1 << "],\n";
        meta << "      \"text\": \"" << blk.text << "\",\n";
        meta << "      \"confidence\": " << blk.avg_conf << ",\n";
        meta << "      \"patch_file\": \"patches/" << patch_name << "\",\n";
        meta << "      \"step_file\": \"steps/" << step_name << "\"\n";
        meta << "    }" << (i + 1 < shrinkwrapped.size() ? "," : "") << "\n";
    }

    meta << "  ],\n  \"counterfactual_samples\": [\n";
    for (size_t i = 0; i < counterfactuals.size(); ++i) {
        const auto& cf = counterfactuals[i];
        const auto cf_patch = img.crop(cf.bbox);
        char cf_name[128];
        std::snprintf(cf_name, sizeof(cf_name), "cf_patch_%03zu_%d_%d_%d_%d.png",
                      i, cf.bbox.x0, cf.bbox.y0, cf.bbox.x1, cf.bbox.y1);
        cf_patch.save_png(sub_dir / "patches" / cf_name);

        meta << "    {\n";
        meta << "      \"id\": " << i << ",\n";
        meta << "      \"bbox\": [" << cf.bbox.x0 << ", " << cf.bbox.y0 << ", "
             << cf.bbox.x1 << ", " << cf.bbox.y1 << "],\n";
        meta << "      \"error_rate\": 1.0,\n";
        meta << "      \"patch_file\": \"patches/" << cf_name << "\"\n";
        meta << "    }" << (i + 1 < counterfactuals.size() ? "," : "") << "\n";
    }
    meta << "  ]\n}\n";

    current_img.save_png(sub_dir / "final_redacted.png");

    if (verbose) {
        std::println("  ✓ {}: {} text blocks, {} counterfactuals",
                     img_path.filename().string(), shrinkwrapped.size(), counterfactuals.size());
    }

    return RedactResult{
        img_path, img.width, img.height,
        static_cast<int>(shrinkwrapped.size()),
        static_cast<int>(counterfactuals.size())
    };
}

} // namespace xn::redact

auto main(int argc, char** argv) -> int {
    if (argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") {
        std::println(
            "redact — Document text redaction and dataset generator (C++26)\n\n"
            "Usage: redact <image_or_directory> [options]\n\n"
            "Options:\n"
            "  -o, --output-dir DIR   Output directory (default: dataset_redacted)\n"
            "  --padding N            Bounding box padding in pixels (default: 4)\n"
            "  --counterfactuals N    Number of negative non-text boxes (default: 5)\n"
            "  --min-conf C           Minimum OCR confidence threshold (default: 20.0)\n"
            "  --verbose              Print execution details\n");
        return 0;
    }

    std::filesystem::path input_path = argv[1];
    std::filesystem::path out_dir = "dataset_redacted";
    int padding = 4;
    int num_cf = 5;
    float min_conf = 20.0f;
    bool verbose = false;

    for (int i = 2; i < argc; ++i) {
        std::string_view a = argv[i];
        if ((a == "-o" || a == "--output-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--padding" && i + 1 < argc) padding = std::stoi(argv[++i]);
        else if (a == "--counterfactuals" && i + 1 < argc) num_cf = std::stoi(argv[++i]);
        else if (a == "--min-conf" && i + 1 < argc) min_conf = std::stof(argv[++i]);
        else if (a == "--verbose") verbose = true;
    }

    std::vector<std::filesystem::path> files;
    if (std::filesystem::is_regular_file(input_path)) {
        files.push_back(input_path);
    } else if (std::filesystem::is_directory(input_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(input_path)) {
            const auto ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
                files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
    } else {
        std::println(stderr, "redact: cannot access '{}'", input_path.string());
        return 1;
    }

    std::println("◈ redact: processing {} image(s) -> {}", files.size(), out_dir.string());
    for (const auto& p : files) {
        auto res = xn::redact::process_image(p, out_dir, padding, num_cf, min_conf, verbose);
        if (!res) {
            std::println(stderr, "  ✗ failed to process {}", p.string());
        }
    }
    std::println("◈ redact: complete");
    return 0;
}
