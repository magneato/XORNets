#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <filesystem>
#include <optional>
#include <algorithm>

#include "vendor/stb/stb_image.h"
#include "vendor/stb/stb_image_write.h"

namespace xn::redact {

struct Box {
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;

    [[nodiscard]] constexpr auto width() const noexcept -> int { return x1 - x0; }
    [[nodiscard]] constexpr auto height() const noexcept -> int { return y1 - y0; }
    [[nodiscard]] constexpr auto area() const noexcept -> int { return width() * height(); }
};

struct TextItem {
    int level = 0;
    int block_num = 0;
    int line_num = 0;
    int word_num = 0;
    Box bbox{};
    float conf = 0.0f;
    std::string text;
};

struct TextBlock {
    Box bbox{};
    std::string text;
    float avg_conf = 0.0f;
};

struct CounterfactualItem {
    Box bbox{};
    float error_rate = 1.0f; // 100% error rate
};

struct ImageRGB {
    int width = 0;
    int height = 0;
    int channels = 3;
    std::vector<std::uint8_t> data;

    [[nodiscard]] auto valid() const noexcept -> bool { return width > 0 && height > 0 && !data.empty(); }

    [[nodiscard]] static inline auto load(const std::filesystem::path& path) -> std::optional<ImageRGB> {
        int w = 0, h = 0, comp = 0;
        stbi_uc* raw = stbi_load(path.string().c_str(), &w, &h, &comp, 3);
        if (!raw) return std::nullopt;

        ImageRGB img;
        img.width = w;
        img.height = h;
        img.channels = 3;
        img.data.assign(raw, raw + (w * h * 3));
        stbi_image_free(raw);
        return img;
    }

    inline auto save_png(const std::filesystem::path& path) const -> bool {
        if (!valid()) return false;
        int res = stbi_write_png(path.string().c_str(), width, height, 3, data.data(), width * 3);
        return res != 0;
    }

    [[nodiscard]] inline auto crop(const Box& b) const -> ImageRGB {
        ImageRGB out;
        const int x0 = std::clamp(b.x0, 0, width);
        const int y0 = std::clamp(b.y0, 0, height);
        const int x1 = std::clamp(b.x1, 0, width);
        const int y1 = std::clamp(b.y1, 0, height);
        const int cw = std::max(0, x1 - x0);
        const int ch = std::max(0, y1 - y0);

        out.width = cw;
        out.height = ch;
        out.channels = 3;
        out.data.resize(cw * ch * 3, 0);

        for (int y = 0; y < ch; ++y) {
            for (int x = 0; x < cw; ++x) {
                const int src_idx = ((y0 + y) * width + (x0 + x)) * 3;
                const int dst_idx = (y * cw + x) * 3;
                out.data[dst_idx]     = data[src_idx];
                out.data[dst_idx + 1] = data[src_idx + 1];
                out.data[dst_idx + 2] = data[src_idx + 2];
            }
        }
        return out;
    }

    inline void fill_rect(const Box& b, std::uint8_t r, std::uint8_t g, std::uint8_t b_col) {
        const int x0 = std::clamp(b.x0, 0, width);
        const int y0 = std::clamp(b.y0, 0, height);
        const int x1 = std::clamp(b.x1, 0, width);
        const int y1 = std::clamp(b.y1, 0, height);

        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const int idx = (y * width + x) * 3;
                data[idx]     = r;
                data[idx + 1] = g;
                data[idx + 2] = b_col;
            }
        }
    }

    inline void draw_text(int start_x, int start_y, std::string_view text,
                          std::uint8_t r = 20, std::uint8_t g = 20, std::uint8_t b_col = 30) {
        int cur_x = start_x;
        int cur_y = start_y;

        for (char ch : text) {
            if (ch == '\n') {
                cur_x = start_x;
                cur_y += 12;
                continue;
            }
            if (cur_x + 8 >= width) break;

            for (int dy = 0; dy < 8; ++dy) {
                for (int dx = 0; dx < 6; ++dx) {
                    const int px = cur_x + dx;
                    const int py = cur_y + dy;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        bool dot = false;
                        if (std::isalnum(static_cast<unsigned char>(ch))) {
                            if (dy == 0 || dy == 7 || dx == 0 || (dy == 3 && dx < 5)) dot = true;
                        } else if (ch != ' ') {
                            if (dy == 4) dot = true;
                        }
                        if (dot) {
                            const int idx = (py * width + px) * 3;
                            data[idx] = r;
                            data[idx + 1] = g;
                            data[idx + 2] = b_col;
                        }
                    }
                }
            }
            cur_x += 7;
        }
    }
};

// OCR via Tesseract TSV
auto run_ocr_tsv(const std::filesystem::path& image_path) -> std::vector<TextItem>;
auto merge_word_boxes(const std::vector<TextItem>& items, int max_x_gap = 25, int max_y_gap = 15) -> std::vector<TextBlock>;
auto shrinkwrap_bbox(const ImageRGB& img, const Box& bbox, int padding = 4) -> Box;
auto generate_counterfactuals(int img_w, int img_h, const std::vector<Box>& text_boxes, int num_samples = 5) -> std::vector<CounterfactualItem>;

// Dataset Generation Pipeline
struct RedactResult {
    std::filesystem::path input_path;
    int width = 0;
    int height = 0;
    int text_blocks = 0;
    int counterfactuals = 0;
};

auto process_image(const std::filesystem::path& img_path,
                   const std::filesystem::path& out_dir,
                   int padding = 4,
                   int num_cf = 5,
                   float min_conf = 20.0f,
                   bool verbose = false) -> std::optional<RedactResult>;

} // namespace xn::redact
