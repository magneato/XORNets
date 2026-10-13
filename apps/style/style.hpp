#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include <optional>
#include <array>
#include <cmath>
#include <algorithm>
#include <span>

#include "vendor/stb/stb_image.h"
#include "vendor/stb/stb_image_write.h"

namespace xn::style {

struct PixelLab {
    float L = 0.0f; // 0..100
    float a = 0.0f; // -128..127
    float b = 0.0f; // -128..127
};

// Fast D65 sRGB to CIE Lab converter
inline auto rgb_to_lab(std::uint8_t r_byte, std::uint8_t g_byte, std::uint8_t b_byte) noexcept -> PixelLab {
    auto pivot = [](float n) noexcept -> float {
        return (n > 0.04045f) ? std::pow((n + 0.055f) / 1.055f, 2.4f) : (n / 12.92f);
    };
    float r = pivot(r_byte / 255.0f);
    float g = pivot(g_byte / 255.0f);
    float b = pivot(b_byte / 255.0f);

    float x = (r * 0.4124564f + g * 0.3575761f + b * 0.1804375f) / 0.95047f;
    float y = (r * 0.2126729f + g * 0.7151522f + b * 0.0721750f) / 1.00000f;
    float z = (r * 0.0193339f + g * 0.1191920f + b * 0.9503041f) / 1.08883f;

    auto f = [](float t) noexcept -> float {
        return (t > 0.008856f) ? std::cbrt(t) : (7.787f * t + 16.0f / 116.0f);
    };
    float fx = f(x), fy = f(y), fz = f(z);

    return PixelLab{
        .L = (116.0f * fy) - 16.0f,
        .a = 500.0f * (fx - fy),
        .b = 200.0f * (fy - fz)
    };
}

// Fast CIE Lab to sRGB converter
inline void lab_to_rgb(const PixelLab& lab, std::uint8_t& r_out, std::uint8_t& g_out, std::uint8_t& b_out) noexcept {
    float fy = (lab.L + 16.0f) / 116.0f;
    float fx = lab.a / 500.0f + fy;
    float fz = fy - lab.b / 200.0f;

    auto inv_f = [](float t) noexcept -> float {
        float t3 = t * t * t;
        return (t3 > 0.008856f) ? t3 : ((t - 16.0f / 116.0f) / 7.787f);
    };

    float xr = inv_f(fx) * 0.95047f;
    float yr = inv_f(fy) * 1.00000f;
    float zr = inv_f(fz) * 1.08883f;

    float r = xr *  3.2404542f + yr * -1.5371385f + zr * -0.4985314f;
    float g = xr * -0.9692660f + yr *  1.8760108f + zr *  0.0415560f;
    float b = xr *  0.0556434f + yr * -0.2040259f + zr *  1.0572252f;

    auto unpivot = [](float n) noexcept -> float {
        return (n > 0.0031308f) ? (1.055f * std::pow(n, 1.0f / 2.4f) - 0.055f) : (12.92f * n);
    };

    r_out = static_cast<std::uint8_t>(std::clamp(std::round(unpivot(r) * 255.0f), 0.0f, 255.0f));
    g_out = static_cast<std::uint8_t>(std::clamp(std::round(unpivot(g) * 255.0f), 0.0f, 255.0f));
    b_out = static_cast<std::uint8_t>(std::clamp(std::round(unpivot(b) * 255.0f), 0.0f, 255.0f));
}

// 7-Plank Bit Plane Descriptor per pixel (packed in 64-bit uint64_t)
// Planks 0..3: Luminance thresholds (L > 25, L > 50, L > 75, L > 100)
// Planks 4..6: Inter-stack XOR planks (P0 ^ P1, P1 ^ P2, P2 ^ P3)
// Additional bits for color chrominance quadrant (a > 0, b > 0)
// and Local Edge / Contrast anomaly bits
struct BitTensor64 {
    std::uint64_t planes = 0ULL;

    [[nodiscard]] constexpr auto get_plane(int i) const noexcept -> bool {
        return (planes & (1ULL << i)) != 0;
    }
    constexpr void set_plane(int i, bool val) noexcept {
        if (val) planes |= (1ULL << i);
        else planes &= ~(1ULL << i);
    }
};

struct ImageBuffer {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb; // 3 bytes per pixel

    [[nodiscard]] auto valid() const noexcept -> bool { return width > 0 && height > 0 && !rgb.empty(); }

    [[nodiscard]] static auto load(const std::filesystem::path& path) -> std::optional<ImageBuffer> {
        int w = 0, h = 0, comp = 0;
        stbi_uc* raw = stbi_load(path.string().c_str(), &w, &h, &comp, 3);
        if (!raw) return std::nullopt;

        ImageBuffer buf;
        buf.width = w;
        buf.height = h;
        buf.rgb.assign(raw, raw + (w * h * 3));
        stbi_image_free(raw);
        return buf;
    }

    auto save_png(const std::filesystem::path& path) const -> bool {
        if (!valid()) return false;
        return stbi_write_png(path.string().c_str(), width, height, 3, rgb.data(), width * 3) != 0;
    }

    [[nodiscard]] auto resize(int new_w, int new_h) const -> ImageBuffer {
        ImageBuffer out;
        out.width = new_w;
        out.height = new_h;
        out.rgb.resize(new_w * new_h * 3);

        const float sx = static_cast<float>(width) / new_w;
        const float sy = static_cast<float>(height) / new_h;

        for (int y = 0; y < new_h; ++y) {
            for (int x = 0; x < new_w; ++x) {
                const int src_x = std::clamp(static_cast<int>(x * sx), 0, width - 1);
                const int src_y = std::clamp(static_cast<int>(y * sy), 0, height - 1);
                const int s_idx = (src_y * width + src_x) * 3;
                const int d_idx = (y * new_w + x) * 3;
                out.rgb[d_idx]     = rgb[s_idx];
                out.rgb[d_idx + 1] = rgb[s_idx + 1];
                out.rgb[d_idx + 2] = rgb[s_idx + 2];
            }
        }
        return out;
    }
};

// Edge & Structure Features extracted using Sobel + Local Variance (Anomalies)
struct EdgeAnomalyMap {
    int width = 0;
    int height = 0;
    std::vector<float> edge_mag; // 0..1 normalized edge gradient
    std::vector<float> edge_angle; // -PI..PI gradient direction
    std::vector<float> local_variance; // Texture / anomaly density
};

auto compute_edges_and_anomalies(const ImageBuffer& img) -> EdgeAnomalyMap;

// Global Color Palette Moment Matching (Reinhard meta-color alignment)
auto transfer_color_moments(const ImageBuffer& content, const ImageBuffer& style) -> ImageBuffer;

// Style Transfer Config
struct TransferConfig {
    int width = 512;
    int height = 512;
    int iterations = 1200000;
    float content_weight = 1.0f;
    float style_pattern_weight = 1.6f;
    float edge_weight = 2.4f;
    float anomaly_weight = 1.8f;
    float jitter_prob = 0.35f;
    std::uint64_t seed = 42;
    bool interactive = false;
};

// State of the ongoing XOR-style engine
struct EngineState {
    int width = 0;
    int height = 0;
    ImageBuffer current_canvas;
    EdgeAnomalyMap content_edges;
    EdgeAnomalyMap style_edges;
    std::vector<BitTensor64> style_planks;
    std::vector<BitTensor64> content_planks;
    std::vector<PixelLab> style_palette;
    
    // Performance counters & loss tracking
    std::uint64_t iterations_done = 0;
    std::uint64_t flips_accepted = 0;
    float current_loss = 1e9f;
};

// High-level style transfer engine
class XonStyleTransfer {
public:
    explicit XonStyleTransfer(TransferConfig cfg);

    auto initialize(const ImageBuffer& content_img, const ImageBuffer& style_img) -> bool;
    auto step_batch(int count) -> float; // Run batch of stochastic bit/color jitters
    [[nodiscard]] const auto& get_state() const noexcept { return state_; }
    [[nodiscard]] const auto& get_canvas() const noexcept { return state_.current_canvas; }

private:
    TransferConfig cfg_;
    EngineState state_;
    
    auto compute_loss_at(int x, int y, const PixelLab& candidate) const noexcept -> float;
};

} // namespace xn::style
