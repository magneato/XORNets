#include "style.hpp"

#include <numeric>
#include <random>
#include <cstring>
#include <iostream>
#include <print>

namespace xn::style {

namespace {

inline auto get_pixel_rgb(const ImageBuffer& img, int x, int y) noexcept -> std::array<std::uint8_t, 3> {
    const int idx = (std::clamp(y, 0, img.height - 1) * img.width + std::clamp(x, 0, img.width - 1)) * 3;
    return {img.rgb[idx], img.rgb[idx + 1], img.rgb[idx + 2]};
}

inline auto encode_bit_tensor(const PixelLab& lab, float edge_val) noexcept -> BitTensor64 {
    BitTensor64 tensor{};
    // Planks 0..3: Luminance thresholds
    const bool p0 = lab.L > 25.0f;
    const bool p1 = lab.L > 50.0f;
    const bool p2 = lab.L > 75.0f;
    const bool p3 = lab.L > 90.0f;
    tensor.set_plane(0, p0);
    tensor.set_plane(1, p1);
    tensor.set_plane(2, p2);
    tensor.set_plane(3, p3);

    // Planks 4..6: Inter-stack XOR planes
    tensor.set_plane(4, p0 ^ p1);
    tensor.set_plane(5, p1 ^ p2);
    tensor.set_plane(6, p2 ^ p3);

    // Planks 7..10: Chrominance quadrants
    tensor.set_plane(7, lab.a > 0.0f);
    tensor.set_plane(8, lab.b > 0.0f);
    tensor.set_plane(9, lab.a > 15.0f || lab.a < -15.0f);
    tensor.set_plane(10, lab.b > 15.0f || lab.b < -15.0f);

    // Plank 11: Edge presence
    tensor.set_plane(11, edge_val > 0.35f);
    return tensor;
}

} // namespace

auto compute_edges_and_anomalies(const ImageBuffer& img) -> EdgeAnomalyMap {
    const int w = img.width;
    const int h = img.height;
    EdgeAnomalyMap map;
    map.width = w;
    map.height = h;
    map.edge_mag.resize(w * h, 0.0f);
    map.edge_angle.resize(w * h, 0.0f);
    map.local_variance.resize(w * h, 0.0f);

    std::vector<float> lums(w * h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int idx = (y * w + x) * 3;
            // Photometric luminance
            lums[y * w + x] = (img.rgb[idx] * 0.299f + img.rgb[idx + 1] * 0.587f + img.rgb[idx + 2] * 0.114f) / 255.0f;
        }
    }

    // Sobel filters for edge gradients + Local 5x5 variance for anomaly texture map
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            // Sobel kernels
            const float tl = lums[(y - 1) * w + (x - 1)];
            const float tc = lums[(y - 1) * w + x];
            const float tr = lums[(y - 1) * w + (x + 1)];
            const float ml = lums[y * w + (x - 1)];
            const float mr = lums[y * w + (x + 1)];
            const float bl = lums[(y + 1) * w + (x - 1)];
            const float bc = lums[(y + 1) * w + x];
            const float br = lums[(y + 1) * w + (x + 1)];

            const float gx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
            const float gy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
            const float mag = std::clamp(std::sqrt(gx * gx + gy * gy) * 1.5f, 0.0f, 1.0f);
            map.edge_mag[y * w + x] = mag;
            map.edge_angle[y * w + x] = std::atan2(gy, gx);

            // 5x5 local variance
            float mean = 0.0f;
            float sq_sum = 0.0f;
            int count = 0;
            for (int dy = -2; dy <= 2; ++dy) {
                int py = std::clamp(y + dy, 0, h - 1);
                for (int dx = -2; dx <= 2; ++dx) {
                    int px = std::clamp(x + dx, 0, w - 1);
                    float val = lums[py * w + px];
                    mean += val;
                    sq_sum += val * val;
                    count++;
                }
            }
            mean /= count;
            float var = std::clamp(std::sqrt(std::max(0.0f, (sq_sum / count) - (mean * mean))) * 3.0f, 0.0f, 1.0f);
            map.local_variance[y * w + x] = var;
        }
    }

    return map;
}

auto transfer_color_moments(const ImageBuffer& content, const ImageBuffer& style) -> ImageBuffer {
    const int w = content.width;
    const int h = content.height;
    ImageBuffer out = content;

    // Convert content and style to CIE Lab to compute means and standard deviations
    std::vector<PixelLab> content_labs(w * h);
    float mean_c_L = 0.0f, mean_c_a = 0.0f, mean_c_b = 0.0f;
    for (int i = 0; i < w * h; ++i) {
        content_labs[i] = rgb_to_lab(content.rgb[i * 3], content.rgb[i * 3 + 1], content.rgb[i * 3 + 2]);
        mean_c_L += content_labs[i].L;
        mean_c_a += content_labs[i].a;
        mean_c_b += content_labs[i].b;
    }
    mean_c_L /= (w * h);
    mean_c_a /= (w * h);
    mean_c_b /= (w * h);

    float var_c_L = 0.0f, var_c_a = 0.0f, var_c_b = 0.0f;
    for (int i = 0; i < w * h; ++i) {
        var_c_L += (content_labs[i].L - mean_c_L) * (content_labs[i].L - mean_c_L);
        var_c_a += (content_labs[i].a - mean_c_a) * (content_labs[i].a - mean_c_a);
        var_c_b += (content_labs[i].b - mean_c_b) * (content_labs[i].b - mean_c_b);
    }
    float std_c_L = std::sqrt(var_c_L / (w * h)) + 1e-4f;
    float std_c_a = std::sqrt(var_c_a / (w * h)) + 1e-4f;
    float std_c_b = std::sqrt(var_c_b / (w * h)) + 1e-4f;

    // Style statistics
    const int sw = style.width;
    const int sh = style.height;
    float mean_s_L = 0.0f, mean_s_a = 0.0f, mean_s_b = 0.0f;
    for (int i = 0; i < sw * sh; ++i) {
        auto lab = rgb_to_lab(style.rgb[i * 3], style.rgb[i * 3 + 1], style.rgb[i * 3 + 2]);
        mean_s_L += lab.L;
        mean_s_a += lab.a;
        mean_s_b += lab.b;
    }
    mean_s_L /= (sw * sh);
    mean_s_a /= (sw * sh);
    mean_s_b /= (sw * sh);

    float var_s_L = 0.0f, var_s_a = 0.0f, var_s_b = 0.0f;
    for (int i = 0; i < sw * sh; ++i) {
        auto lab = rgb_to_lab(style.rgb[i * 3], style.rgb[i * 3 + 1], style.rgb[i * 3 + 2]);
        var_s_L += (lab.L - mean_s_L) * (lab.L - mean_s_L);
        var_s_a += (lab.a - mean_s_a) * (lab.a - mean_s_a);
        var_s_b += (lab.b - mean_s_b) * (lab.b - mean_s_b);
    }
    float std_s_L = std::sqrt(var_s_L / (sw * sh)) + 1e-4f;
    float std_s_a = std::sqrt(var_s_a / (sw * sh)) + 1e-4f;
    float std_s_b = std::sqrt(var_s_b / (sw * sh)) + 1e-4f;

    // Transfer Reinhard moments
    for (int i = 0; i < w * h; ++i) {
        PixelLab adj;
        adj.L = ((content_labs[i].L - mean_c_L) * (std_s_L / std_c_L)) + mean_s_L;
        adj.a = ((content_labs[i].a - mean_c_a) * (std_s_a / std_c_a)) + mean_s_a;
        adj.b = ((content_labs[i].b - mean_c_b) * (std_s_b / std_c_b)) + mean_s_b;

        adj.L = std::clamp(adj.L, 0.0f, 100.0f);
        lab_to_rgb(adj, out.rgb[i * 3], out.rgb[i * 3 + 1], out.rgb[i * 3 + 2]);
    }

    return out;
}

XonStyleTransfer::XonStyleTransfer(TransferConfig cfg)
    : cfg_(std::move(cfg)) {}

auto XonStyleTransfer::initialize(const ImageBuffer& content_img, const ImageBuffer& style_img) -> bool {
    if (!content_img.valid() || !style_img.valid()) return false;

    state_.width = cfg_.width;
    state_.height = cfg_.height;

    // Normalize images to target resolution
    const auto resized_content = content_img.resize(cfg_.width, cfg_.height);
    const auto resized_style = style_img.resize(cfg_.width, cfg_.height);

    // Initial canvas: Transfer color moments so base palette already has Starry Night tones
    state_.current_canvas = transfer_color_moments(resized_content, resized_style);

    // Compute edge gradient maps and texture anomaly variance
    state_.content_edges = compute_edges_and_anomalies(resized_content);
    state_.style_edges = compute_edges_and_anomalies(resized_style);

    // Build Style Palette from representative cluster samples of the style image
    state_.style_palette.clear();
    state_.style_palette.reserve(512);
    for (int y = 0; y < resized_style.height; y += 8) {
        for (int x = 0; x < resized_style.width; x += 8) {
            auto rgb = get_pixel_rgb(resized_style, x, y);
            state_.style_palette.push_back(rgb_to_lab(rgb[0], rgb[1], rgb[2]));
        }
    }

    // Build 7-Plank Bit Tensors
    const int n_pixels = cfg_.width * cfg_.height;
    state_.content_planks.resize(n_pixels);
    state_.style_planks.resize(n_pixels);

    for (int y = 0; y < cfg_.height; ++y) {
        for (int x = 0; x < cfg_.width; ++x) {
            const int idx = y * cfg_.width + x;
            auto c_rgb = get_pixel_rgb(resized_content, x, y);
            auto c_lab = rgb_to_lab(c_rgb[0], c_rgb[1], c_rgb[2]);
            state_.content_planks[idx] = encode_bit_tensor(c_lab, state_.content_edges.edge_mag[idx]);

            auto s_rgb = get_pixel_rgb(resized_style, x, y);
            auto s_lab = rgb_to_lab(s_rgb[0], s_rgb[1], s_rgb[2]);
            state_.style_planks[idx] = encode_bit_tensor(s_lab, state_.style_edges.edge_mag[idx]);
        }
    }

    state_.iterations_done = 0;
    state_.flips_accepted = 0;
    state_.current_loss = 0.0f;

    // Initial loss computation
    for (int y = 0; y < cfg_.height; ++y) {
        for (int x = 0; x < cfg_.width; ++x) {
            auto rgb = get_pixel_rgb(state_.current_canvas, x, y);
            auto lab = rgb_to_lab(rgb[0], rgb[1], rgb[2]);
            state_.current_loss += compute_loss_at(x, y, lab);
        }
    }

    return true;
}

auto XonStyleTransfer::compute_loss_at(int x, int y, const PixelLab& candidate) const noexcept -> float {
    const int idx = y * state_.width + x;

    // 1. Content preservation loss (Luminance structure and position)
    auto cur_tensor = encode_bit_tensor(candidate, state_.content_edges.edge_mag[idx]);
    const auto& content_t = state_.content_planks[idx];
    
    // XOR distance between bit planes
    const std::uint64_t xor_diff = cur_tensor.planes ^ content_t.planes;
    const float bit_dist = static_cast<float>(std::popcount(xor_diff));

    // 2. Edge alignment penalty
    // An edge in content should maintain high contrast / bit transitions
    float edge_penalty = 0.0f;
    if (state_.content_edges.edge_mag[idx] > 0.4f) {
        // Must preserve edge plank 11 and inter-stack XOR planks 4..6
        if (!cur_tensor.get_plane(11)) edge_penalty += 3.5f;
        if (cur_tensor.get_plane(4) != content_t.get_plane(4)) edge_penalty += 2.0f;
    }

    // 3. Style Pattern & Color Affinity
    // Find closest style palette match
    float min_color_dist = 1e9f;
    for (size_t i = 0; i < std::min<size_t>(state_.style_palette.size(), 48); ++i) {
        const auto& s_col = state_.style_palette[i];
        const float dL = candidate.L - s_col.L;
        const float da = candidate.a - s_col.a;
        const float db = candidate.b - s_col.b;
        const float d = (dL * dL * 0.45f) + (da * da) + (db * db);
        if (d < min_color_dist) min_color_dist = d;
    }
    const float style_loss = std::sqrt(min_color_dist);

    // 4. Texture anomaly & swirl matching
    // Higher variance regions should match starry night brush strokes & directional flows
    float anomaly_cost = 0.0f;
    if (state_.style_edges.local_variance[idx] > 0.3f) {
        // Encourage plank 5 & 6 inter-stack XOR transitions to reflect painterly brushstrokes
        const float expected_swirl = state_.style_edges.local_variance[idx];
        const float current_var = cur_tensor.get_plane(5) ? 1.0f : 0.0f;
        anomaly_cost += std::abs(current_var - expected_swirl) * 2.0f;
    }

    // Directional stroke coherence along edges
    float directional_cost = 0.0f;
    if (state_.content_edges.edge_mag[idx] > 0.35f) {
        // Check neighboring pixel along tangent for luminance continuity
        const float theta = state_.content_edges.edge_angle[idx] + 1.5707963f; // Tangent direction
        const int dx = static_cast<int>(std::round(std::cos(theta)));
        const int dy = static_cast<int>(std::round(std::sin(theta)));
        const int nx = std::clamp(x + dx, 0, state_.width - 1);
        const int ny = std::clamp(y + dy, 0, state_.height - 1);
        const int n_idx = (ny * state_.width + nx) * 3;
        const float n_lum = (state_.current_canvas.rgb[n_idx] * 0.299f +
                             state_.current_canvas.rgb[n_idx + 1] * 0.587f +
                             state_.current_canvas.rgb[n_idx + 2] * 0.114f);
        directional_cost = std::abs((candidate.L * 2.55f) - n_lum) * 0.015f;
    }

    return (cfg_.content_weight * bit_dist) +
           (cfg_.edge_weight * edge_penalty) +
           (cfg_.style_pattern_weight * style_loss * 0.05f) +
           (cfg_.anomaly_weight * anomaly_cost) +
           directional_cost;
}

auto XonStyleTransfer::step_batch(int count) -> float {
    static thread_local std::mt19937_64 rng(cfg_.seed + state_.iterations_done);
    std::uniform_int_distribution<int> dist_x(0, state_.width - 1);
    std::uniform_int_distribution<int> dist_y(0, state_.height - 1);
    std::uniform_real_distribution<float> dist_jitter(-12.0f, 12.0f);
    std::uniform_real_distribution<float> dist_prob(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> dist_pal(0, state_.style_palette.size() - 1);

    for (int step = 0; step < count; ++step) {
        const int x = dist_x(rng);
        const int y = dist_y(rng);
        const int idx = (y * state_.width + x) * 3;

        auto old_lab = rgb_to_lab(state_.current_canvas.rgb[idx],
                                  state_.current_canvas.rgb[idx + 1],
                                  state_.current_canvas.rgb[idx + 2]);
        const float old_loss = compute_loss_at(x, y, old_lab);

        PixelLab cand = old_lab;

        // Jitter mechanism: stochastic color adjustments or palette flips
        if (dist_prob(rng) < cfg_.jitter_prob) {
            // Style Palette Bit Flip
            cand = state_.style_palette[dist_pal(rng)];
            cand.L = std::clamp(cand.L + dist_jitter(rng) * 0.5f, 0.0f, 100.0f);
        } else {
            // Color jitter perturbation
            cand.L = std::clamp(cand.L + dist_jitter(rng), 0.0f, 100.0f);
            cand.a = std::clamp(cand.a + dist_jitter(rng), -128.0f, 127.0f);
            cand.b = std::clamp(cand.b + dist_jitter(rng), -128.0f, 127.0f);
        }

        const float new_loss = compute_loss_at(x, y, cand);

        // Core XN principle: If the bit is flipped / jittered and it looks better/closer, USE IT!
        if (new_loss < old_loss) {
            lab_to_rgb(cand,
                       state_.current_canvas.rgb[idx],
                       state_.current_canvas.rgb[idx + 1],
                       state_.current_canvas.rgb[idx + 2]);
            state_.flips_accepted++;
            state_.current_loss += (new_loss - old_loss);
        }

        state_.iterations_done++;
    }

    return state_.current_loss;
}

} // namespace xn::style
