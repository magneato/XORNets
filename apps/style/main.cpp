#include "style.hpp"
#include "raylib.h"

#include <iostream>
#include <print>
#include <chrono>

using namespace xn::style;

auto main(int argc, char** argv) -> int {
    std::filesystem::path content_path = "source.png";
    std::filesystem::path style_path = "starry.png";
    std::filesystem::path output_path = "styled_output.png";
    int iterations = 1000000;
    int res = 512;
    bool headless = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--content" && i + 1 < argc) content_path = argv[++i];
        else if (a == "--style" && i + 1 < argc) style_path = argv[++i];
        else if (a == "-o" && i + 1 < argc) output_path = argv[++i];
        else if (a == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
        else if (a == "--res" && i + 1 < argc) res = std::stoi(argv[++i]);
        else if (a == "--headless") headless = true;
        else if (a == "--help" || a == "-h") {
            std::println("xn_style: XOR-based Meta-level Style Transfer Engine\n\n"
                         "Usage: xn_style [options]\n"
                         "  --content PATH     Content image (default: source.png)\n"
                         "  --style PATH       Style image (default: starry.png)\n"
                         "  -o PATH            Output rendered PNG (default: styled_output.png)\n"
                         "  --iterations N     Number of jitter/flip iterations (default: 1000000)\n"
                         "  --res N            Resolution dimension (default: 512)\n"
                         "  --headless         Run non-interactively without GUI window\n");
            return 0;
        }
    }

    std::println("◈ xn_style: loading content '{}' and style '{}'", content_path.string(), style_path.string());
    auto content_opt = ImageBuffer::load(content_path);
    auto style_opt = ImageBuffer::load(style_path);

    if (!content_opt) {
        std::println(stderr, "xn_style: failed to load content image '{}'", content_path.string());
        return 1;
    }
    if (!style_opt) {
        std::println(stderr, "xn_style: failed to load style image '{}'", style_path.string());
        return 1;
    }

    TransferConfig cfg{
        .width = res,
        .height = res,
        .iterations = iterations,
        .content_weight = 1.0f,
        .style_pattern_weight = 1.8f,
        .edge_weight = 2.5f,
        .anomaly_weight = 1.6f,
        .jitter_prob = 0.35f,
        .seed = 42,
        .interactive = !headless
    };

    XonStyleTransfer transfer(cfg);
    std::println("◈ xn_style: initializing 7-plank bit tensors, Reinhard color moments & anomaly maps...");
    if (!transfer.initialize(*content_opt, *style_opt)) {
        std::println(stderr, "xn_style: initialization failed!");
        return 1;
    }
    std::println("  ✓ meta-level initialization complete (initial loss: {:.2f})", transfer.get_state().current_loss);

    if (headless) {
        std::println("◈ xn_style: running in headless batch mode for {} iterations...", iterations);
        auto t0 = std::chrono::steady_clock::now();
        const int batch_size = 50000;
        while (transfer.get_state().iterations_done < static_cast<std::uint64_t>(iterations)) {
            transfer.step_batch(batch_size);
            std::println("  · [{:7d} / {}] loss: {:.2f} | flips accepted: {}",
                         transfer.get_state().iterations_done, iterations,
                         transfer.get_state().current_loss,
                         transfer.get_state().flips_accepted);
        }
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::println("◈ xn_style: done in {:.2f}s ({:.0f} it/s)", elapsed, iterations / elapsed);
        transfer.get_canvas().save_png(output_path);
        std::println("  ✓ saved styled output to '{}'", output_path.string());
        return 0;
    }

    // Raylib Interactive Visualizer
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT);
    SetTraceLogLevel(LOG_WARNING);
    const int win_w = res * 2 + 60;
    const int win_h = res + 100;
    InitWindow(win_w, win_h, "XORNets Style Transfer — Starry Night [Raylib]");
    SetTargetFPS(60);

    // Prepare Raylib textures
    const auto& canvas = transfer.get_canvas();
    Image rl_canvas_img = {
        .data = const_cast<uint8_t*>(canvas.rgb.data()),
        .width = res,
        .height = res,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8
    };
    Texture2D tex_canvas = LoadTextureFromImage(rl_canvas_img);

    // Style preview texture
    auto style_resized = style_opt->resize(res, res);
    Image rl_style_img = {
        .data = style_resized.rgb.data(),
        .width = res,
        .height = res,
        .mipmaps = 1,
        .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8
    };
    Texture2D tex_style = LoadTextureFromImage(rl_style_img);

    bool paused = false;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_S)) {
            transfer.get_canvas().save_png(output_path);
            std::println("  ✓ manual snapshot saved to '{}'", output_path.string());
        }

        if (!paused && transfer.get_state().iterations_done < static_cast<std::uint64_t>(iterations)) {
            // Run 25,000 jitter-and-flip steps per frame
            transfer.step_batch(25000);
            // Update texture
            UpdateTexture(tex_canvas, transfer.get_canvas().rgb.data());
        }

        BeginDrawing();
        ClearBackground(Color{18, 18, 26, 255});

        // Draw Headers & Stats
        DrawText("XORNets Meta-Level Style Transfer", 20, 15, 20, Color{250, 218, 94, 255});
        DrawText(TextFormat("Iterations: %llu / %d  |  Flips Accepted: %llu  |  Loss: %.1f  %s",
                            transfer.get_state().iterations_done, iterations,
                            transfer.get_state().flips_accepted,
                            transfer.get_state().current_loss,
                            paused ? "[PAUSED - Space to resume]" : "[RUNNING]"),
                 20, 42, 16, Color{130, 200, 250, 255});

        // Draw Images
        DrawTexture(tex_canvas, 20, 75, WHITE);
        DrawRectangleLines(19, 74, res + 2, res + 2, Color{250, 218, 94, 180});
        DrawText("Styled XOR Canvas", 25, 80, 16, Color{255, 255, 255, 220});

        DrawTexture(tex_style, res + 40, 75, WHITE);
        DrawRectangleLines(res + 39, 74, res + 2, res + 2, Color{80, 160, 255, 180});
        DrawText("Target Style (Starry Night)", res + 45, 80, 16, Color{255, 255, 255, 220});

        // Key instructions
        DrawText("[SPACE] Pause/Resume   [S] Save Snapshot   [ESC] Exit",
                 20, win_h - 22, 14, Color{180, 180, 190, 255});

        EndDrawing();
    }

    UnloadTexture(tex_canvas);
    UnloadTexture(tex_style);
    CloseWindow();

    transfer.get_canvas().save_png(output_path);
    std::println("◈ xn_style: final output saved to '{}'", output_path.string());

    return 0;
}
