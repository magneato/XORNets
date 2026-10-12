// ═══════════════════════════════════════════════════════════════════════════
//  xnist ─ Interactive Terminal UI Drawing Pad & Inspector for XN MNIST
//
//  Palette:
//    • Royal Purple : #7852FF / rgb(120, 82, 255)  -> headers, frames, borders
//    • Electric Blue: #00E5FF / rgb(0, 229, 255)   -> active strokes, cursor, highlights
//    • Neon Red     : #FF0055 / rgb(255, 0, 85)    -> highest confidence, alerts, hot keys
//    • Gold Leaf    : #FFD700 / rgb(255, 215, 0)   -> scores, badges, metrics, crown
//
//  Input: Mouse (SGR tracking: click & drag to draw) + Keyboard
//  Engine: XN Layer Stacking Geometry (7-plank bit tensor)
// ═══════════════════════════════════════════════════════════════════════════

#include <xn.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <print>
#include <span>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

namespace {

// ── Color Theme ─────────────────────────────────────────────────────────────
constexpr auto C_RESET       = "\033[0m";
constexpr auto C_PURPLE      = "\033[38;2;120;82;255m";
constexpr auto C_PURPLE_BG   = "\033[48;2;40;20;85m";
constexpr auto C_BLUE        = "\033[38;2;0;229;255m";
constexpr auto C_BLUE_BG     = "\033[48;2;0;50;80m";
constexpr auto C_RED         = "\033[38;2;255;0;85m";
constexpr auto C_RED_BG      = "\033[48;2;80;0;25m";
constexpr auto C_GOLD        = "\033[38;2;255;215;0m";
constexpr auto C_GOLD_BG     = "\033[48;2;70;55;0m";
constexpr auto C_MUTED       = "\033[38;2;130;120;160m";
constexpr auto C_DARK_BG     = "\033[48;2;18;12;32m";
constexpr auto C_CANVAS_BG   = "\033[48;2;10;8;20m";
constexpr auto C_BOLD        = "\033[1m";
constexpr auto C_DIM         = "\033[2m";

// ── Terminal Mode Management ────────────────────────────────────────────────
struct TerminalGuard {
    termios orig_termios{};
    bool active = false;

    TerminalGuard() {
        if (!isatty(STDIN_FILENO)) return;
        tcgetattr(STDIN_FILENO, &orig_termios);
        termios raw = orig_termios;
        raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        raw.c_oflag &= ~(OPOST);
        raw.c_cflag |= (CS8);
        raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1; // 100ms timeout
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

        // Enter alternate screen, hide cursor, enable mouse tracking (SGR 1006 + 1002/1000)
        std::print("\033[?1049h\033[?25l\033[?1000h\033[?1002h\033[?1006h");
        std::fflush(stdout);
        active = true;
    }

    ~TerminalGuard() {
        if (!active) return;
        // Disable mouse, show cursor, leave alternate screen
        std::print("\033[?1006l\033[?1002l\033[?1000l\033[?25h\033[?1049l");
        std::fflush(stdout);
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    }
};

// ── Drawing Canvas ──────────────────────────────────────────────────────────
// 28x28 grayscale canvas [0..255]
struct Canvas {
    std::array<std::uint8_t, 28 * 28> pixels{};

    void clear() {
        pixels.fill(0);
    }

    void stamp(int cx, int cy, int radius = 1, std::uint8_t val = 255) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const int x = cx + dx;
                const int y = cy + dy;
                if (x >= 0 && x < 28 && y >= 0 && y < 28) {
                    const int dist_sq = dx * dx + dy * dy;
                    if (dist_sq <= radius * radius) {
                        const int idx = y * 28 + x;
                        int intensity = val;
                        if (dist_sq > 0) intensity = val * 3 / 4;
                        pixels[idx] = static_cast<std::uint8_t>(std::min(255, pixels[idx] + intensity));
                    }
                }
            }
        }
    }

    void load_sample(std::uint64_t idx, std::uint8_t* out_lbl = nullptr) {
        const auto* img = xn::mnist::test_sample(idx, out_lbl);
        if (img) {
            std::copy_n(img, 784, pixels.begin());
        }
    }
};

// Layout constants (top-left coordinates of components in the terminal)
// Pad is 28x14 characters (each character block is 1x2 pixels: upper/lower half block)
constexpr int PAD_ROW = 4;
constexpr int PAD_COL = 4;
constexpr int STATS_ROW = 4;
constexpr int STATS_COL = 38;
constexpr int STACK_ROW = 4;
constexpr int STACK_COL = 72;

void move_to(int row, int col) {
    std::print("\033[{};{}H", row, col);
}

// ── Rendering the UI ─────────────────────────────────────────────────────────
void render_frame() {
    std::print("\033[2J"); // Clear screen

    // Header bar in Royal Purple with Gold Leaf branding
    move_to(1, 1);
    std::print("{}{}{}  ◈ XNIST ◈  1-Bit Cortex Drawing Pad · Layer Stacking Geometry  {}\033[K",
               C_PURPLE_BG, C_GOLD, C_BOLD, C_RESET);

    move_to(2, 1);
    std::print("{}{}  Palette: Royal Purple ⊕ Electric Blue ⊕ Neon Red ⊕ Gold Leaf{}\033[K",
               C_DIM, C_MUTED, C_RESET);

    // Box 1: Drawing Pad Frame
    move_to(PAD_ROW - 1, PAD_COL - 1);
    std::print("{}{}+----------------------------+{}", C_PURPLE, C_BOLD, C_RESET);
    move_to(PAD_ROW - 1, PAD_COL + 2);
    std::print("{}{} CANVAS (28x28) {}", C_BLUE, C_BOLD, C_PURPLE);

    for (int r = 0; r < 14; ++r) {
        move_to(PAD_ROW + r, PAD_COL - 1);
        std::print("{}|{}", C_PURPLE, C_RESET);
        move_to(PAD_ROW + r, PAD_COL + 28);
        std::print("{}|{}", C_PURPLE, C_RESET);
    }
    move_to(PAD_ROW + 14, PAD_COL - 1);
    std::print("{}{}+----------------------------+{}", C_PURPLE, C_BOLD, C_RESET);

    // Box 2: Prediction & Logits Frame
    move_to(STATS_ROW - 1, STATS_COL - 1);
    std::print("{}{}+------------------------------+{}", C_PURPLE, C_BOLD, C_RESET);
    move_to(STATS_ROW - 1, STATS_COL + 2);
    std::print("{}{} CORTEX LOGITS {}", C_GOLD, C_BOLD, C_PURPLE);

    for (int r = 0; r < 14; ++r) {
        move_to(STATS_ROW + r, STATS_COL - 1);
        std::print("{}|{}", C_PURPLE, C_RESET);
        move_to(STATS_ROW + r, STATS_COL + 30);
        std::print("{}|{}", C_PURPLE, C_RESET);
    }
    move_to(STATS_ROW + 14, STATS_COL - 1);
    std::print("{}{}+------------------------------+{}", C_PURPLE, C_BOLD, C_RESET);

    // Box 3: Layer Stacking Geometry Frame
    move_to(STACK_ROW - 1, STACK_COL - 1);
    std::print("{}{}+-----------------------------------------+{}", C_PURPLE, C_BOLD, C_RESET);
    move_to(STACK_ROW - 1, STACK_COL + 2);
    std::print("{}{} 7-PLANK LAYER STACKING GEOMETRY {}", C_RED, C_BOLD, C_PURPLE);

    for (int r = 0; r < 14; ++r) {
        move_to(STACK_ROW + r, STACK_COL - 1);
        std::print("{}|{}", C_PURPLE, C_RESET);
        move_to(STACK_ROW + r, STACK_COL + 41);
        std::print("{}|{}", C_PURPLE, C_RESET);
    }
    move_to(STACK_ROW + 14, STACK_COL - 1);
    std::print("{}{}+-----------------------------------------+{}", C_PURPLE, C_BOLD, C_RESET);

    // Footer Help Bar
    move_to(20, 2);
    std::print("{}[Mouse]{} Draw/Drag  {}[C]{} Clear  {}[0-9]{} Load Sample  {}[Space]{} Infer  {}[Q/Esc]{} Quit",
               C_BLUE, C_MUTED, C_RED, C_MUTED, C_GOLD, C_MUTED, C_BLUE, C_MUTED, C_RED, C_MUTED);

    move_to(21, 2);
    std::print("{}[Arrows/WASD]{} Move Cursor  {}[Enter/X]{} Dot  {}[N/P]{} Next/Prev Sample  {}[B]{} Benchmark",
               C_PURPLE, C_MUTED, C_BLUE, C_MUTED, C_GOLD, C_MUTED, C_RED, C_MUTED);
}

void render_pad(const Canvas& canvas, int cur_x, int cur_y) {
    // 28 rows rendered as 14 rows of half-blocks
    // Top pixel = row * 2, Bottom pixel = row * 2 + 1
    for (int r = 0; r < 14; ++r) {
        move_to(PAD_ROW + r, PAD_COL);
        for (int c = 0; c < 28; ++c) {
            const int top_val = canvas.pixels[(r * 2) * 28 + c];
            const int bot_val = canvas.pixels[(r * 2 + 1) * 28 + c];

            const bool is_cur_top = (cur_x == c && cur_y == r * 2);
            const bool is_cur_bot = (cur_x == c && cur_y == r * 2 + 1);

            // Cursor highlighted in Electric Blue
            if (is_cur_top || is_cur_bot) {
                std::print("{}{}{}▮{}", C_BLUE_BG, C_BLUE, C_BOLD, C_RESET);
                continue;
            }

            // Both pixels off
            if (top_val == 0 && bot_val == 0) {
                std::print("{}·{}", C_MUTED, C_RESET);
                continue;
            }

            // Pick color according to density
            // Upper half block: \u2580
            if (top_val > 0 && bot_val == 0) {
                const auto col = (top_val > 160) ? C_GOLD : (top_val > 96 ? C_BLUE : C_PURPLE);
                std::print("{}▀{}", col, C_RESET);
            } else if (top_val == 0 && bot_val > 0) {
                const auto col = (bot_val > 160) ? C_GOLD : (bot_val > 96 ? C_BLUE : C_PURPLE);
                std::print("{}▄{}", col, C_RESET);
            } else {
                const auto col = (top_val + bot_val > 250) ? C_GOLD : C_BLUE;
                std::print("{}█{}", col, C_RESET);
            }
        }
    }
}

void render_stats(const Canvas& canvas, int& out_pred) {
    // Perform inference using XN Layer Stacking
    xn::mnist::stack st = xn::mnist::stack::from_image(canvas.pixels.data());
    std::array<std::int32_t, 10> logits{};
    const int pred = st.infer(logits);
    out_pred = pred;

    // Find min and max logits for bar scaling
    int32_t max_logit = logits[0];
    int32_t min_logit = logits[0];
    for (int k = 1; k < 10; ++k) {
        max_logit = std::max(max_logit, logits[k]);
        min_logit = std::min(min_logit, logits[k]);
    }
    const int32_t range = std::max<int32_t>(1, max_logit - min_logit);

    // Crown / Prediction display at top of box
    move_to(STATS_ROW, STATS_COL + 1);
    std::print("  {}PREDICTED DIGIT: {}{}{} {} ",
               C_GOLD, C_RED_BG, C_GOLD, C_BOLD, pred);

    move_to(STATS_ROW + 1, STATS_COL + 1);
    std::print("  {}Confidence Peak: {}{:+7d}{}", C_MUTED, C_GOLD, max_logit, C_RESET);

    // Render 10 score bars
    for (int k = 0; k < 10; ++k) {
        move_to(STATS_ROW + 3 + k, STATS_COL);
        const bool is_winner = (k == pred);
        const int bar_len = std::clamp<int>((logits[k] - min_logit) * 14 / range, 0, 14);

        std::string bar;
        for (int b = 0; b < bar_len; ++b) bar += "▮";
        for (int b = bar_len; b < 14; ++b) bar += " ";

        if (is_winner) {
            std::print("{}{} [{}]: {:>6} {}{}{}",
                       C_RED, C_BOLD, k, logits[k], C_GOLD, bar, C_RESET);
        } else {
            std::print("{}{} [{}]: {:>6} {}{}{}",
                       C_MUTED, C_RESET, k, logits[k], C_PURPLE, bar, C_RESET);
        }
    }

    // Render Layer Stacking Geometry Details
    // 7 Planks: 4 base planes + 3 inter-stack XOR planes
    const char* plank_titles[] = {
        "Plank 0: P0  (>32  Silhouette)",
        "Plank 1: P1  (>96  Core Body)",
        "Plank 2: P2  (>160 Dense Mass)",
        "Plank 3: P3  (>224 Peak Ink)",
        "Plank 4: P0^P1 (Halo Penumbra)",
        "Plank 5: P1^P2 (Mid Boundary)",
        "Plank 6: P2^P3 (Core Ridge)"
    };

    int total_stack_bits = 0;
    for (int p = 0; p < 7; ++p) {
        const uint64_t w0 = st.raw.words[p * 3];
        const uint64_t w1 = st.raw.words[p * 3 + 1];
        const uint64_t w2 = st.raw.words[p * 3 + 2] & 0xFFFF;
        const int pop = std::popcount(w0) + std::popcount(w1) + std::popcount(w2);
        total_stack_bits += pop;

        move_to(STACK_ROW + p * 2, STACK_COL);
        const auto title_color = (p < 4) ? C_BLUE : C_RED;
        std::print("{}{:<30}{} {:>3}/144", title_color, plank_titles[p], C_GOLD, pop);

        move_to(STACK_ROW + p * 2 + 1, STACK_COL + 2);
        std::print("{}{:#018x} {:#018x}{}", C_MUTED, w0, w1, C_RESET);
    }

    move_to(STACK_ROW + 14, STACK_COL + 1);
    std::print("{}Total Active Bits in Geometry: {}{}{}/1008{}",
               C_MUTED, C_GOLD, C_BOLD, total_stack_bits, C_RESET);
}

} // namespace

auto main(int argc, char** argv) -> int {
    // If run non-interactively or with --help, print friendly synopsis
    if (argc > 1 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        std::println(
            "xnist ─ Interactive TUI Drawing Pad for XN MNIST\n\n"
            "Usage: xnist [--sample N]\n\n"
            "Interactive controls:\n"
            "  Mouse:        Click & drag on the canvas to draw\n"
            "  Keyboard:     WASD or Arrow keys to move brush\n"
            "                Enter or X to draw a dot\n"
            "                C to clear canvas\n"
            "                0-9 or N/P to load digit test samples\n"
            "                B to run high-speed benchmark\n"
            "                Q or Esc to exit\n\n"
            "Palette: Royal Purple, Electric Blue, Neon Red, Gold Leaf\n"
            "Powered by XN Layer Stacking Geometry (7-plank 1-bit tensor)\n");
        return 0;
    }

    Canvas canvas;
    std::uint64_t sample_idx = 0;
    uint8_t sample_lbl = 0;
    canvas.load_sample(sample_idx, &sample_lbl);

    int cur_x = 14;
    int cur_y = 14;
    bool dirty = true;
    int current_pred = 0;

    TerminalGuard term;
    render_frame();

    while (true) {
        if (dirty) {
            render_pad(canvas, cur_x, cur_y);
            render_stats(canvas, current_pred);
            std::fflush(stdout);
            dirty = false;
        }

        // Read input
        char c = 0;
        const ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n <= 0) {
            usleep(10000); // 10ms rest
            continue;
        }

        // Handle escape sequences (mouse + arrows)
        if (c == '\033') {
            char seq[64]{};
            ssize_t sn = read(STDIN_FILENO, seq, sizeof(seq) - 1);
            if (sn <= 0) {
                // Bare escape -> quit
                break;
            }
            seq[sn] = '\0';

            // SGR mouse tracking: \033[<btn;col;rowM or m
            if (seq[0] == '[' && seq[1] == '<') {
                int btn = 0, mouse_col = 0, mouse_row = 0;
                char type = 0;
                if (std::sscanf(seq + 2, "%d;%d;%d%c", &btn, &mouse_col, &mouse_row, &type) == 4) {
                    // Check if mouse event is inside the drawing pad
                    // Pad starts at PAD_ROW, PAD_COL
                    // Pad width: 28 cols, height: 14 rows (28 pixel rows)
                    if (mouse_col >= PAD_COL && mouse_col < PAD_COL + 28 &&
                        mouse_row >= PAD_ROW && mouse_row < PAD_ROW + 14) {
                        const int px = mouse_col - PAD_COL;
                        const int py = (mouse_row - PAD_ROW) * 2;
                        cur_x = px;
                        cur_y = py;

                        // Button 0 = left click or drag with left down (btn 32 = drag)
                        if ((btn == 0 || btn == 32) && type == 'M') {
                            canvas.stamp(px, py, 1, 240);
                            dirty = true;
                        }
                    }
                }
                continue;
            }

            // Arrow keys: \033[A, B, C, D
            if (seq[0] == '[') {
                if (seq[1] == 'A') { cur_y = std::max(0, cur_y - 1); dirty = true; } // Up
                else if (seq[1] == 'B') { cur_y = std::min(27, cur_y + 1); dirty = true; } // Down
                else if (seq[1] == 'C') { cur_x = std::min(27, cur_x + 1); dirty = true; } // Right
                else if (seq[1] == 'D') { cur_x = std::max(0, cur_x - 1); dirty = true; } // Left
                continue;
            }
        }

        // Regular keys
        if (c == 'q' || c == 'Q') {
            break;
        } else if (c == 'c' || c == 'C') {
            canvas.clear();
            dirty = true;
        } else if (c >= '0' && c <= '9') {
            // Find sample with this digit
            const int target_digit = c - '0';
            for (std::uint64_t s = 0; s < xn::mnist::test_count(); ++s) {
                uint8_t l = 0;
                xn::mnist::test_sample(s, &l);
                if (l == target_digit) {
                    sample_idx = s;
                    canvas.load_sample(sample_idx, &sample_lbl);
                    dirty = true;
                    break;
                }
            }
        } else if (c == 'n' || c == 'N') {
            sample_idx = (sample_idx + 1) % xn::mnist::test_count();
            canvas.load_sample(sample_idx, &sample_lbl);
            dirty = true;
        } else if (c == 'p' || c == 'P') {
            sample_idx = (sample_idx == 0) ? (xn::mnist::test_count() - 1) : (sample_idx - 1);
            canvas.load_sample(sample_idx, &sample_lbl);
            dirty = true;
        } else if (c == 'w' || c == 'W') {
            cur_y = std::max(0, cur_y - 1);
            dirty = true;
        } else if (c == 's' || c == 'S') {
            cur_y = std::min(27, cur_y + 1);
            dirty = true;
        } else if (c == 'a' || c == 'A') {
            cur_x = std::max(0, cur_x - 1);
            dirty = true;
        } else if (c == 'd' || c == 'D') {
            cur_x = std::min(27, cur_x + 1);
            dirty = true;
        } else if (c == 'x' || c == 'X' || c == '\r' || c == '\n') {
            canvas.stamp(cur_x, cur_y, 1, 240);
            dirty = true;
        } else if (c == ' ') {
            // Re-trigger inference refresh
            dirty = true;
        } else if (c == 'b' || c == 'B') {
            // Mini benchmark
            const auto tb0 = std::chrono::steady_clock::now();
            constexpr int bench_runs = 50000;
            const auto st = xn::mnist::stack::from_image(canvas.pixels.data());
            int sum = 0;
            for (int i = 0; i < bench_runs; ++i) sum += st.infer();
            const auto tb1 = std::chrono::steady_clock::now();
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tb1 - tb0).count();

            move_to(22, 2);
            std::print("{}{}  BENCHMARK: 50,000 stacked inferences in {:.2f} ms ({:.1f} ns/infer)  {}",
                       C_RED_BG, C_GOLD, ns / 1e6, static_cast<double>(ns) / bench_runs, C_RESET);
            std::fflush(stdout);
            sleep(1);
            render_frame();
            dirty = true;
        }
    }

    return 0;
}
