// ═══════════════════════════════════════════════════════════════════════════
//  xnc ─ the instrument bench.  C++26 on the outside, the asm cortex inside.
//
//    xnc inspect FILE.xn                 anatomy, without learning anything
//    xnc table   FILE.xn                 train if needed, print the truth table
//    xnc sweep   FILE.xn [--seeds N]     train under N seeds, report the odds
//    xnc swarm   FILE.xn [--agents N]    axn: N one-life agents, keep the champion
//    xnc forge   --inputs N [--target T | --truth W] [--hidden 4,2]
//                [--name S] [--seed S]   write a fresh .xn to stdout
//    xnc version
// ═══════════════════════════════════════════════════════════════════════════
#include <xn.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace std::literals;
namespace rv = std::views;

constexpr auto glyph_on  = "▮"sv;
constexpr auto glyph_off = "▯"sv;

auto usage() -> int {
    std::print(stderr,
        "xnc ─ instrument bench for (x)or (n)ets\n\n"
        "  xnc inspect FILE.xn\n"
        "  xnc table   FILE.xn\n"
        "  xnc sweep   FILE.xn [--seeds N]\n"
        "  xnc swarm   FILE.xn [--agents N] [--threads T]\n"
        "  xnc forge   --inputs N [--target xor|xnor|and|or|nand|nor | --truth W]\n"
        "              [--hidden 4,2] [--name S] [--seed S]\n"
        "  xnc mnist   [--sample N] [--bench] [--stack]\n"
        "  xnc version\n");
    return 2;
}


template <class... Parts>
[[noreturn]] void die(std::format_string<Parts...> fmt, Parts&&... parts) {
    std::print(stderr, "xnc: ");
    std::println(stderr, fmt, std::forward<Parts>(parts)...);
    std::exit(2);
}

auto parse_number(std::string_view s) -> std::optional<std::uint64_t> {
    int base = 10;
    if (s.starts_with("0x") || s.starts_with("0X")) { base = 16; s.remove_prefix(2); }
    else if (s.starts_with("0b") || s.starts_with("0B")) { base = 2; s.remove_prefix(2); }
    std::uint64_t v{};
    auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), v, base);
    if (ec != std::errc{} || end != s.data() + s.size() || s.empty()) return std::nullopt;
    return v;
}

auto load_or_die(std::string_view path) -> xn::model {
    auto m = xn::model::load(std::string(path));
    if (!m) {
        if (m.error().line) die("{}:{}: {}", path, m.error().line, m.error().what());
        die("{}: {}", path, m.error().what());
    }
    return *std::move(m);
}

auto glyphs(std::uint64_t w, unsigned n) -> std::string {
    std::string s;
    for (unsigned i = n; i-- > 0;) s += (w >> i) & 1 ? glyph_on : glyph_off;
    return s;
}

// ── inspect ─────────────────────────────────────────────────────────────────
auto inspect(std::string_view path) -> int {
    const auto m = load_or_die(path);
    const auto& raw = m.raw();
    static constexpr std::array target_names{"none"sv, "xor"sv, "xnor"sv, "and"sv,
                                             "or"sv, "nand"sv, "nor"sv, "truth"sv};

    std::println("⊕ {}", m.name());
    std::println("  file       {}", path);
    std::println("  inputs     {}  ({} rows)", m.inputs(), m.rows());
    std::println("  target     {}  truth {:#x}  reply {}", target_names.at(raw.target), m.target(),
                 xn::reply(m.target(), m.rows()));
    std::print  ("  shape      {}", m.inputs());
    for (const auto& l : m.layers()) std::print(" → {}", l.neurons);
    std::println("");

    std::size_t bits = 0;
    for (const auto& l : m.layers()) bits += l.neurons * (l.fan_in + 7);  // weight bits + a byte of threshold
    std::println("  genome     {} bits across {} cells", bits,
                 std::ranges::fold_left(m.layers() | rv::transform(&xn::layer::neurons), 0u, std::plus{}));
    std::println("  options    seed {:#x} · epochs {} · restarts {} · verbose {}",
                 raw.seed, raw.epochs, raw.restarts, raw.verbose);
    std::println("  weights    {}", m.carries_weights() ? "carried in the file" : "to be learned");

    if (m.carries_weights()) {
        for (auto [i, l] : m.layers() | rv::enumerate) {
            std::println("\n  layer {} · fan-in {}", i, l.fan_in);
            for (unsigned n = 0; n < l.neurons; ++n)
                std::println("    n{:<3} w {}  t {}", n, glyphs(l.w[n], l.fan_in), l.t[n]);
        }
        std::println("\n  loss       {}/{}", m.loss(), m.rows());
    }
    return 0;
}

// ── table ───────────────────────────────────────────────────────────────────
auto table(std::string_view path) -> int {
    auto m = load_or_die(path);
    if (!m.carries_weights()) m.train();
    std::println("row,{},target,cortex", std::string(m.inputs(), 'x'));
    for (unsigned x = 0; x < m.rows(); ++x)
        std::println("{},{},{},{}", x, xn::row_bits(x, m.inputs()), (m.target() >> x) & 1, int(m.infer(x)));
    return m.loss() == 0 ? 0 : 1;
}

// ── sweep ───────────────────────────────────────────────────────────────────
auto sweep(std::string_view path, std::span<const std::string_view> args) -> int {
    unsigned seeds = 32;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--seeds" && i + 1 < args.size()) {
            auto n = parse_number(args[++i]);
            if (!n || *n == 0 || *n > 100000) die("--seeds wants 1..100000");
            seeds = static_cast<unsigned>(*n);
        } else die("sweep: unexpected '{}'", args[i]);
    }

    const auto base = load_or_die(path);
    std::map<std::uint64_t, unsigned> losses;
    std::uint64_t steps = 0, restarts = 0;
    const auto t0 = std::chrono::steady_clock::now();

    for (unsigned s = 0; s < seeds; ++s) {
        auto m = base;                                  // fork the organism
        m.raw().seed = base.raw().seed + s * 0x9E3779B9ull;
        const auto loss = m.train();
        ++losses[loss];
        steps += m.raw().steps;
        restarts += m.raw().used;
    }

    const auto dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0);
    const unsigned hits = losses.contains(0) ? losses[0] : 0;
    std::println("⊕ sweep {} · {} seeds · {:.1f} ms", base.name(), seeds, dt.count());
    std::println("  contact    {}/{}  ({:.1f}%)", hits, seeds, 100.0 * hits / seeds);
    std::println("  mean       {:.1f} mutations · {:.2f} restarts per seed",
                 double(steps) / seeds, double(restarts) / seeds);
    std::println("  loss histogram");
    const auto peak = std::ranges::max(losses | rv::values);
    for (const auto& [loss, count] : losses) {
        const auto bar = std::max(1u, count * 40 / peak);
        std::string ink;
        for (unsigned _ = 0; _ < bar; ++_) ink += "━";
        std::println("    {:>3} │{} {}", loss, ink, count);
    }
    return hits == seeds ? 0 : 1;
}

// ── swarm ───────────────────────────────────────────────────────────────────
auto swarm(std::string_view path, std::span<const std::string_view> args) -> int {
    std::uint64_t agents = 10000;
    unsigned threads = 0;
    for (std::size_t i = 0; i < args.size(); ++i) {
        auto want = [&] { if (i + 1 >= args.size()) die("missing value for {}", args[i]); return parse_number(args[++i]); };
        if (args[i] == "--agents") { auto n = want(); if (!n || !*n) die("--agents wants a positive number"); agents = *n; }
        else if (args[i] == "--threads") {
            auto n = want();
            if (!n || !*n || *n > XN_MAX_THREADS) die("--threads wants 1..{}", XN_MAX_THREADS);
            threads = static_cast<unsigned>(*n);
        } else die("swarm: unexpected '{}'", args[i]);
    }

    auto m = load_or_die(path);
    const auto s = m.swarm(agents, threads);
    const double secs = s.ns / 1e9;
    std::println("⊕ axn swarm · {} · {} agents on {} threads", m.name(), s.agents, s.threads);
    std::println("  contact    {}  ({:.2f}%)", s.contact, 100.0 * s.contact / s.agents);
    std::println("  champion   agent #{} · loss {} · {} mutations · seed {:#x}",
                 s.best_id, s.best_loss, s.best_steps, xn_agent_seed(m.raw().seed, s.best_id));
    std::println("  pace       {:.3f} s · {:.0f} agents/s · {:.3g} mutations/s",
                 secs, s.agents / secs, s.steps / secs);
    std::println("  memory     {} B stored per agent · {} B live per thread",
                 sizeof(std::uint64_t), xn_agent_footprint());
    std::println("  reply      {}", xn::reply(m.truth(), m.rows()));
    return s.best_loss == 0 ? 0 : 1;
}

// ── forge ───────────────────────────────────────────────────────────────────
auto forge(std::span<const std::string_view> args) -> int {
    std::optional<std::uint64_t> inputs, truth;
    std::string_view target = "xor", name = "forged";
    std::uint64_t seed = 0x5EED;
    std::vector<unsigned> hidden;

    for (std::size_t i = 0; i < args.size(); ++i) {
        auto want = [&](std::string_view flag) {
            if (i + 1 >= args.size()) die("missing value for {}", flag);
            return args[++i];
        };
        const auto a = args[i];
        if (a == "--inputs") inputs = parse_number(want(a));
        else if (a == "--target") target = want(a);
        else if (a == "--truth") { truth = parse_number(want(a)); if (!truth) die("--truth wants a number"); }
        else if (a == "--name") name = want(a);
        else if (a == "--seed") seed = parse_number(want(a)).value_or(seed);
        else if (a == "--hidden") {
            for (auto part : want(a) | rv::split(','))
                if (auto n = parse_number(std::string_view(part))) hidden.push_back(static_cast<unsigned>(*n));
                else die("--hidden wants a list like 4,2");
        } else die("forge: unexpected '{}'", a);
    }
    if (!inputs || *inputs < 1 || *inputs > XN_MAX_INPUTS) die("--inputs wants 1..{}", XN_MAX_INPUTS);
    if (hidden.empty()) hidden.push_back(static_cast<unsigned>(*inputs) + 1);
    if (hidden.size() + 1 > XN_MAX_LAYERS) die("at most {} hidden layers", XN_MAX_LAYERS - 1);

    std::string text = std::format("# ⊕ {}.xn · forged by xnc · {}\n\n[model]\nname    = {}\ninputs  = {}\n",
                                   name, xn::version(), name, *inputs);
    text += truth ? std::format("truth   = {:#x}\n", *truth) : std::format("target  = {}\n", target);
    for (auto n : hidden) text += std::format("\n[layer]\nneurons = {}\n", n);
    text += std::format("\n[layer]\nneurons = 1\n\n[options]\nseed     = {:#x}\nepochs   = {}\nrestarts = {}\n",
                        seed, 2000u << *inputs, 64);

    if (auto check = xn::model::parse(text); !check)            // the cortex has the final word
        die("forged model rejected (line {}): {}", check.error().line, check.error().what());
    std::print("{}", text);
    return 0;
}

// ── mnist ───────────────────────────────────────────────────────────────────
auto mnist_cmd(std::span<const std::string_view> args) -> int {
    std::optional<std::uint64_t> sample_idx{};
    bool bench = false;
    bool show_stack = false;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto a = args[i];
        if (a == "--sample" && i + 1 < args.size()) {
            sample_idx = parse_number(args[++i]);
        } else if (a == "--bench") {
            bench = true;
        } else if (a == "--stack") {
            show_stack = true;
        } else {
            die("mnist: unexpected '{}'", a);
        }
    }

    const auto total = xn::mnist::test_count();
    std::println("⊕ xnc mnist · layer stacking geometry ({}-plank bit tensor)", 7);

    if (sample_idx) {
        if (*sample_idx >= total) die("sample index {} out of range (0..{})", *sample_idx, total - 1);
        std::uint8_t true_lbl = 0;
        const auto* img = xn::mnist::test_sample(*sample_idx, &true_lbl);
        const auto st = xn::mnist::stack::from_image(img);

        std::array<std::int32_t, 10> logits{};
        const int pred = st.infer(logits);

        std::println("  sample {} · true label: {} · predicted: {} {}",
                     *sample_idx, true_lbl, pred, pred == true_lbl ? "✓" : "✗");

        // Display ascii art preview (28x28 rendered at 14x14)
        std::println("\n  input image (28×28):");
        for (int r = 0; r < 28; r += 2) {
            std::string line = "    ";
            for (int c = 0; c < 28; c += 2) {
                int avg = (img[r * 28 + c] + img[r * 28 + c + 1] +
                           img[(r + 1) * 28 + c] + img[(r + 1) * 28 + c + 1]) / 4;
                line += (avg > 128) ? "▮" : (avg > 32 ? "·" : " ");
            }
            std::println("{}", line);
        }

        if (show_stack) {
            std::println("\n  layer stacking geometry (7 planks × 144 bits):");
            const char* plank_names[] = {
                "Plank 0: Bit plane >32  (base silhouette)",
                "Plank 1: Bit plane >96  (core body)",
                "Plank 2: Bit plane >160 (dense mass)",
                "Plank 3: Bit plane >224 (peak ink)",
                "Plank 4: Inter-stack XOR (P0 ⊕ P1: outer halo / penumbra)",
                "Plank 5: Inter-stack XOR (P1 ⊕ P2: mid boundary)",
                "Plank 6: Inter-stack XOR (P2 ⊕ P3: core ridge)"
            };
            for (int p = 0; p < 7; ++p) {
                std::println("    {} [words: {:#018x} {:#018x} {:#06x}]",
                             plank_names[p],
                             st.raw.words[p * 3], st.raw.words[p * 3 + 1], st.raw.words[p * 3 + 2]);
            }
        }

        std::println("\n  output logits:");
        for (int k = 0; k < 10; ++k) {
            std::println("    digit {} : {:>7}  {}", k, logits[k], k == pred ? "◄" : " ");
        }
        return 0;
    }

    // Run test set evaluation
    int correct = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < total; ++i) {
        std::uint8_t true_lbl = 0;
        const auto* img = xn::mnist::test_sample(i, &true_lbl);
        const int pred = xn::mnist::infer_image(img, nullptr);
        if (pred == true_lbl) ++correct;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::println("  accuracy   {}/{} ({:.1f}%)", correct, total, (100.0 * correct) / total);
    std::println("  latency    {:.2f} µs/image (total: {} µs for {} images)",
                 static_cast<double>(elapsed_us) / total, elapsed_us, total);

    if (bench) {
        std::println("\n  benchmark  running 100,000 stacked inferences...");
        const auto tb0 = std::chrono::steady_clock::now();
        constexpr int bench_runs = 100000;
        const auto* img = xn::mnist::test_sample(0, nullptr);
        const auto st = xn::mnist::stack::from_image(img);
        int checksum = 0;
        for (int i = 0; i < bench_runs; ++i) {
            checksum += st.infer();
        }
        const auto tb1 = std::chrono::steady_clock::now();
        const auto bench_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tb1 - tb0).count();
        std::println("  throughput {:.2f} million inferences/sec ({:.1f} ns/infer, checksum {})",
                     (bench_runs * 1e9) / (bench_ns * 1e6),
                     static_cast<double>(bench_ns) / bench_runs, checksum);
    }

    return 0;
}

}  // namespace

auto main(int argc, char** argv) -> int {
    const std::vector<std::string_view> args(argv + 1, argv + argc);
    if (args.empty()) return usage();
    const auto cmd = args[0];
    const auto rest = std::span(args).subspan(1);

    if (cmd == "version") { std::println("xnc · {} · abi {}", xn::version(), xn_abi_version()); return 0; }
    if (cmd == "forge") return forge(rest);
    if (cmd == "mnist") return mnist_cmd(rest);
    if (rest.empty()) return usage();
    if (cmd == "inspect") return inspect(rest[0]);
    if (cmd == "table") return table(rest[0]);
    if (cmd == "sweep") return sweep(rest[0], rest.subspan(1));
    if (cmd == "swarm") return swarm(rest[0], rest.subspan(1));
    return usage();
}

