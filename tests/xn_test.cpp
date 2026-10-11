// ═══════════════════════════════════════════════════════════════════════════
//  xn_test ─ the cortex under the microscope.  Each probe checks the asm
//  engine against a plain C++ reference of the same idea.
//
//    build/xn_test            run every probe
//    build/xn_test parser     run probes whose name contains "parser"
//    XN_MODELS=dir            where the shipped .xn files live (default models)
// ═══════════════════════════════════════════════════════════════════════════
#include <xn.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <print>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using namespace std::literals;

namespace {

struct probe { std::string_view name; std::function<void()> body; };
auto registry() -> std::vector<probe>& { static std::vector<probe> r; return r; }
struct enlist { enlist(std::string_view n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };

int failures = 0;

void check(bool ok, std::string_view what, std::source_location at = std::source_location::current()) {
    if (ok) return;
    ++failures;
    std::println("      ✗ {}:{}  {}", at.file_name(), at.line(), what);
}

#define PROBE(id, label) \
    void probe_##id(); \
    const enlist enlist_##id{label, probe_##id}; \
    void probe_##id()
#define CHECK(expr) check(static_cast<bool>(expr), #expr)

// ── reference cortex, written the obvious way ───────────────────────────────
auto ref_cell(std::uint64_t x, std::uint64_t w, unsigned fan, unsigned t) -> bool {
    unsigned c = 0;
    for (unsigned i = 0; i < fan; ++i) c += ((x >> i) & 1) != ((w >> i) & 1);
    return c >= t;
}

auto ref_forward(const xn_model& m, std::uint64_t x) -> bool {
    for (unsigned l = 0; l < m.nlayers; ++l) {
        const auto& L = m.layers[l];
        std::uint64_t out = 0;
        for (unsigned n = 0; n < L.neurons; ++n)
            if (ref_cell(x, L.w[n], L.fan_in, L.t[n])) out |= 1ull << n;
        x = out;
    }
    return x & 1;
}

auto ref_target(unsigned kind, unsigned inputs) -> std::uint64_t {
    std::uint64_t word = 0;
    const unsigned rows = 1u << inputs;
    for (unsigned x = 0; x < rows; ++x) {
        const bool odd = std::popcount(x) & 1, all = x == rows - 1, any = x != 0;
        bool y = false;
        switch (kind) {
            case XN_T_XOR:  y = odd;  break;  case XN_T_XNOR: y = !odd; break;
            case XN_T_AND:  y = all;  break;  case XN_T_NAND: y = !all; break;
            case XN_T_OR:   y = any;  break;  case XN_T_NOR:  y = !any; break;
        }
        if (y) word |= 1ull << x;
    }
    return word;
}

auto parse(std::string_view text) { return xn::model::parse(text); }

auto expect_error(std::string_view text, xn::status code, std::uint32_t line,
                  std::source_location at = std::source_location::current()) {
    auto m = parse(text);
    check(!m.has_value(), "model should be rejected", at);
    if (m) return;
    check(m.error().code == code, std::format("status {} (\"{}\"), wanted {}", int(m.error().code),
                                              m.error().what(), int(code)), at);
    check(m.error().line == line, std::format("line {}, wanted {}", m.error().line, line), at);
}

constexpr auto tiny = "[model]\ninputs = 2\n[layer]\nneurons = 2\n[layer]\nneurons = 1\n";

// ── abi ─────────────────────────────────────────────────────────────────────
PROBE(abi, "abi: struct layout matches engine/xn.inc") {
    CHECK(xn_model_size() == sizeof(xn_model));
    CHECK(xn_abi_version() == XN_ABI_VERSION);
    CHECK(offsetof(xn_model, truth) == 16);
    CHECK(offsetof(xn_model, restarts) == 40);
    CHECK(offsetof(xn_model, loss) == 64);
    CHECK(offsetof(xn_model, name) == 80);
    CHECK(offsetof(xn_model, layers) == 112);
    CHECK(offsetof(xn_layer, w) == 16);
    CHECK(offsetof(xn_layer, t) == 528);
    xn_model m; xn_model_init(&m);
    CHECK(std::string_view(reinterpret_cast<const char*>(&m.magic), 4) == "XN01");
    CHECK(m.seed == 0x5EED && m.restarts == 32 && m.epochs == 20000);
    CHECK(std::string_view(m.name) == "unnamed");
}

// ── primitives ──────────────────────────────────────────────────────────────
PROBE(xor_primitives, "xor: parity, hamming, masks") {
    for (std::uint64_t x : {0ull, 1ull, 3ull, 0x8000000000000001ull, ~0ull, 0x123456789ull})
        CHECK(xn_parity(x) == unsigned(std::popcount(x) & 1));
    CHECK(xn_hamming(0b1010, 0b0110) == 2);
    CHECK(xn_fanmask(0) == 0 && xn_fanmask(3) == 7 && xn_fanmask(64) == ~0ull);
    CHECK(xn_rowmask(1) == 0x3 && xn_rowmask(2) == 0xF && xn_rowmask(6) == ~0ull);
}

PROBE(xor_cell, "xor: cell agrees with the reference on every small case") {
    for (unsigned fan = 1; fan <= 4; ++fan)
        for (std::uint64_t x = 0; x < (1u << fan); ++x)
            for (std::uint64_t w = 0; w < (1u << fan); ++w)
                for (unsigned t = 0; t <= fan; ++t)
                    CHECK((xn_cell(x, w, xn_fanmask(fan), t) != 0) == ref_cell(x, w, fan, t));
}

PROBE(xor_targets, "xor: named targets for 1..6 inputs") {
    for (unsigned kind = XN_T_XOR; kind <= XN_T_NOR; ++kind)
        for (unsigned n = 1; n <= XN_MAX_INPUTS; ++n)
            CHECK(xn_target_truth(kind, n) == ref_target(kind, n));
    CHECK(xn_target_truth(XN_T_XOR, 2) == 0b0110);
}

PROBE(xor_rng, "xor: rng is deterministic and bounded") {
    std::uint64_t a = 42, b = 42, z = 0;
    for (int i = 0; i < 100; ++i) CHECK(xn_rng_next(&a) == xn_rng_next(&b));
    CHECK(xn_rng_next(&z) != 0 && z != 0);                  // a dead state is reborn
    for (int i = 0; i < 1000; ++i) CHECK(xn_rng_below(&a, 7) < 7);
}

// ── layers and inference ────────────────────────────────────────────────────
PROBE(inference_fuzz, "inference: random organisms agree with the reference") {
    std::uint64_t rng = 0xF00D;
    for (int trial = 0; trial < 300; ++trial) {
        xn_model m; xn_model_init(&m);
        m.inputs = 1 + xn_rng_below(&rng, XN_MAX_INPUTS);
        m.nlayers = 1 + xn_rng_below(&rng, 4);
        unsigned fan = m.inputs;
        for (unsigned l = 0; l < m.nlayers; ++l) {
            auto& L = m.layers[l];
            L.neurons = l + 1 == m.nlayers ? 1 : 1 + xn_rng_below(&rng, XN_MAX_NEURONS);
            L.fan_in = fan;
            for (unsigned n = 0; n < L.neurons; ++n) {
                L.w[n] = xn_rng_next(&rng) & xn_fanmask(fan);
                L.t[n] = static_cast<std::uint8_t>(xn_rng_below(&rng, fan + 1));
            }
            fan = L.neurons;
        }
        for (std::uint64_t x = 0; x < (1u << m.inputs); ++x)
            CHECK((xn_forward(&m, x) != 0) == ref_forward(m, x));
    }
}

// ── parser ──────────────────────────────────────────────────────────────────
PROBE(parser_minimal, "parser: minimal model wires fan-in and defaults to xor") {
    auto m = parse(tiny);
    CHECK(m.has_value());
    if (!m) return;
    CHECK(m->inputs() == 2 && m->layers().size() == 2);
    CHECK(m->layers()[0].fan_in == 2 && m->layers()[1].fan_in == 2);
    CHECK(m->raw().target == XN_T_XOR && m->target() == 0b0110);
    CHECK(!m->carries_weights());
}

PROBE(parser_syntax, "parser: comments, numerals, whitespace, CRLF") {
    auto m = parse("# header\r\n; another\r\n[model]   # trailing\r\n  name = deep  space  # kept inner spaces\r\n"
                   "inputs=0b11\r\ntruth = 0x9_6\r\n\r\n[layer]\r\n\tneurons = 4\r\n[layer]\r\nneurons=1\r\n"
                   "[options]\r\nseed = 1_000\r\nepochs = 0x10\r\nrestarts = 3\r\nverbose = 0\r\n");
    CHECK(m.has_value());
    if (!m) { std::println("      {} @{}", m.error().what(), m.error().line); return; }
    CHECK(m->name() == "deep  space");
    CHECK(m->inputs() == 3 && m->target() == 0x96 && m->raw().target == XN_T_TRUTH);
    CHECK(m->raw().seed == 1000 && m->raw().epochs == 16 && m->raw().restarts == 3 && m->raw().verbose == 0);
}

PROBE(parser_targets, "parser: every named target") {
    for (auto [word, kind] : {std::pair{"xor", XN_T_XOR}, {"xnor", XN_T_XNOR}, {"and", XN_T_AND},
                              {"or", XN_T_OR}, {"nand", XN_T_NAND}, {"nor", XN_T_NOR}}) {
        auto m = parse(std::format("[model]\ninputs=3\ntarget={}\n[layer]\nneurons=1\n", word));
        CHECK(m && m->target() == ref_target(kind, 3));
    }
}

PROBE(parser_errors, "parser: rejections carry the right status and line") {
    using enum xn::status;
    expect_error("", inputs, 0);
    expect_error("inputs = 2\n", section, 1);
    expect_error("[model]\ninputs = 2\n[weird]\n", section, 3);
    expect_error("[model]\ninputs 2\n", syntax, 2);
    expect_error("[model\ninputs = 2\n", syntax, 1);
    expect_error("[model] junk\n", syntax, 1);
    expect_error("[model]\ncolour = red\n", key, 2);
    expect_error("[model]\ninputs = two\n", value, 2);
    expect_error("[model]\ninputs =\n", value, 2);
    expect_error("[model]\ninputs = 7\n", inputs, 2);
    expect_error("[model]\ninputs = 2\ntarget = maybe\n", target, 3);
    expect_error("[model]\ninputs = 2\n", no_layers, 0);
    expect_error("[model]\ninputs = 2\n[layer]\nneurons = 65\n", limit, 4);
    expect_error("[model]\ninputs = 2\n[layer]\nneurons = 2\n", output, 0);
    expect_error("[model]\ninputs = 2\n[layer]\n", limit, 0);
    expect_error("[model]\ninputs=2\n[layer]\nneurons=1\n[options]\nepochs=0\n", limit, 6);
    expect_error("[model]\ninputs=2\n[layer]\nneurons=1\n[options]\nseed=0xZZ\n", value, 6);
    expect_error("[model]\ninputs=2\n[layer]\nneurons=1\nweights=1,2\nthresholds=1\n", weights, 0);
    expect_error("[model]\ninputs=2\n[layer]\nneurons=1\nweights=1\nthresholds=9\n", value, 0);
    expect_error("[model]\ninputs=2\n[layer]\nneurons=1\nthresholds=99\n", value, 5);
    std::string deep = "[model]\ninputs=2\n";
    for (int i = 0; i < 9; ++i) deep += "[layer]\nneurons=1\n";
    expect_error(deep, limit, 19);
}

PROBE(parser_weights, "parser: a hand-wired cortex is born trained") {
    auto m = parse("[model]\ninputs=2\n[layer]\nneurons=2\nweights=0b00, 0b00\nthresholds=1,2\n"
                   "[layer]\nneurons=1\nweights=0b10\nthresholds=2\n");
    CHECK(m.has_value());
    if (!m) return;
    CHECK(m->carries_weights());
    CHECK(m->loss() == 0 && m->truth() == 0b0110);
    CHECK(!m->infer(0b00) && m->infer(0b01) && m->infer(0b10) && !m->infer(0b11));
}

PROBE(options_set, "options: shared setter for files and command lines") {
    xn::model m;
    CHECK(m.set("seed", "0xBEEF").has_value() && m.raw().seed == 0xBEEF);
    CHECK(m.set("restarts", "9").has_value() && m.raw().restarts == 9);
    CHECK(m.set("color", "1").error().code == xn::status::key);
    CHECK(m.set("epochs", "0").error().code == xn::status::limit);
    CHECK(m.set("verbose", "x").error().code == xn::status::value);
}

// ── learning ────────────────────────────────────────────────────────────────
PROBE(train_gates, "train: every 2-input gate makes contact") {
    for (auto word : {"xor", "xnor", "and", "or", "nand", "nor"}) {
        auto m = parse(std::format("[model]\ninputs=2\ntarget={}\n[layer]\nneurons=2\n[layer]\nneurons=1\n", word));
        CHECK(m.has_value());
        if (!m) continue;
        CHECK(m->train() == 0);
        CHECK(m->loss() == 0 && m->truth() == m->target());
        CHECK(m->raw().flags & XN_F_TRAINED);
    }
}

PROBE(train_parity, "train: parity of 3 and 4 bits") {
    for (unsigned n : {3u, 4u}) {
        auto m = parse(std::format("[model]\ninputs={}\n[layer]\nneurons={}\n[layer]\nneurons=1\n"
                                   "[options]\nepochs=50000\nrestarts=256\n", n, n + 2));
        CHECK(m && m->train() == 0);
    }
}

PROBE(train_deterministic, "train: same seed, same organism; the result agrees with the reference") {
    auto a = parse(tiny), b = parse(tiny);
    CHECK(a && b);
    if (!a || !b) return;
    a->train(); b->train();
    CHECK(std::memcmp(a->raw().layers, b->raw().layers, sizeof a->raw().layers) == 0);
    CHECK(a->raw().steps == b->raw().steps);
    for (std::uint64_t x = 0; x < a->rows(); ++x) CHECK(a->infer(x) == ref_forward(a->raw(), x));
}

PROBE(train_impossible, "train: a single cell cannot say xor, and admits it") {
    auto m = parse("[model]\ninputs=2\n[layer]\nneurons=1\n[options]\nepochs=500\nrestarts=4\n");
    CHECK(m.has_value());
    if (!m) return;
    CHECK(m->train() > 0);
    CHECK(m->raw().used >= 1 && m->raw().used <= 4);
}

// ── axn: agents ─────────────────────────────────────────────────────────────
PROBE(agents_abi, "agents: swarm ledger layout and footprints") {
    CHECK(xn_swarm_size() == sizeof(xn_swarm));
    CHECK(offsetof(xn_swarm, best_id) == 32 && offsetof(xn_swarm, ns) == 72);
    CHECK(xn_cpus() >= 1);
    CHECK(xn_agent_footprint() > sizeof(xn_model) && xn_agent_footprint() < xn_swarm_stack());
    const auto t0 = xn_clock_ns();
    CHECK(xn_clock_ns() >= t0);
}

PROBE(agents_seed, "agents: an agent is a pure function of (seed, id)") {
    CHECK(xn_agent_seed(1, 0) == xn_agent_seed(1, 0));
    CHECK(xn_agent_seed(1, 0) != xn_agent_seed(1, 1));
    CHECK(xn_agent_seed(1, 0) != xn_agent_seed(2, 0));
}

PROBE(agents_threads, "agents: the champion does not depend on the thread count") {
    auto base = parse("[model]\ninputs=3\n[layer]\nneurons=3\n[layer]\nneurons=1\n[options]\nepochs=2000\n");
    CHECK(base.has_value());
    if (!base) return;
    auto one = *base;
    const auto ref = one.swarm(3000, 1);
    CHECK(ref.threads == 1);
    for (unsigned t : {2u, 5u, 0u, 256u}) {
        auto m = *base;
        const auto s = m.swarm(3000, t);
        check(s.best_id == ref.best_id && s.best_loss == ref.best_loss && s.best_steps == ref.best_steps,
              std::format("champion moved with {} threads", t));
        check(s.contact == ref.contact && s.steps == ref.steps, std::format("ledger moved with {} threads", t));
        check(std::memcmp(m.raw().layers, one.raw().layers, sizeof one.raw().layers) == 0,
              std::format("weights moved with {} threads", t));
    }
}

PROBE(agents_relive, "agents: the champion re-lives from nothing but its number") {
    auto m = parse(tiny);
    CHECK(m.has_value());
    if (!m) return;
    auto lone = *m;
    const auto base_seed = m->raw().seed;
    const auto s = m->swarm(500, 4);
    CHECK(s.best_loss == 0 && s.best_loss == m->loss());
    CHECK(m->raw().seed == base_seed && m->raw().restarts == 32);    // options restored
    lone.raw().seed = xn_agent_seed(base_seed, s.best_id);
    lone.raw().restarts = 1;
    CHECK(lone.train() == s.best_loss);
    CHECK(lone.raw().steps == s.best_steps);
    CHECK(std::memcmp(lone.raw().layers, m->raw().layers, sizeof lone.raw().layers) == 0);
}

PROBE(agents_edges, "agents: zero agents means one, threads never outnumber agents") {
    auto m = parse(tiny);
    CHECK(m.has_value());
    if (!m) return;
    const auto s = m->swarm(0, 8);
    CHECK(s.agents == 1 && s.threads == 1 && s.best_id == 0);
    const auto t = m->swarm(3, 64);
    CHECK(t.threads == 3);
}

PROBE(agents_hard, "agents: a swarm cracks 4-bit parity with one short life each") {
    auto m = parse("[model]\ninputs=4\n[layer]\nneurons=6\n[layer]\nneurons=1\n[options]\nepochs=20000\n");
    CHECK(m.has_value());
    if (!m) return;
    const auto s = m->swarm(2000);
    CHECK(s.best_loss == 0 && m->loss() == 0 && s.contact > 0);
}

// ── transmission ────────────────────────────────────────────────────────────
PROBE(emit_roundtrip, "emit: trained organism survives the round trip") {
    auto m = parse("[model]\nname=round-trip\ninputs=3\n[layer]\nneurons=4\n[layer]\nneurons=1\n"
                   "[options]\nepochs=50000\nrestarts=64\n");
    CHECK(m.has_value());
    if (!m) return;
    CHECK(m->train() == 0);

    char path[] = "/tmp/xn_test_XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(m->emit(fd) == XN_OK);
    close(fd);

    auto back = xn::model::load(path);
    std::filesystem::remove(path);
    CHECK(back.has_value());
    if (!back) { std::println("      {} @{}", back.error().what(), back.error().line); return; }
    CHECK(back->carries_weights());
    CHECK(back->name() == "round-trip");
    CHECK(back->truth() == m->truth() && back->loss() == 0);
    for (std::size_t l = 0; l < m->layers().size(); ++l)
        for (unsigned n = 0; n < m->layers()[l].neurons; ++n) {
            CHECK(back->layers()[l].w[n] == m->layers()[l].w[n]);
            CHECK(back->layers()[l].t[n] == m->layers()[l].t[n]);
        }
}

PROBE(load_missing, "load: a missing file is an io error") {
    auto m = xn::model::load("/nonexistent/void.xn");
    CHECK(!m && m.error().code == xn::status::io && m.error().line == 0);
}

PROBE(load_shipped, "load: every shipped model parses and makes contact") {
    const std::filesystem::path dir = std::getenv("XN_MODELS") ? std::getenv("XN_MODELS") : "models";
    int seen = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() != ".xn") continue;
        ++seen;
        auto m = xn::model::load(entry.path());
        check(m.has_value(), entry.path().string());
        if (!m) continue;
        if (!m->carries_weights()) m->train();
        check(m->loss() == 0, std::format("{} loss {}", entry.path().string(), m->loss()));
    }
    CHECK(seen >= 4);
}

}  // namespace

auto main(int argc, char** argv) -> int {
    const std::string_view filter = argc > 1 ? argv[1] : "";
    std::println("⊕ xn_test · {}", xn::version());
    int ran = 0;
    for (const auto& [name, body] : registry()) {
        if (!name.contains(filter)) continue;
        const int before = failures;
        body();
        ++ran;
        std::println("  {} {}", failures == before ? "✓" : "✗", name);
    }
    if (ran == 0) { std::println("  no probe matches '{}'", filter); return 2; }
    std::println("\n  {} probes · {}", ran, failures ? std::format("{} failed checks", failures) : "all clear"s);
    return failures ? 1 : 0;
}
