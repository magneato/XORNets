// ═══════════════════════════════════════════════════════════════════════════
//  xn.hpp ─ C++26 face of the cortex.  Header-only, value semantics, no heap
//  inside the engine.  Errors travel as std::expected<_, xn::error>.
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include "xn.h"

#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace xn {

enum class status : int {
    ok = XN_OK, io = XN_EIO, syntax = XN_ESYNTAX, section = XN_ESECTION,
    key = XN_EKEY, value = XN_EVALUE, limit = XN_ELIMIT, no_layers = XN_ENOLAYERS,
    output = XN_EOUTPUT, target = XN_ETARGET, inputs = XN_EINPUTS,
    weights = XN_EWEIGHTS, arg = XN_EARG,
};

struct error {
    xn::status    code;
    std::uint32_t line = 0;                         // 0: the model as a whole
    [[nodiscard]] auto what() const -> std::string_view { return xn_strerror(static_cast<int>(code)); }
};

using layer = xn_layer;

class model {
public:
    model() { xn_model_init(&raw_); }

    [[nodiscard]] static auto load(const std::filesystem::path& path) -> std::expected<model, error> {
        model m;
        return m.adopt(xn_load(&m.raw_, path.c_str()), std::move(m));
    }

    [[nodiscard]] static auto parse(std::string_view text) -> std::expected<model, error> {
        model m;
        return m.adopt(xn_parse(&m.raw_, text.data(), text.size()), std::move(m));
    }

    auto set(std::string_view key, std::string_view value) -> std::expected<void, error> {
        if (int s = xn_option_set(&raw_, key.data(), key.size(), value.data(), value.size()); s != XN_OK)
            return std::unexpected(error{static_cast<xn::status>(s)});
        return {};
    }

    auto train() -> std::uint64_t { return xn_train(&raw_); }

    // axn: spawn a swarm of one-life agents, keep the champion in this model
    auto swarm(std::uint64_t agents, unsigned threads = 0) -> xn_swarm {
        xn_swarm s{};
        s.agents = agents;
        s.threads = threads;
        xn_swarm_run(&raw_, &s);
        return s;
    }

    [[nodiscard]] auto infer(std::uint64_t row) const -> bool { return xn_forward(&raw_, row) != 0; }
    [[nodiscard]] auto truth()   const -> std::uint64_t { return xn_truth(&raw_); }
    [[nodiscard]] auto target()  const -> std::uint64_t { return raw_.truth; }
    [[nodiscard]] auto loss()    const -> std::uint64_t { return xn_loss(&raw_); }
    [[nodiscard]] auto inputs()  const -> unsigned      { return raw_.inputs; }
    [[nodiscard]] auto rows()    const -> unsigned      { return 1u << raw_.inputs; }
    [[nodiscard]] auto name()    const -> std::string_view { return raw_.name; }
    [[nodiscard]] auto carries_weights() const -> bool  { return raw_.flags & XN_F_WEIGHTS; }
    [[nodiscard]] auto layers() const -> std::span<const layer> { return {raw_.layers, raw_.nlayers}; }

    [[nodiscard]] auto raw()       -> xn_model&       { return raw_; }
    [[nodiscard]] auto raw() const -> const xn_model& { return raw_; }

    auto emit(int fd) const -> int { return xn_emit(&raw_, fd); }

    // a model is a value: copy it to fork an organism, never alias the C struct
    auto operator&() const = delete("take raw() if you need the C struct");

private:
    auto adopt(int s, model&& m) const -> std::expected<model, error> {
        if (s != XN_OK) return std::unexpected(error{static_cast<xn::status>(s), m.raw_.err_line});
        return std::move(m);
    }

    xn_model raw_;
};

// truth word → "0110…", row 0 first: the order the console speaks in
[[nodiscard]] inline auto reply(std::uint64_t word, unsigned rows) -> std::string {
    std::string s(rows, '0');
    for (unsigned x = 0; x < rows; ++x) s[x] = static_cast<char>('0' + ((word >> x) & 1));
    return s;
}

// row → "msb…lsb", the spelling --infer accepts
[[nodiscard]] inline auto row_bits(std::uint64_t row, unsigned inputs) -> std::string {
    std::string s(inputs, '0');
    for (unsigned i = 0; i < inputs; ++i) s[inputs - 1 - i] = static_cast<char>('0' + ((row >> i) & 1));
    return s;
}

[[nodiscard]] inline auto version() -> std::string_view { return xn_version(); }

namespace mnist {

struct stack {
    xn_mnist_stack raw{};

    [[nodiscard]] static auto from_image(std::span<const std::uint8_t, 784> img) -> stack {
        stack s;
        xn_mnist_preprocess(img.data(), &s.raw);
        return s;
    }

    [[nodiscard]] static auto from_image(const std::uint8_t *img784) -> stack {
        stack s;
        xn_mnist_preprocess(img784, &s.raw);
        return s;
    }

    [[nodiscard]] auto infer(std::span<std::int32_t, 10> out_logits) const -> int {
        return xn_mnist_infer(&raw, out_logits.data());
    }

    [[nodiscard]] auto infer() const -> int {
        return xn_mnist_infer(&raw, nullptr);
    }
};

[[nodiscard]] inline auto infer_image(const std::uint8_t *img784, std::int32_t *logits = nullptr) -> int {
    return xn_mnist_infer_image(img784, logits);
}

[[nodiscard]] inline auto test_count() -> std::uint64_t {
    return xn_mnist_test_count();
}

inline auto test_sample(std::uint64_t idx, std::uint8_t *out_label = nullptr) -> const std::uint8_t* {
    return xn_mnist_test_sample(idx, out_label);
}

}  // namespace mnist

}  // namespace xn

