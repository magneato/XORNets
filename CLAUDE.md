# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

XORNets / **xn**: a tiny 1-bit neural engine. Weights, activations and outputs are all bits, and every neuron is the same XOR cell: `y = popcnt((x ⊕ w) & fan) >= t`. The runtime is x86-64 NASM assembly for Linux with no libc. The tools and tests are C++26. No Python anywhere. Licensed under the Clear BSD License. `WDR1_KERNEL.ASM` is the original 16-bit DOS/MASM prototype. It is not built and is not part of xn.

## Commands

```sh
./setup.sh [--check]     # install/verify nasm, binutils, C++26 compiler (CXX= to override)
./build.sh               # everything → build/
./test.sh                # build + C++ probes + CLI checks (exit 1 on any failure)
./test.sh parser         # build + only C++ probes whose name contains "parser" (skips CLI checks)
build/xn_test train      # run a probe subset without rebuilding
build/xnist              # interactive TUI drawing pad for MNIST (mouse + keyboard + layer stacking)
./run.sh [model.xn]      # interactive: xn --model … --prompt
```

`build.sh` assembles each `engine/*.asm` with `nasm -f elf64 -I engine/`. All modules except `xn.asm` (including `agents.asm` and `mnist.asm`) go into `build/libxn.a`. `build/xn` is `xn.o + libxn.a` linked with `ld -static -nostdlib` (entry `_start`). `xnc`, `xnist`, and `xn_test` are compiled with `g++ -std=c++26 -Isdk/include … build/libxn.a`.


## Architecture

- **One ABI, defined twice.** `engine/xn.inc` (NASM `struc`) and `sdk/include/xn.h` (C structs) describe the same `xn_model` (4848 bytes), `xn_layer` (592 bytes) and `xn_swarm` (80 bytes). Any layout change must be made in both. `xn.h` static_asserts the sizes, and the `abi` probe checks the offsets. Status codes, target kinds and flags are mirrored too, and so are the error strings in `model.asm` (indexed by status).
- **No allocation.** A model is a fixed struct: ≤8 layers, ≤64 neurons per layer, ≤6 inputs. A layer's input/output vectors are single 64-bit registers. Because inputs ≤ 6, the full truth table (2^n rows) is one `uint64_t`: `model.truth` is the target, `xn_truth()` is the model's output word, and loss is `popcnt` of their XOR.
- **Data flow.** `xn_load` mmaps the file → `xn_parse` (model.asm) fills the struct section by section. `[options]` keys are delegated to `xn_option_set` (options.asm), which the CLI also uses for `--seed/--epochs/--restarts/--verbose`. `finalize` then wires `fan_in` (layer i's fan-in = layer i−1's neurons), validates, computes the target truth word, and sets `XN_F_WEIGHTS` if every layer listed weights and thresholds.
- **Training** (train.asm) is stochastic bit-flip hill climbing with restarts, not backprop. `xn_layer_mutate` returns an undo record in `rax:rdx`, and `xn_layer_restore` reverts it. The best-layers snapshot lives on the stack, so training is reentrant. Results are deterministic for a given `seed`, and tests rely on that.
- **Agents (axn, agents.asm).** `xn_swarm_run(model, swarm)` streams N one-life agents across raw `clone` threads (mmap'd 128 KiB stacks, `lock xadd` chunks of 64, join via futex on the kernel-cleared tid word; the caller also works). An agent is only its id: seed = `xn_agent_seed(model.seed, id)`, restarts forced to 1. The champion is the minimum `(loss, mutations, id)`, so results must stay thread-count invariant; the `agents_threads` probe and `test.sh` enforce this. Afterwards the champion is re-trained from its seed into the caller's model, and the model's `seed`/`restarts` are restored. `xn_swarm` (80 bytes) is the third mirrored struct. Anything a worker calls must stay reentrant and must not touch libc or TLS.
- **Output.** `io.asm` holds the only process-global state: a 64 KB buffer flushed to `xn_io_fd`. `xn_emit` temporarily retargets it to the given fd. `xn.asm` (`_start`) flushes at exit.
- **C++ side.** `xn.hpp` wraps the C ABI as `xn::model` (value type; copying forks an organism) with `std::expected<_, xn::error>`. `tests/xn_test.cpp` checks the asm against naive C++ reference implementations (`ref_cell`, `ref_forward`, `ref_target`). New probes use the `PROBE(id, "label")` macro. The `load_shipped` probe requires every `models/*.xn` to reach loss 0, so keep shipped models fast to train or hand-wired.
- **Layer Stacking Geometry & MNIST.** `engine/mnist.asm` implements multi-layer bit plane stacking for complex computer vision tasks like handwritten digit recognition. Images are transformed into a massive bit geometry tensor consisting of 4 base bit planes (>32, >96, >160, >224) and 3 inter-stack XOR planes ($P_0 \oplus P_1$, $P_1 \oplus P_2$, $P_2 \oplus P_3$), yielding 7 stacked planks (1008 bits total packed in 21 64-bit words). A 1-bit integer accumulation and thresholding hidden rank outputs 10 class logits with zero floating-point computation, achieving 95.3% accuracy on full MNIST test sets. Accessible via C ABI (`xn_mnist_*`), C++26 SDK (`xn::mnist::*`), and CLI (`xnc mnist [--sample N] [--bench] [--stack]`).
- **Redaction & Unredaction Tools (`apps/redact`).**
  - `build/redact`: C++26 image analysis and dataset utility for document processing.
  - `build/unredact`: C++26 visual recovery and inpainting tool for redacted imagery.
- **Style Transfer Tool (`apps/style`).**
  - `build/xn_style`: C++26 XOR-based meta-level style transfer engine with interactive Raylib visualization.


## Assembly conventions

- System V x86-64: exported functions preserve `rbx rbp r12–r15`. Code is RIP-relative (`default rel`) so `libxn.a` links into PIE C++ binaries. Never index a symbol directly (`[sym + rcx*8]`): `lea` the base first. Each file ends with `NOEXECSTACK`.
- Errors are reported as `file:line: message`. On parse errors `model.err_line` is set, and 0 means a whole-model (finalize) error.
- CLI exit codes (tested in `test.sh`): 0 loss 0 · 1 loss > 0 · 2 bad arguments · 3 rejected `.xn`.
