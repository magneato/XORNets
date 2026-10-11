# XORNets
## AI using XOR

This is the XORnets project that I have been promising for some time. It is a 1-bit instruction stream on a 1-bit processor performing AI using XOR to emulate itself: an XOR logic gate. This is all achieved with fewer resources than this period.

> **Site:** https://magneato.github.io/XORNets/ · **Short version:** [TLDR.md](TLDR.md) (literal, scannable, no fluff)
>
> **!!!!The bigger they are, the harder they fall!!!!**

```
  ⊕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⊕
     x n  ·  (x)or (n)ets  ·  1-bit cortex  ·  0.1.0
  ⊕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⊕
```

**xn** is a tiny 1-bit neural engine. The runtime is pure x86-64 assembly: no libc, a ~21 KB static binary. The instruments are C++26. Every weight is a bit, every activation is a bit, and every neuron is the same XOR cell:

```
y = popcnt((x ⊕ w) & fan) >= t
```

A cell XORs its input vector against its weight vector, counts the disagreements and fires once there are at least `t` of them. Vectors never leave a register. With at most 6 inputs, the whole truth table (64 rows) is a single 64-bit word, and so is the loss: `popcnt(truth ⊕ target)`.

There are no gradients in a 1-bit world, so xn learns by **stochastic bit-flip search**. It mutates one weight bit or threshold, keeps the change if the loss didn't grow (sideways drift is allowed), and undoes it otherwise. When a run stalls, it re-rolls the whole cortex and keeps the best organism ever seen.

## Quick start

```sh
./setup.sh                 # nasm, binutils, a C++26 compiler (GCC ≥ 15 / Clang ≥ 19)
./build.sh                 # → build/xn, build/libxn.a, build/xnc, build/xn_test
./test.sh                  # C++26 probes + console checks
./run.sh                   # pick a model, watch it learn, talk to it
```

```sh
build/xn --help --model models/first-contact.xn    # usage + a description of the model
build/xn --model models/first-contact.xn           # train, show anatomy + truth table
build/xn --model models/first-contact.xn --infer 10
build/xn --model models/parity3.xn --emit > p3.trained.xn   # weights included
build/xn --model p3.trained.xn --prompt            # no training: the file carries the cortex
```

Exit codes: `0` signal acquired (loss 0) · `1` trained but loss > 0 · `2` bad command line · `3` the `.xn` file was rejected.

## axn: agents on xn

You don't steer agents. You spawn them and keep the best:

```sh
build/xn --model models/first-contact.xn --agents 1000000          # every cpu
build/xn --model build/ds.xn --agents 2000 --epochs 60000 --threads 8
build/xnc swarm models/beacon.xn --agents 100000
```

```
  ◇ swarm · 1000000 agents · 12 threads · one life of 4096 mutations each
    contact   974268 agents (97.42%)
    champion  agent #50 · 0 mutations · loss 0 · seed 0x2c58daa06446969a
    pace      10791.01 ms · 92669 agents/s · 1029652671 mutations
```

- **An agent is its number.** Agent *i*'s genome is `splitmix64(model.seed + (i+1)·φ)`. Spawning one costs nothing and nothing is stored. Each agent wakes in a per-thread scratch copy of the model, lives **one life** (one restart, `epochs` mutations of bit-flip search), gets judged, and the next agent reuses the same bytes. A billion agents need the same memory as one: about 9.9 KB of stack per thread.
- **The winner** is the agent with the lowest `(loss, mutations, id)`. That ordering is total, so the champion, the contact count and the emitted weights are **identical for any `--threads`**. The champion is then re-lived from its seed into the model, so `--emit`, `--infer` and `--prompt` work as usual. `--agents` always trains from scratch, even if the file carries weights.
- **Threads** are raw kernel threads: `clone(CLONE_VM|CLONE_THREAD|…)` onto an mmap'd stack. Agents are handed out 64 at a time with `lock xadd`, and threads are joined with `futex` on the tid word the kernel clears. The calling thread joins the swarm too.
- **Why a swarm helps.** A single 6-bit-parity run averages ~2.5M mutations. In a 2000-agent swarm, 5% of the agents make contact, and the champion needed only 1,801 mutations.

`xn --stats` (add `--model` for that model's genome) prints the size of every object in the system:

```
    cell                   9    64-bit weight + 8-bit threshold
    model               4848    8 layers + a 112-byte header · fixed, never allocated
    agent                  8    an agent is its seed; nothing else is kept
    live agent          9888    peak stack while an agent lives, per thread
    swarm                 80    the shared tournament ledger
    1e9 agents             0    stored · streamed, judged, forgotten
```

## The `.xn` format

```ini
# comments start with # or ;
[model]
name    = first-contact
inputs  = 2              # 1..6
target  = xor            # xor xnor and or nand nor   (default xor)
# truth = 0b0110         # …or an explicit truth word: bit x = f(row x)

[layer]                  # layers stack in file order
neurons = 2              # 1..64

[layer]                  # the last layer must hold exactly 1 neuron
neurons    = 1
weights    = 0b10        # optional: one per neuron …
thresholds = 2           # … with thresholds. All layers weighted ⇒ born trained

[options]                # any of these can be overridden on the command line
seed     = 0x5EED
epochs   = 4096          # mutations per restart
restarts = 32
verbose  = 1             # 0 hides the anatomy
```

Numbers may be decimal, `0x` hex or `0b` binary, with `_` as a digit separator. Rows are written most significant bit first (`--infer 10` is row 2). Errors report `file:line`.

## Models

| file | what it says |
|---|---|
| `first-contact.xn` | 2-bit XOR, 2 → 2 → 1, learned |
| `parity3.xn` | 3-bit parity, learned |
| `beacon.xn` | 4-bit parity through two hidden ranks, learned |
| `majority.xn` | 3-input majority vote from an explicit truth word, a single cell |
| `signal.xn` | XOR hand-wired as OR ∧ ¬AND, no training |
| `deep-space.xn` | 6-bit parity hand-wired: 64 rows in one register |

## Layout

```
engine/   x86-64 NASM runtime (System V ABI, Linux syscalls)
  xn.inc        the ABI: struct layout, limits, status codes
  xor.asm       the cell, parity, masks, xorshift64* rng, named targets
  layer.asm     evaluate / randomize / mutate / restore a rank of cells
  inference.asm forward pass, truth word, loss
  train.asm     bit-flip search with restarts (snapshot on the stack: reentrant)
  agents.asm    axn: clone()d worker threads streaming one-life agents, champion kept
  options.asm   [options] + CLI overrides, shared numeral/text scanners
  model.asm     init, .xn parser (mmap'd), validation, emit, strerror
  io.asm        buffered syscall output (the only process-global state)
  xn.asm        the console: _start, argv, report, --prompt
sdk/include/
  xn.h          C ABI mirror of xn.inc (static_asserted)
  xn.hpp        C++26 wrapper: xn::model, std::expected errors
tools/xnc.cpp   inspect · table · sweep · forge
tests/xn_test.cpp  probes the asm against plain C++ reference implementations
```

`WDR1_KERNEL.ASM` is the original 16-bit DOS prototype (MASM syntax) and is not part of the build.

## Instruments (`xnc`)

```sh
build/xnc inspect models/signal.xn            # anatomy, genome size, without training
build/xnc table   models/majority.xn          # CSV truth table
build/xnc sweep   models/beacon.xn --seeds 50 # contact rate, mean mutations, loss histogram
build/xnc swarm   models/beacon.xn --agents 100000 # same idea, as an axn swarm
build/xnc forge   --inputs 3 --target nand --hidden 4,2 --name probe > probe.xn
```
