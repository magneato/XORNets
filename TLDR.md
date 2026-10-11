# TL;DR — xn (XORNets)

Short version. Literal language. No metaphors except in the box marked **lore**.

---

## What is it? (3 facts)

1. **A very small neural network program.** Every weight is 1 bit (0 or 1).
2. **Every neuron does the same thing:** XOR its inputs with its weights, count the 1s, and output 1 if the count is big enough.
3. **The engine is written in x86-64 assembly.** The helper tools and tests are C++26. There is no Python.

## What can it learn?

Small yes/no functions with up to 6 inputs. Examples: XOR, AND, OR, parity, majority vote.

It **cannot** do images, text or chat. That's on purpose: it is tiny.

## How does it learn?

- It does **not** use gradients or backpropagation.
- It flips one random bit.
  - The answers got worse → undo the flip.
  - The answers did not get worse → keep it.
- It repeats until every answer is correct, or it runs out of tries.

## What is an "agent"?

- An agent is one attempt at learning, started from one random seed.
- `--agents 1000000` runs one million attempts, spread over all your CPU cores.
- Only the best agent is kept. The rest are thrown away.
- **"Best" means:** fewest wrong answers first, then fewest flips, then lowest agent number.
- The result is the same no matter how many CPU cores you use.

## How do I run it? (copy-paste)

Linux on x86-64 only.

```sh
git clone https://github.com/magneato/XORNets
cd XORNets
./setup.sh      # installs nasm, binutils, g++ if missing
./build.sh      # builds everything into build/
./test.sh       # runs all tests; last line says "all channels clear"
./run.sh        # pick a model by number, then type bits like 01
```

## The 5 commands you need

| Command | What it does |
|---|---|
| `build/xn --help` | Shows all options. |
| `build/xn --model models/first-contact.xn` | Learns XOR and prints the result table. |
| `build/xn --model models/first-contact.xn --infer 10` | Prints the answer for input `10`. |
| `build/xn --model models/parity3.xn --agents 100000` | Runs 100,000 agents and keeps the best. |
| `build/xn --stats` | Prints how many bytes each part uses. |

## How do I read the output?

- `✓` means that row is correct. `✗` means it is wrong.
- `loss 0/4` means 0 wrong answers out of 4. **0 is perfect.**
- `SIGNAL ACQUIRED` means loss is 0.
- `NOISE` means loss is not 0.
- `▮` is a weight bit of 1. `▯` is a weight bit of 0.

## Exit codes (for scripts)

| Code | Meaning |
|---|---|
| 0 | Learned perfectly (loss 0). |
| 1 | Finished, but some answers are wrong. |
| 2 | Bad command-line arguments. |
| 3 | The `.xn` model file has an error. The message gives the line number. |

## The model file (`.xn`)

Plain text. Three kinds of sections:

```ini
[model]
inputs = 2        # how many input bits (1 to 6)
target = xor      # what to learn: xor xnor and or nand nor

[layer]
neurons = 2       # hidden layer size (1 to 64)

[layer]
neurons = 1       # the last layer must have exactly 1 neuron

[options]
seed = 0x5EED     # same seed = same result, every time
```

## Sizes (from `xn --stats`)

| Thing | Bytes |
|---|---|
| The whole `xn` program | about 21,500 |
| One model | 4,848 |
| One neuron | 9 |
| One agent (stored) | 8 (it is just a number) |
| One billion agents (stored) | 0 (they are never stored) |

## Where things are

| Folder | Contents |
|---|---|
| `engine/` | Assembly source code (the actual engine). |
| `sdk/include/` | C and C++ headers for using the engine from your own code. |
| `tools/` | `xnc`, a C++ helper tool. |
| `tests/` | C++ tests. |
| `models/` | Example `.xn` model files. |

---

> **lore** (story only, not technical)
> The project pretends the program is an alien civilization made of one repeated gate. They agreed with each other for so long that they couldn't think, so they needed XOR, the gate of disagreement. That's why the output says things like `SIGNAL ACQUIRED`. It's decoration. The code is ordinary, real, tested software.

More detail: [README.md](README.md) · Website: [index.html](index.html)
