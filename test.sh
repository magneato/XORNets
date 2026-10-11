#!/usr/bin/env bash
# test.sh ─ build, probe the SDK in C++26, then interrogate the xn console.
#
#   ./test.sh              everything
#   ./test.sh parser       only the C++ probes whose name contains "parser"
#                          (the console checks are skipped when filtering)
set -uo pipefail
cd "$(dirname "$0")"

./build.sh || exit 1
echo

XN=build/xn
XNC=build/xnc
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

build/xn_test "${1:-}"
status=$?
[[ -n ${1:-} ]] && exit $status

pass=0 fail=0
ok()  { pass=$((pass + 1)); printf '  \342\234\223 %s\n' "$1"; }
bad() { fail=$((fail + 1)); printf '  \342\234\227 %s\n' "$1"; [[ -n ${2:-} ]] && printf '%s\n' "$2" | sed 's/^/      /'; }

# expect NAME EXIT PATTERN -- command...   (PATTERN: grep -E over stdout+stderr)
expect() {
    local name=$1 want=$2 pattern=$3; shift 4
    local out code
    out=$("$@" 2>&1); code=$?
    if [[ $code -ne $want ]]; then bad "$name (exit $code, wanted $want)" "$out"
    elif [[ -n $pattern ]] && ! grep -qE -- "$pattern" <<<"$out"; then bad "$name (no /$pattern/)" "$out"
    else ok "$name"; fi
}

echo
echo "⊕ console"
expect "help"                    0 "usage +xn --model"            -- $XN --help
expect "help describes a model"  0 "this model.*"                 -- $XN --help --model models/first-contact.xn
expect "help shows the shape"    0 "cortex +2 → 2 → 1"            -- $XN --help --model models/first-contact.xn
expect "no arguments → help"     0 "usage"                        -- $XN
expect "version"                 0 "^xn 0\.1\.0"                  -- $XN --version
expect "unknown option"          2 "unknown option '--warp'"      -- $XN --warp
expect "missing value"           2 "missing value for --model"    -- $XN --model
expect "no model"                2 "no --model"                   -- $XN --quiet

for m in models/*.xn; do
    expect "contact: $(basename "$m")" 0 "SIGNAL ACQUIRED"        -- $XN --model "$m"
done

for row in 00:0 01:1 10:1 11:0; do
    expect "infer ${row%:*} → ${row#*:}" 0 "^${row#*:}$"           -- $XN --model models/first-contact.xn --infer "${row%:*}" --quiet
done
expect "infer 6-bit row"         0 "101101 → 0"                   -- $XN --model models/deep-space.xn --infer 101101
expect "infer wrong width"       2 "needs 2 bits"                 -- $XN --model models/first-contact.xn --infer 101
expect "infer not bits"          2 "needs 2 bits"                 -- $XN --model models/first-contact.xn --infer 2x

expect "override seed"           0 "seed 0x7"                     -- $XN --model models/parity3.xn --seed 7
expect "override rejects junk"   2 "--epochs: value outside"      -- $XN --model models/parity3.xn --epochs 0
printf '[model]\ninputs = 2\n[layer]\nneurons = 1\n[options]\nepochs = 300\nrestarts = 2\n' > "$tmp/lonely.xn"
expect "one cell cannot xor"     1 "NOISE"                        -- $XN --model "$tmp/lonely.xn"

printf '[model]\ninputs = 2\n\n[layer]\nneurons = 2\nwobble = 1\n' > "$tmp/bad.xn"
expect "rejects with line"       3 "bad\.xn:6: unknown key"       -- $XN --model "$tmp/bad.xn"
expect "missing file"            3 "cannot read"                  -- $XN --model "$tmp/void.xn"

$XN --model models/parity3.xn --emit > "$tmp/p3.xn"
expect "emit carries weights"    0 "^thresholds = "               -- cat "$tmp/p3.xn"
expect "emitted file is trained" 0 "training skipped"             -- $XN --model "$tmp/p3.xn"
expect "--train relearns"        0 "listening"                    -- $XN --model "$tmp/p3.xn" --train
expect "emit is deterministic"   0 ""                             -- cmp "$tmp/p3.xn" <($XN --model models/parity3.xn --emit)

expect "prompt converses"        0 "→ 1   target 1  ✓"            -- bash -c "printf '01\nq\n' | $XN --model models/first-contact.xn --prompt"
expect "prompt survives junk"    0 "hears only 2 bits"            -- bash -c "printf 'zz\n' | $XN --model models/first-contact.xn --prompt"
expect "prompt closes on eof"    0 "channel closed"               -- bash -c "$XN --model models/signal.xn --prompt < /dev/null"

echo
echo "⊕ axn agents"
expect "swarm makes contact"     0 "SIGNAL ACQUIRED"              -- $XN --model models/parity3.xn --agents 2000
expect "swarm reports champion"  0 "champion  agent #[0-9]+"      -- $XN --model models/parity3.xn --agents 2000
expect "swarm honours threads"   0 "2000 agents · 3 threads"      -- $XN --model models/parity3.xn --agents 2000 --threads 3
expect "swarm retrains weights"  0 "◇ swarm"                      -- $XN --model models/signal.xn --agents 100
a=$($XN --model models/beacon.xn --agents 3000 --threads 1 --epochs 5000 --emit)
b=$($XN --model models/beacon.xn --agents 3000 --threads 7 --epochs 5000 --emit)
[[ $a == "$b" ]] && ok "swarm is thread-count invariant" || bad "swarm is thread-count invariant"
expect "swarm infer"             0 "^1$"                          -- $XN --model models/parity3.xn --agents 500 --infer 111 --quiet
expect "agents must be positive" 2 "--agents: wants a positive"   -- $XN --model models/parity3.xn --agents 0
expect "threads capped"          2 "--threads: wants"             -- $XN --model models/parity3.xn --agents 9 --threads 257
expect "stats"                   0 "model +4848"                  -- $XN --stats
expect "stats: agent is a seed"  0 "agent +8 +an agent is its seed" -- $XN --stats
expect "stats with a model"      0 "genome bits +85"              -- $XN --stats --model models/beacon.xn

echo
echo "⊕ xnc"
expect "xnc swarm"               0 "champion +agent #"            -- $XNC swarm models/first-contact.xn --agents 5000
expect "xnc version"             0 "abi 1"                        -- $XNC version
expect "xnc inspect"             0 "genome +27 bits"              -- $XNC inspect models/signal.xn
expect "xnc table"               0 "^7,111,1,1$"                  -- $XNC table models/majority.xn
expect "xnc sweep"               0 "contact +16/16"               -- $XNC sweep models/first-contact.xn --seeds 16
expect "xnc forge"               0 "^target  = nand$"             -- $XNC forge --inputs 3 --target nand --hidden 3
$XNC forge --inputs 3 --truth 0x81 --hidden 4 --name twins > "$tmp/twins.xn"
expect "forged model trains"     0 "SIGNAL ACQUIRED"              -- $XN --model "$tmp/twins.xn"
expect "xnc forge limits"        2 "--inputs wants 1..6"          -- $XNC forge --inputs 7
expect "xnc bad file"            2 "bad\.xn:6: unknown key"       -- $XNC inspect "$tmp/bad.xn"

echo
echo "  console checks · $pass passed · $fail failed"
((status == 0 && fail == 0)) && echo "  ⊕ all channels clear" || exit 1
