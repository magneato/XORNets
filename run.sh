#!/usr/bin/env bash
# run.sh ─ open a channel.  Pick a model, watch it learn, then talk to it.
#
#   ./run.sh                      choose from models/ interactively
#   ./run.sh models/beacon.xn     go straight to that model
#   extra arguments are passed to xn, e.g.  ./run.sh models/parity3.xn --seed 7
set -euo pipefail
cd "$(dirname "$0")"

[[ -x build/xn ]] || ./build.sh

model=${1:-}
if [[ -z $model ]]; then
    mapfile -t models < <(ls models/*.xn)
    printf '\n  \342\212\225 known transmissions\n\n'
    for i in "${!models[@]}"; do
        name=$(basename "${models[$i]}" .xn)
        note=$(sed -n '2s/^# *//p' "${models[$i]}")
        printf '    %d  %-14s %s\n' "$((i + 1))" "$name" "$note"
    done
    printf '\n  choose [1]: '
    read -r pick || pick=""
    pick=${pick:-1}
    [[ $pick =~ ^[0-9]+$ ]] && ((pick >= 1 && pick <= ${#models[@]})) \
        || { echo "  no such transmission"; exit 2; }
    model=${models[$((pick - 1))]}
else
    shift
fi

exec build/xn --model "$model" --prompt "$@"
