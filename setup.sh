#!/usr/bin/env bash
# setup.sh ─ gather what the cortex needs: nasm, binutils, a C++26 compiler.
#
# Nothing is downloaded beyond your distribution's packages: the engine is
# self-contained assembly and the tools use only the C++ standard library.
# Re-running is safe; it installs only what is missing, then verifies.
#
#   ./setup.sh            install missing packages (uses sudo when not root)
#   ./setup.sh --check    verify only, install nothing
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '  \342\212\225 %s\n' "$*"; }
fail() { printf '  \342\234\227 %s\n' "$*" >&2; exit 1; }

check_only=0
[[ ${1:-} == --check ]] && check_only=1

[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] \
    || fail "xn speaks Linux x86-64 syscalls; this host is $(uname -sm)"

missing=()
command -v nasm >/dev/null || missing+=(nasm)
command -v ld   >/dev/null || missing+=(binutils)
command -v ar   >/dev/null || missing+=(binutils)
command -v "${CXX:-g++}" >/dev/null || missing+=(g++)

if ((${#missing[@]})); then
    ((check_only)) && fail "missing: ${missing[*]}"
    sudo=""; ((EUID)) && sudo=sudo
    say "installing: ${missing[*]}"
    if   command -v apt-get >/dev/null; then $sudo apt-get update -qq && $sudo apt-get install -y "${missing[@]}"
    elif command -v dnf     >/dev/null; then $sudo dnf install -y "${missing[@]/g++/gcc-c++}"
    elif command -v pacman  >/dev/null; then $sudo pacman -S --needed --noconfirm "${missing[@]/g++/gcc}"
    elif command -v zypper  >/dev/null; then $sudo zypper install -y "${missing[@]/g++/gcc-c++}"
    else fail "unknown package manager; install by hand: ${missing[*]}"
    fi
fi

# ── verify ──────────────────────────────────────────────────────────────────
say "nasm      $(nasm -v | awk '{print $3}')"
say "ld        $(ld -v | awk '{print $NF}')"

probe=$(mktemp -d)
trap 'rm -rf "$probe"' EXIT
cat > "$probe/c26.cpp" <<'EOF'
#include <expected>
#include <print>
#include <ranges>
void sealed() = delete("c++26 deleted-with-reason");
template <class... T> using first = T...[0];
int main() { auto _ = 1; auto _ = 2; first<int, char> v = 26; std::println("{}", v); }
EOF
"${CXX:-g++}" -std=c++26 "$probe/c26.cpp" -o "$probe/c26" 2>/dev/null \
    || fail "${CXX:-g++} cannot compile C++26 (need GCC >= 15 or Clang >= 19 with libstdc++ 15); set CXX="
[[ $("$probe/c26") == 26 ]] || fail "C++26 probe ran but misbehaved"
say "c++       $("${CXX:-g++}" --version | head -1)  (c++26 ok)"

grep -qw popcnt /proc/cpuinfo || fail "this CPU lacks POPCNT; the cortex counts with it"
say "cpu       popcnt ok"
say "ready · next: ./build.sh"
