#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${CLANG_FORMAT_BIN:-}" ]]; then
  CLANG_FORMAT="${CLANG_FORMAT_BIN}"
elif command -v clang-format-22 >/dev/null 2>&1; then
  CLANG_FORMAT="$(command -v clang-format-22)"
elif command -v clang-format-18 >/dev/null 2>&1; then
  CLANG_FORMAT="$(command -v clang-format-18)"
elif command -v clang-format >/dev/null 2>&1; then
  CLANG_FORMAT="$(command -v clang-format)"
else
  echo "clang-format not found. Set CLANG_FORMAT_BIN to an explicit binary path." >&2
  exit 1
fi

echo "Using clang-format: ${CLANG_FORMAT}"
"${CLANG_FORMAT}" --version

find include tests \
  \( -name '*.hpp' -o -name '*.cpp' \) \
  -print \
  | sort \
  | xargs "${CLANG_FORMAT}" --dry-run --Werror
