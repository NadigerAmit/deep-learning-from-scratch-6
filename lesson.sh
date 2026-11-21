#!/usr/bin/env bash
# Guided lesson (explain -> run -> explain, interactive, in this terminal):
#   ./lesson.sh ch02 03          (also: 2 3 | ch02/03 | 02_attn_math)
# Plain run only (no explanation):
#   ./lesson.sh --run ch02 03
#   ./lesson.sh --run --with tiktoken ch02 03     (temporary extra library)
# List lessons:
#   ./lesson.sh list [chapter]
# Uses the project's .venv (CPU torch) via `uv run --no-sync`; never plain `uv run`/`uv sync`.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"
ROOT="$(pwd)"
STUDY_DIR="$(dirname "$ROOT")"   # holds .agents/skills/lesson

[ -d .venv ] || { echo "Missing .venv. See study_progress/setup_and_execution_guide.md"; exit 1; }

mode=guided; with=()
if [ "${1:-}" = "list" ] || [ $# -eq 0 ]; then
  ch="${2:-}"
  if [ -n "$ch" ]; then ch="ch$(printf '%02d' "$((10#${ch#ch}))")"; ls "$ch"/*.py; else ls ch*/[0-9]*.py; fi
  exit 0
fi
if [ "$1" = "--run" ]; then mode=run; shift; fi
while [ "${1:-}" = "--with" ]; do with+=(--with "$2"); shift 2; done

# Normalise the arguments: "ch02 03", "2 3", "ch02/03", "02_attn_math"
arg="$*"; arg="${arg//\// }"
read -r a b <<<"$arg"
if [[ "$a" =~ ^(ch)?([0-9]+)$ ]]; then
  ch="ch$(printf '%02d' "$((10#${BASH_REMATCH[2]}))")"
else
  echo "Chapter missing or invalid: '$a'. Try: ./lesson.sh list"; exit 1
fi
[ -n "${b:-}" ] || { echo "Lesson number missing. Files:"; ls "$ch"/*.py; exit 1; }
num="$(printf '%02d' "$((10#${b%%_*}))")"

mapfile -t matches < <(ls "$ch"/"$num"_*.py 2>/dev/null || true)
if [ ${#matches[@]} -eq 0 ]; then echo "No lesson $ch/$num. Files:"; ls "$ch"/*.py; exit 1; fi
if [ ${#matches[@]} -gt 1 ]; then
  echo "More than one match; pick one:"; printf '  %s\n' "${matches[@]}"; exit 1
fi
file="${matches[0]}"

if [ "$mode" = guided ]; then
  command -v copilot >/dev/null || { echo "copilot CLI not found (npm install -g --prefix ~/.local @github/copilot)"; exit 1; }
  # --allow-all-tools lets the lesson read files, run the script and write progress notes without prompting
  cd "$STUDY_DIR"
  exec copilot --allow-all-tools --add-dir "$ROOT" -i "/lesson $ch $num"
fi

# Plain run: back up files this lesson overwrites (only if no backup exists yet)
case "$file" in
  ch01/07_tiny_codes.py) targets=(codebot/merge_rules.pkl) ;;
  ch01/09_bpe_encode.py) targets=(codebot/tiny_codes.bin) ;;
  ch03/01_pretrain.py)   targets=(codebot/model_pretrain.pt) ;;
  ch03/03_sft.py)        targets=(codebot/model_sft.pt) ;;
  ch03/09_grpo.py)       targets=(codebot/model_grpo.pt) ;;
  *) targets=() ;;
esac
for t in "${targets[@]}"; do
  bak="${t%.*}_original.${t##*.}"
  if [ -f "$t" ] && [ ! -f "$bak" ]; then cp -v "$t" "$bak"; fi
done

echo "=== Running $file ==="
exec uv run --no-sync "${with[@]}" python "$file"
