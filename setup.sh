#!/usr/bin/env bash
# Muse Glimmer 30B — local text-only chat on Apple Silicon via mlx-vlm.
# Built and tested on a MacBook Pro (M5 Pro, 24 GB).
#
# Installs everything into ~/.muse (override with MUSE_HOME=...):
#   ~/.muse/venv     Python 3.13 env with mlx-vlm (pinned)
#   ~/.muse/model    4-bit Muse Glimmer with the vision tower stripped (~15 GB)
#   ~/.muse/chat.py  terminal chat: "thinking..." spinner, then only the answer
# and adds a `muse` alias to ~/.zshrc.
#
# Re-running is safe; finished steps are skipped.
# Uninstall: rm -rf ~/.muse  and delete the "muse" lines from ~/.zshrc.

set -euo pipefail

MUSE_HOME="${MUSE_HOME:-$HOME/.muse}"
REPO="mlx-community/Muse-Glimmer-30B-4bit"
REVISION="3e7677d7a40d348a3daba263a2b1c0aa41910710"  # pinned: shard layout below depends on it
MLX_VLM_VERSION="0.7.8"
PYTHON_VERSION="3.13"
MIN_RAM_GB=24
MIN_FREE_GB=35  # 15.6 GB download + 15 GB stripped copy, briefly both on disk

say()  { printf '\033[1;35m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- 1. Preflight -------------------------------------------------------------
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] \
  || fail "needs a Mac with Apple Silicon (M1 or newer)."

ram_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
(( ram_gb >= MIN_RAM_GB )) \
  || fail "needs at least ${MIN_RAM_GB} GB of memory; this Mac has ${ram_gb} GB."

mkdir -p "$MUSE_HOME"
free_gb=$(df -g "$MUSE_HOME" | awk 'NR==2 {print $4}')
if [[ ! -f "$MUSE_HOME/model/model.safetensors.index.json" ]] && (( free_gb < MIN_FREE_GB )); then
  fail "needs ~${MIN_FREE_GB} GB free disk during setup; only ${free_gb} GB free."
fi
say "Mac OK: $(sysctl -n machdep.cpu.brand_string), ${ram_gb} GB RAM, ${free_gb} GB free"

# --- 2. uv (Python package manager; also fetches Python itself) ---------------
if ! command -v uv >/dev/null 2>&1; then
  if [[ -x "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

# --- 3. Python env with mlx-vlm ----------------------------------------------
PY="$MUSE_HOME/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  say "Creating Python $PYTHON_VERSION environment"
  uv venv --quiet --python "$PYTHON_VERSION" "$MUSE_HOME/venv"
fi
say "Installing mlx-vlm $MLX_VLM_VERSION"
uv pip install --quiet --python "$PY" "mlx-vlm==$MLX_VLM_VERSION" "huggingface_hub[hf_xet]"

# --- 4. Download + strip vision weights --------------------------------------
if [[ ! -f "$MUSE_HOME/model/model.safetensors.index.json" ]]; then
  STAGE="$MUSE_HOME/download"
  say "Downloading $REPO (~15.6 GB; shard 4 is vision-only and skipped)"
  "$MUSE_HOME/venv/bin/hf" download "$REPO" --revision "$REVISION" \
    --exclude "model-00004-of-00004.safetensors" --local-dir "$STAGE"

  say "Removing vision weights (text-only copy fits in 24 GB Macs)"
  "$PY" -I - "$STAGE" "$MUSE_HOME/model.tmp" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
import mlx.core as mx

DROP = ("vision_tower.", "vision_adapter.", "vision_projection.")
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(dst, ignore_errors=True)
dst.mkdir(parents=True)
for f in src.iterdir():
    if f.is_file() and f.suffix != ".safetensors" and f.name != "model.safetensors.index.json":
        shutil.copy(f, dst / f.name)
weight_map = {}
for shard in sorted(src.glob("*.safetensors")):
    weights = mx.load(str(shard))
    kept = {k: v for k, v in weights.items() if not k.startswith(DROP)}
    if kept:
        mx.save_safetensors(str(dst / shard.name), kept, metadata={"format": "mlx"})
        weight_map.update({k: shard.name for k in kept})
    print(f"  {shard.name}: kept {len(kept)}/{len(weights)} tensors")
    del weights, kept
(dst / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))
PYEOF
  rm -rf "$MUSE_HOME/model"
  mv "$MUSE_HOME/model.tmp" "$MUSE_HOME/model"
  rm -rf "$STAGE"
else
  say "Model already in place, skipping download"
fi

# --- 5. Chat script ----------------------------------------------------------
say "Writing chat script"
cat > "$MUSE_HOME/chat.py" <<'PYEOF'
"""Terminal chat for Muse Glimmer that hides the reasoning channel.

Shows a "thinking..." spinner while the model reasons, then streams only the
final to=user answer. Commands: /clear, /exit.
"""

import argparse
import logging
import os
import readline  # noqa: F401  (arrow keys / history in input())
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from rich.console import Console

from mlx_vlm import load
from mlx_vlm.generate import PromptCacheState, stream_generate
from mlx_vlm.models.muse_glimmer.processing_muse_glimmer import (
    _USER_CHANNEL_RE,
    _extract_final_channel,
)
from mlx_vlm.prompt_utils import apply_chat_template

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
parser.add_argument("--max-tokens", type=int, default=4096)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--kv-bits", type=int, default=8)
args = parser.parse_args()

console = Console()
with console.status("[dim]loading Muse Glimmer...[/dim]"):
    model, processor = load(args.model)
console.print("[dim]Muse Glimmer ready. /clear to reset, /exit to quit.[/dim]\n")

history = []
cache_state = PromptCacheState()


def visible(answer: str) -> str:
    """Hold back a trailing partial marker like '<|eo' until it resolves."""
    cut = answer.rfind("<", max(0, len(answer) - 12))
    return answer[:cut] if cut != -1 else answer


def respond() -> str:
    prompt = apply_chat_template(processor, model.config, history, num_images=0)
    text, shown = "", 0
    status = console.status("[dim italic]thinking...[/dim italic]", spinner="dots")
    status.start()
    try:
        for chunk in stream_generate(
            model,
            processor,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            prompt_cache_state=cache_state,
            kv_bits=args.kv_bits,
        ):
            text += chunk.text
            if not _USER_CHANNEL_RE.search(text):
                continue
            if status is not None:
                status.stop()
                status = None
                console.print("[bold magenta]Muse:[/bold magenta] ", end="")
            answer = visible(_extract_final_channel(text))
            if len(answer) > shown:
                print(answer[shown:], end="", flush=True)
                shown = len(answer)
    finally:
        if status is not None:
            status.stop()

    answer = _extract_final_channel(text)
    if status is None:
        print(answer[shown:])  # flush anything held back
    else:
        # Ran out of tokens before answering.
        console.print("[yellow](no answer within the token limit — try again or raise --max-tokens)[/yellow]")
    print()
    return answer


while True:
    try:
        user = console.input("[bold cyan]You:[/bold cyan] ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user == "/exit":
        break
    if user == "/clear":
        history.clear()
        cache_state = PromptCacheState()
        console.print("[dim]conversation cleared[/dim]\n")
        continue

    history.append({"role": "user", "content": [{"type": "text", "text": user}]})
    try:
        answer = respond()
    except KeyboardInterrupt:
        console.print("\n[dim](interrupted)[/dim]\n")
        history.pop()
        cache_state = PromptCacheState()
        continue
    history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
PYEOF

# --- 6. Alias ----------------------------------------------------------------
RC="$HOME/.zshrc"
ALIAS_LINE="alias muse='\"$PY\" \"$MUSE_HOME/chat.py\"'"
touch "$RC"
if grep -q "^alias muse=" "$RC"; then
  # Replace any existing muse alias so it points at this install.
  grep -v "^alias muse=" "$RC" > "$RC.muse-tmp" && mv "$RC.muse-tmp" "$RC"
fi
grep -q "# Muse Glimmer" "$RC" || printf '\n# Muse Glimmer local chat (muse-glimmer-mlx)\n' >> "$RC"
echo "$ALIAS_LINE" >> "$RC"

# --- 7. Smoke test -----------------------------------------------------------
say "Testing (loads the model and asks one question, ~30 s)"
LOG="$MUSE_HOME/smoke-test.log"
reply=""
for attempt in 1 2; do
  printf 'Reply with just the word: ready\n/exit\n' \
    | "$PY" "$MUSE_HOME/chat.py" --temperature 0 --max-tokens 1024 > "$LOG" 2>&1 || true
  reply=$(tr '\r' '\n' < "$LOG" | grep -o 'Muse: .*' | head -1 || true)
  [[ -n "$reply" ]] && break
done
if [[ -n "$reply" ]]; then
  say "Model said -> ${reply#Muse: }"
  rm -f "$LOG"
else
  printf '\033[1;33mwarning:\033[0m install finished, but the test chat got no answer. Last output:\n' >&2
  tr '\r' '\n' < "$LOG" | grep -v '^\s*$' | tail -15 >&2
  printf 'Full log: %s\nTry `muse` in a new terminal anyway.\n' "$LOG" >&2
fi

cat <<EOF

Done. Open a new terminal (or run: source ~/.zshrc) and type:

    muse

In the chat: /clear starts over, /exit quits, Ctrl+C stops a reply.
Options: muse --temperature 0.3 --max-tokens 8192
EOF
