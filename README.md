# muse-glimmer-mlx

One-command setup to chat with **Muse Glimmer 30B** locally on an Apple Silicon Mac, using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Everything runs offline on your Mac once it's installed.

```
$ muse
Muse Glimmer ready. /clear to reset, /exit to quit.

You: What's the capital of Australia?
⠋ thinking...
Muse: The capital of Australia is Canberra.
```

## Requirements

**Built and tested on a MacBook Pro with M5 Pro and 24 GB of memory.** All the sizing choices (4-bit weights, no vision tower, 8-bit chat memory) are tuned to fit that machine. Other Apple Silicon Macs with 24 GB or more should work, but they haven't been tested.

- Mac with Apple Silicon (M1 or newer)
- **24 GB of memory or more**
- ~35 GB free disk during setup (15 GB once finished)
- Internet for the one-time ~15.6 GB download

Nothing else is needed. The installer brings its own Python.

## Install

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/nickwu00/muse-glimmer-mlx/main/setup.sh)
```

or clone the repo and run `./setup.sh`. Then open a new terminal and type `muse`.

Running it again is safe: steps that are already done are skipped.

## What the installer does

1. **Checks your Mac.** It needs Apple Silicon, at least 24 GB of memory and enough free disk.
2. **Installs [uv](https://github.com/astral-sh/uv)** if it's missing, then creates a Python 3.13 environment at `~/.muse/venv` with `mlx-vlm` pinned to 0.7.8.
3. **Downloads the model.** It uses [`mlx-community/Muse-Glimmer-30B-4bit`](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit), pinned to a specific revision. The fourth file only holds vision weights, so it's skipped.
4. **Removes the vision weights.** It writes a text-only copy to `~/.muse/model` (~15 GB, down from 19.4 GB), then deletes the download. mlx-vlm sees the missing vision parts and switches them off, so they never take up memory. That's what makes the model fit on a 24 GB Mac: it peaks at about 16 GB, under macOS's default GPU limit of about 19 GB.
5. **Writes the chat program** to `~/.muse/chat.py`. Muse Glimmer reasons before it answers. The chat shows a `thinking...` spinner during that step, then streams only the final answer.
6. **Adds a `muse` alias** to `~/.zshrc`.
7. **Runs a smoke test** that loads the model and gets one reply.

## Using it

| In the chat | |
|---|---|
| `/clear` | start a new conversation |
| `/exit` or Ctrl+D | quit |
| Ctrl+C | stop the current reply |

You can pass options when starting:

```bash
muse --temperature 0.3      # default 0.7
muse --max-tokens 8192      # default 4096; includes thinking tokens
muse --kv-bits 4            # default 8; shrinks memory used by long chats
```

If the model is still thinking when it reaches `--max-tokens`, you'll see a yellow note instead of an answer. Raise the limit or ask again.

Speed on an M5 Pro is about 18 tokens/sec. Close other memory-heavy apps while chatting.

## Text only

This setup can't read images. The full vision model is 19.4 GB, which is too big for a 24 GB Mac. If you have 32 GB or more and want image input, run `mlx-community/Muse-Glimmer-30B-4bit` directly with mlx-vlm.

## Uninstall

```bash
rm -rf ~/.muse
```

Then delete the two `muse` lines at the end of `~/.zshrc`. If the installer added uv, remove it with `rm ~/.local/bin/uv ~/.local/bin/uvx`.

## Install somewhere else

```bash
MUSE_HOME=/path/to/dir ./setup.sh
```

## License

The setup script and chat program are MIT-licensed (see [LICENSE](LICENSE)). This repo doesn't contain any model weights. The installer downloads Muse Glimmer from Hugging Face, where it's licensed under Apache 2.0. mlx-vlm is installed from PyPI under its MIT license.
