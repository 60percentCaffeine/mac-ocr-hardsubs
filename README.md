# ocranime

Extract burned-in subtitles (hardsubs) from anime videos into SRT files using macOS Vision framework OCR.

## Install

Requires **macOS** (uses the native Vision framework for OCR) and **Python 3.10+**.

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg and Poetry
brew install ffmpeg
brew install poetry

# Clone and install
git clone <repo-url> && cd ocranime
poetry install
```

### Ollama (optional)

Local LLM cleanup via [Ollama](https://ollama.com/). Free, runs on your machine.

```bash
brew install ollama
ollama serve &          # start the server
ollama pull qwen3:8b-q4_K_M  # pull the default model
```

Then use `--cleanup-backend=ollama`.

### OpenRouter (optional)

Cloud LLM cleanup via [OpenRouter](https://openrouter.ai/). Create an account and get an API key, then add it to a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-...
```

Then use `--cleanup-backend=openrouter`.

### Claude (optional, recommended)

LLM cleanup via [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Requires the `claude` CLI to be installed and authenticated.

Install Claude Code, then use `--cleanup-backend=claude`.

### Global install (run from anywhere)

To use `ocranime` as a command from any directory:

```bash
poetry install
# Add the virtualenv bin to your PATH
echo 'export PATH="$(poetry env info -p)/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Now you can run `ocranime` from anywhere:

```
ocranime video.mp4 --cleanup-backend=none
```

## Usage

```
poetry run python ocranime.py video.mp4 --cleanup-backend=none
```

### How it works

1. **Fast detection pass** — extracts frames at 4 fps and uses fast Apple OCR to classify each frame as "has text" or "no text"
2. **Clip building** — groups consecutive text frames into clips, drops clips shorter than 0.5s
3. **Accurate OCR pass** — runs accurate (slow) OCR only on frames within detected clips
4. **Deduplication & LLM cleanup** — merges consecutive identical entries, optionally cleans OCR artifacts with an LLM

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o/--output` | Output SRT path | `<video>.srt` |
| `--fps` | Frames per second to sample | `4` |
| `--crop` | ffmpeg crop filter | `iw*0.7:ih*0.15:iw*0.15:ih*0.8` |
| `--languages` | OCR languages | `zh-Hant` |
| `--cleanup-backend` | LLM backend: `none`, `ollama`, `openrouter` | required |
| `--cleanup-model` | Model for cleanup | auto per backend |
| `--cleanup-reasoning` | Enable (1) or disable (0) thinking | required if backend != none |
| `--scan-only` | Only detect subtitle frames, skip OCR/SRT | off |

> **Reasoning tips:** Reasoning (`--cleanup-reasoning=1`) is recommended with Claude and the default OpenRouter model for best cleanup quality. For Ollama, reasoning is **not recommended** — it makes inference very slow with local models.

### Examples

```
# Japanese subs, no LLM cleanup
poetry run python ocranime.py anime.mkv --languages ja --cleanup-backend=none

# Custom crop region, OpenRouter cleanup with reasoning
poetry run python ocranime.py video.mp4 --crop "iw*0.8:ih*0.2:iw*0.1:ih*0.75" --cleanup-backend=openrouter --cleanup-reasoning=1

# Just detect subtitle frames without full OCR
poetry run python ocranime.py video.mp4 --scan-only
```

Requires macOS and ffmpeg.
