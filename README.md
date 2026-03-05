# ocranime

Extract burned-in subtitles (hardsubs) from anime videos into SRT files using macOS Vision framework OCR.

## Setup

```
poetry install
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
