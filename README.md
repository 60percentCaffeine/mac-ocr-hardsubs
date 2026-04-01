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
4. **Deduplication & cleanup** — merges consecutive identical entries, optionally cleans OCR artifacts with rule-based cleanup

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o/--output` | Output SRT path | `<video>.srt` |
| `--fps` | Frames per second to sample | `4` |
| `--crop` | ffmpeg crop filter | `iw*0.7:ih*0.15:iw*0.15:ih*0.8` |
| `--languages` | OCR languages | `zh-Hant` |
| `--cleanup-backend` | Cleanup backend: `none`, `rules` | required |
| `--scan-only` | Only detect subtitle frames, skip OCR/SRT | off |

### Examples

```
# Japanese subs, no cleanup
poetry run python ocranime.py anime.mkv --languages ja --cleanup-backend=none

# Rules-based cleanup
poetry run python ocranime.py video.mp4 --cleanup-backend=rules

# Just detect subtitle frames without full OCR
poetry run python ocranime.py video.mp4 --scan-only
```

Requires macOS and ffmpeg.
