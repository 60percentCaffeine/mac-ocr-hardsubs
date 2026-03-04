# ocranime

Extract burned-in subtitles (hardsubs) from anime videos into SRT files using macOS Vision framework OCR.

## Setup

```
poetry install
```

## Usage

```
poetry run python ocranime.py video.mp4
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o/--output` | Output SRT path | `<video>.srt` |
| `-i/--interval` | Seconds between sampled frames | `1.0` |
| `--crop` | ffmpeg crop filter | `iw*0.7:ih*0.15:iw*0.15:ih*0.8` |
| `--languages` | OCR languages | `zh-Hant` |
| `--fast` | Use fast OCR mode | off |

### Examples

```
# Japanese subs, sample every 0.5s
poetry run python ocranime.py anime.mkv --languages ja -i 0.5

# Custom crop region, fast mode
poetry run python ocranime.py video.mp4 --crop "iw*0.8:ih*0.2:iw*0.1:ih*0.75" --fast
```

Requires macOS and ffmpeg.
