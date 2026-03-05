#!/usr/bin/env python3
"""Extract burned-in subtitles from anime videos into SRT using macOS Vision OCR."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

import Vision
from Foundation import NSData

from dotenv import load_dotenv

load_dotenv()


MIN_CLIP_DURATION = 0.5  # seconds — clips shorter than this are dropped


def extract_frames(video_path, tmpdir, crop, fps):
    """Extract cropped frames from video using ffmpeg at given fps."""
    output_pattern = os.path.join(tmpdir, "%06d.png")
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"crop={crop},fps={fps}",
        "-loglevel",
        "error",
        output_pattern,
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(Path(tmpdir).glob("*.png"))
    return frames


def detect_text_frames(frames, languages):
    """OCR all frames, return list of recognized text (empty string if no text).

    Uses accurate mode because fast mode doesn't reliably detect CJK text.
    Results are cached so the accurate OCR pass can reuse them.
    """
    texts = []
    for idx, frame in enumerate(frames):
        text = ocr_frame(frame, languages, fast=False).strip()
        texts.append(text)
        progress = (idx + 1) / len(frames) * 100
        print(f"\rOCR: {progress:.0f}%", end="", flush=True)
    print()
    return texts


def build_clips(texts, sample_interval):
    """Build clips (start_idx, end_idx) from consecutive non-empty texts, drop short ones."""
    clips = []
    start = None
    for i, text in enumerate(texts):
        if text and start is None:
            start = i
        elif not text and start is not None:
            clips.append((start, i - 1))
            start = None
    if start is not None:
        clips.append((start, len(texts) - 1))

    # Filter out clips shorter than MIN_CLIP_DURATION
    filtered = []
    for s, e in clips:
        duration = (e - s + 1) * sample_interval
        if duration >= MIN_CLIP_DURATION:
            filtered.append((s, e))
    return filtered


def ocr_frame(frame_path, languages, fast):
    """Run Vision OCR on a single frame, return recognized text."""
    data = NSData.dataWithContentsOfFile_(str(frame_path))
    if data is None:
        return ""

    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(1 if fast else 0)
    request.setRecognitionLanguages_(languages)

    handler.performRequests_error_([request], None)

    results = request.results()
    if not results:
        return ""

    lines = []
    for obs in results:
        candidate = obs.topCandidates_(1)
        if candidate:
            lines.append(candidate[0].string())

    return "\n".join(lines)


def format_timestamp(seconds):
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _edit_distance(a, b):
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def deduplicate(entries, max_dist=0):
    """Merge consecutive entries with identical (or near-identical) text."""
    if not entries:
        return []

    merged = []
    current_text, current_start, current_end = entries[0]

    for text, start, end in entries[1:]:
        if text == current_text or (
            max_dist > 0 and _edit_distance(text, current_text) <= max_dist
        ):
            current_end = end
        else:
            merged.append((current_text, current_start, current_end))
            current_text, current_start, current_end = text, start, end

    merged.append((current_text, current_start, current_end))
    return merged


def strip_thinking(text):
    """Remove <think>...</think> blocks from Qwen 3 reasoning output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(messages, model="qwen3:8b-q4_K_M", temperature=0.3):
    """Call Ollama native chat API with streaming. Returns assistant content."""
    url = "http://localhost:11434/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"Error: cannot connect to Ollama at {url}: {e}", file=sys.stderr)
        print(
            "Make sure Ollama is running (ollama serve) and the model is pulled.",
            file=sys.stderr,
        )
        print("Use --cleanup-backend=none to skip LLM cleanup.", file=sys.stderr)
        sys.exit(1)
    content_parts = []
    try:
        for line in resp:
            chunk = json.loads(line)
            content_parts.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                break
    finally:
        resp.close()
    return strip_thinking("".join(content_parts))


def call_openrouter(messages, model="qwen/qwen3-235b-a22b-2507", temperature=0.3):
    """Call OpenRouter chat API with streaming. Returns assistant content."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "Error: OPENROUTER_API_KEY not set. Add it to .env or environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        print(f"Error: cannot connect to OpenRouter: {e}", file=sys.stderr)
        sys.exit(1)
    content_parts = []
    try:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content_parts.append(delta.get("content", ""))
    finally:
        resp.close()
    return strip_thinking("".join(content_parts))


def call_claude(messages, model="claude-sonnet-4-6", temperature=0.3, thinking=True):
    """Call Claude via Claude Code headless mode. Returns assistant content."""
    # Build the prompt: system prompt + user message
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            user_prompt = msg["content"]

    cmd = ["claude", "-p", "--model", model, "--output-format", "text"]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    env = None
    if not thinking:
        env = os.environ.copy()
        env["MAX_THINKING_TOKENS"] = "0"

    try:
        result = subprocess.run(
            cmd,
            input=user_prompt,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except FileNotFoundError:
        print(
            "Error: 'claude' CLI not found. Install Claude Code first.", file=sys.stderr
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Error: Claude CLI timed out after 300s.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(f"Error: Claude CLI failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    return result.stdout.strip()


def parse_cleanup_response(response_text, batch_entries):
    """Parse KEEP/REMOVE lines from LLM response. Returns filtered entries."""
    keep_pattern = re.compile(r"^KEEP\s+(\d+)\s*:\s*(.+)$", re.MULTILINE)
    remove_pattern = re.compile(r"^REMOVE\s+(\d+)\s*$", re.MULTILINE)

    keeps = {
        int(m.group(1)): m.group(2).strip().replace(" | ", "\n")
        for m in keep_pattern.finditer(response_text)
    }
    removes = {int(m.group(1)) for m in remove_pattern.finditer(response_text)}

    if not keeps and not removes:
        return None  # Unparseable

    result = []
    for i, (text, start, end) in enumerate(batch_entries, 1):
        if i in removes:
            continue
        if i in keeps:
            cleaned = keeps[i]
            result.append((cleaned, start, end))
        else:
            # Not mentioned — keep unchanged (safe fallback)
            result.append((text, start, end))
    return result


def cleanup_with_llm(
    entries,
    model="qwen3:8b-q4_K_M",
    batch_size=30,
    languages=None,
    thinking=True,
    backend="ollama",
):
    """Clean OCR artifacts using LLM. Returns filtered entries."""
    if not entries:
        return entries

    if languages is None:
        languages = ["zh-Hant"]

    lang_names = {
        "zh-Hant": "Traditional Chinese",
        "zh-Hans": "Simplified Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "en": "English",
    }
    lang_desc = ", ".join(lang_names.get(l, l) for l in languages)

    prompt_path = Path(__file__).parent / "cleanup-prompt.md"
    system_prompt = (
        prompt_path.read_text(encoding="utf-8").strip().format(lang_desc=lang_desc)
    )

    temperature = 0.2
    if backend == "openrouter":
        call_llm_fn = call_openrouter
    elif backend == "claude":
        call_llm_fn = lambda msgs, model, temperature: call_claude(
            msgs, model=model, temperature=temperature, thinking=thinking
        )
    else:
        call_llm_fn = call_ollama

    all_results = []
    total_batches = (len(entries) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch = entries[start : start + batch_size]
        print(f"  Batch {batch_idx + 1}/{total_batches}...", flush=True)

        user_lines = []
        for i, (text, _, _) in enumerate(batch, 1):
            # Replace newlines with ' | ' so multi-line entries are unambiguous
            user_lines.append(f"[{i}] {text.replace(chr(10), ' | ')}")
        user_prompt = "\n".join(user_lines)
        if not thinking and backend != "claude":
            user_prompt += " /no_think"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = call_llm_fn(messages, model=model, temperature=temperature)
        result = parse_cleanup_response(response, batch)

        if result is None:
            # Retry once
            print(
                f"  Batch {batch_idx + 1}: response unparseable, retrying...",
                flush=True,
            )
            response = call_llm_fn(messages, model=model, temperature=temperature)
            result = parse_cleanup_response(response, batch)
            if result is None:
                print(
                    f"Error: LLM response for batch {batch_idx + 1} is unparseable after retry.",
                    file=sys.stderr,
                )
                print(f"Response was:\n{response}", file=sys.stderr)
                sys.exit(1)

        all_results.extend(result)

    return all_results


def fill_clip_gaps(entries, clips, sample_interval):
    """Extend subtitles so every frame inside detected clips is covered.

    After LLM cleanup some entries may be removed, leaving gaps within clips
    that were determined to contain subtitles.  For each uncovered frame,
    extend the previous subtitle's end time to cover it.
    """
    if not entries:
        return entries

    # Build the set of timestamps (as frame indices) that must be covered
    covered_times = set()
    for s, e in clips:
        for idx in range(s, e + 1):
            covered_times.add(round(idx * sample_interval, 6))

    # For each required timestamp, check if any entry covers it
    result = list(entries)
    for t in sorted(covered_times):
        t_rounded = round(t, 6)
        # Find if any entry covers this timestamp
        covered = False
        for text, start, end in result:
            if round(start, 6) <= t_rounded < round(end, 6):
                covered = True
                break
        if not covered:
            # Find the last entry that ends at or before this timestamp
            best = None
            for i, (text, start, end) in enumerate(result):
                if round(end, 6) <= t_rounded:
                    best = i
            if best is not None:
                text, start, end = result[best]
                new_end = round(t + sample_interval, 6)
                result[best] = (text, start, new_end)

    return result


def write_srt(entries, output_path):
    """Write entries as SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (text, start, end) in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract hardsubs from anime video to SRT"
    )
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output SRT path (default: <video>.srt)")
    parser.add_argument(
        "--crop",
        default="iw*0.7:ih*0.15:iw*0.15:ih*0.8",
        help="ffmpeg crop filter (default: iw*0.7:ih*0.15:iw*0.15:ih*0.8)",
    )
    parser.add_argument(
        "--fps", type=int, default=4, help="Frames per second to sample (default: 4)"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["zh-Hant"],
        help="OCR languages (default: zh-Hant)",
    )
    parser.add_argument(
        "--cleanup-backend",
        default=None,
        choices=["none", "ollama", "openrouter", "claude"],
        help="LLM backend for OCR cleanup (none to skip)",
    )
    parser.add_argument(
        "--cleanup-model",
        default=None,
        help="Model for cleanup (default: qwen3:8b-q4_K_M for ollama, "
        "qwen/qwen3-235b-a22b-2507 for openrouter, "
        "claude-sonnet-4-6 for claude)",
    )
    parser.add_argument(
        "--cleanup-reasoning",
        type=int,
        choices=[0, 1],
        default=None,
        help="Enable (1) or disable (0) thinking/reasoning for cleanup model",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only generate subtitle frame PNGs in scanned_frames/, skip SRT generation",
    )
    args = parser.parse_args()

    scan_only_incompatible = []
    if args.scan_only:
        if args.output is not None:
            scan_only_incompatible.append("--output")
        if args.languages != ["zh-Hant"]:
            scan_only_incompatible.append("--languages")
        if args.cleanup_backend is not None:
            scan_only_incompatible.append("--cleanup-backend")
        if args.cleanup_model is not None:
            scan_only_incompatible.append("--cleanup-model")
        if args.cleanup_reasoning is not None:
            scan_only_incompatible.append("--cleanup-reasoning")
        if scan_only_incompatible:
            parser.error(
                f"--scan-only is incompatible with: {', '.join(scan_only_incompatible)}"
            )
    else:
        if args.cleanup_backend is None:
            parser.error("--cleanup-backend is required when not using --scan-only")
        if args.cleanup_backend != "none" and args.cleanup_reasoning is None:
            parser.error(
                "--cleanup-reasoning is required when --cleanup-backend is not 'none'"
            )

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (os.path.splitext(video_path)[0] + ".srt")

    # Prepare scanned_frames directory (empty it on each run)
    scanned_dir = Path("scanned_frames")
    if scanned_dir.exists():
        shutil.rmtree(scanned_dir)
    scanned_dir.mkdir()

    fps = args.fps
    sample_interval = 1.0 / fps

    with tempfile.TemporaryDirectory(prefix="ocranime_") as tmpdir:
        print(f"Extracting frames ({fps} fps)...")
        frames = extract_frames(video_path, tmpdir, args.crop, fps)
        print(f"Extracted {len(frames)} frames")

        # OCR all frames (accurate mode — fast mode can't detect CJK)
        ocr_texts = detect_text_frames(frames, args.languages)
        clips = build_clips(ocr_texts, sample_interval)
        text_frame_count = sum(e - s + 1 for s, e in clips)
        print(
            f"Found {len(clips)} subtitle clips ({text_frame_count} frames with text)"
        )

        # Copy clip frames to scanned_frames/ with timestamp names
        for s, e in clips:
            for idx in range(s, e + 1):
                timestamp = idx * sample_interval
                h = int(timestamp // 3600)
                m = int((timestamp % 3600) // 60)
                s_sec = int(timestamp % 60)
                ms = int(round((timestamp - int(timestamp)) * 1000))
                dest = scanned_dir / f"{h:02d}-{m:02d}-{s_sec:02d}-{ms:03d}.png"
                shutil.copy2(frames[idx], dest)
        print(f"Saved {text_frame_count} frame PNGs to {scanned_dir}/")

        if args.scan_only:
            print("Scan-only mode: skipping SRT generation.")
            return

        # Build entries from OCR results within detected clips
        raw_entries = []
        for s, e in clips:
            for idx in range(s, e + 1):
                timestamp = idx * sample_interval
                text = ocr_texts[idx]
                if text:
                    raw_entries.append((text, timestamp, timestamp + sample_interval))

        entries = deduplicate(raw_entries)
        if args.cleanup_backend != "none":
            # Save raw (pre-cleanup) SRT alongside the final one
            raw_path = os.path.splitext(output_path)[0] + ".raw.srt"
            write_srt(entries, raw_path)
            print(f"Written {len(entries)} raw subtitle entries to {raw_path}")

            backend = args.cleanup_backend
            default_models = {
                "openrouter": "qwen/qwen3-235b-a22b-2507",
                "claude": "claude-sonnet-4-6",
                "ollama": "qwen3:8b-q4_K_M",
            }
            default_model = default_models.get(backend, "qwen3:8b-q4_K_M")
            cleanup_model = args.cleanup_model or default_model
            thinking = bool(args.cleanup_reasoning)
            print(f"Cleaning up OCR artifacts with {cleanup_model} ({backend})...")
            entries = cleanup_with_llm(
                entries,
                model=cleanup_model,
                languages=args.languages,
                thinking=thinking,
                backend=backend,
            )
            # Fuzzy dedup: merge consecutive near-duplicate entries the LLM missed
            entries = deduplicate(entries, max_dist=2)
            # Ensure every frame in detected clips is covered by a subtitle
            entries = fill_clip_gaps(entries, clips, sample_interval)
        write_srt(entries, output_path)
        print(f"Written {len(entries)} subtitle entries to {output_path}")


if __name__ == "__main__":
    main()
