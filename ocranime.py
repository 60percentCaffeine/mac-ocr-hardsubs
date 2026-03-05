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


def extract_frames(video_path, tmpdir, interval, crop):
    """Extract cropped frames from video using ffmpeg."""
    output_pattern = os.path.join(tmpdir, "%06d.png")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"crop={crop},fps=1/{interval}",
        "-loglevel", "error",
        output_pattern,
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(Path(tmpdir).glob("*.png"))
    return frames


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
        if text == current_text or (max_dist > 0 and _edit_distance(text, current_text) <= max_dist):
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
        print("Make sure Ollama is running (ollama serve) and the model is pulled.", file=sys.stderr)
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
        print("Error: OPENROUTER_API_KEY not set. Add it to .env or environment.", file=sys.stderr)
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


def parse_cleanup_response(response_text, batch_entries):
    """Parse KEEP/REMOVE lines from LLM response. Returns filtered entries."""
    keep_pattern = re.compile(r"^KEEP\s+(\d+)\s*:\s*(.+)$", re.MULTILINE)
    remove_pattern = re.compile(r"^REMOVE\s+(\d+)\s*$", re.MULTILINE)

    keeps = {int(m.group(1)): m.group(2).strip().replace(" | ", "\n") for m in keep_pattern.finditer(response_text)}
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


def cleanup_with_llm(entries, model="qwen3:8b-q4_K_M", batch_size=30, languages=None, thinking=True,
                     backend="ollama"):
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
    system_prompt = prompt_path.read_text(encoding="utf-8").strip().format(lang_desc=lang_desc)

    temperature = 0.2
    call_llm_fn = call_openrouter if backend == "openrouter" else call_ollama

    all_results = []
    total_batches = (len(entries) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        batch = entries[start:start + batch_size]
        print(f"  Batch {batch_idx + 1}/{total_batches}...", flush=True)

        user_lines = []
        for i, (text, _, _) in enumerate(batch, 1):
            # Replace newlines with ' | ' so multi-line entries are unambiguous
            user_lines.append(f"[{i}] {text.replace(chr(10), ' | ')}")
        user_prompt = "\n".join(user_lines)
        if not thinking:
            user_prompt += " /no_think"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = call_llm_fn(messages, model=model, temperature=temperature)
        result = parse_cleanup_response(response, batch)

        if result is None:
            # Retry once
            print(f"  Batch {batch_idx + 1}: response unparseable, retrying...", flush=True)
            response = call_llm_fn(messages, model=model, temperature=temperature)
            result = parse_cleanup_response(response, batch)
            if result is None:
                print(f"Error: LLM response for batch {batch_idx + 1} is unparseable after retry.",
                      file=sys.stderr)
                print(f"Response was:\n{response}", file=sys.stderr)
                sys.exit(1)

        all_results.extend(result)

    return all_results


def write_srt(entries, output_path):
    """Write entries as SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (text, start, end) in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Extract hardsubs from anime video to SRT")
    parser.add_argument("video", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output SRT path (default: <video>.srt)")
    parser.add_argument("-i", "--interval", type=float, default=1.0,
                        help="Seconds between sampled frames (default: 1.0)")
    parser.add_argument("--crop", default="iw*0.7:ih*0.15:iw*0.15:ih*0.8",
                        help="ffmpeg crop filter (default: iw*0.7:ih*0.15:iw*0.15:ih*0.8)")
    parser.add_argument("--languages", nargs="+", default=["zh-Hant"],
                        help="OCR languages (default: zh-Hant)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast OCR mode instead of accurate")
    parser.add_argument("--cleanup-backend", default=None,
                        choices=["none", "ollama", "openrouter"],
                        help="LLM backend for OCR cleanup (none to skip)")
    parser.add_argument("--cleanup-model", default=None,
                        help="Model for cleanup (default: qwen3:8b-q4_K_M for ollama, "
                             "qwen/qwen3-235b-a22b-2507 for openrouter)")
    parser.add_argument("--cleanup-reasoning", type=int, choices=[0, 1], default=None,
                        help="Enable (1) or disable (0) thinking/reasoning for cleanup model")
    parser.add_argument("--scan-only", action="store_true",
                        help="Only generate subtitle frame PNGs in scanned_frames/, skip SRT generation")
    args = parser.parse_args()

    scan_only_incompatible = []
    if args.scan_only:
        if args.output is not None:
            scan_only_incompatible.append("--output")
        if args.languages != ["zh-Hant"]:
            scan_only_incompatible.append("--languages")
        if args.fast:
            scan_only_incompatible.append("--fast")
        if args.cleanup_backend is not None:
            scan_only_incompatible.append("--cleanup-backend")
        if args.cleanup_model is not None:
            scan_only_incompatible.append("--cleanup-model")
        if args.cleanup_reasoning is not None:
            scan_only_incompatible.append("--cleanup-reasoning")
        if scan_only_incompatible:
            parser.error(f"--scan-only is incompatible with: {', '.join(scan_only_incompatible)}")
    else:
        if args.cleanup_backend is None:
            parser.error("--cleanup-backend is required when not using --scan-only")
        if args.cleanup_backend != "none" and args.cleanup_reasoning is None:
            parser.error("--cleanup-reasoning is required when --cleanup-backend is not 'none'")

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

    with tempfile.TemporaryDirectory(prefix="ocranime_") as tmpdir:
        print(f"Extracting frames (interval={args.interval}s)...")
        frames = extract_frames(video_path, tmpdir, args.interval, args.crop)
        print(f"Extracted {len(frames)} frames")

        # Copy frames to scanned_frames with timestamp names
        for idx, frame in enumerate(frames):
            timestamp = idx * args.interval
            h = int(timestamp // 3600)
            m = int((timestamp % 3600) // 60)
            s = int(timestamp % 60)
            dest = scanned_dir / f"{h:02d}-{m:02d}-{s:02d}.png"
            shutil.copy2(frame, dest)
        print(f"Saved {len(frames)} frame PNGs to {scanned_dir}/")

        if args.scan_only:
            print("Scan-only mode: skipping OCR and SRT generation.")
            return

        raw_entries = []
        for idx, frame in enumerate(frames):
            timestamp = idx * args.interval
            text = ocr_frame(frame, args.languages, args.fast).strip()
            if text:
                raw_entries.append((text, timestamp, timestamp + args.interval))
            progress = (idx + 1) / len(frames) * 100
            print(f"\rOCR: {progress:.0f}%", end="", flush=True)
        print()

        entries = deduplicate(raw_entries)
        if args.cleanup_backend != "none":
            # Save raw (pre-cleanup) SRT alongside the final one
            raw_path = os.path.splitext(output_path)[0] + ".raw.srt"
            write_srt(entries, raw_path)
            print(f"Written {len(entries)} raw subtitle entries to {raw_path}")

            backend = args.cleanup_backend
            default_model = ("qwen/qwen3-235b-a22b-2507" if backend == "openrouter"
                             else "qwen3:8b-q4_K_M")
            cleanup_model = args.cleanup_model or default_model
            thinking = bool(args.cleanup_reasoning)
            print(f"Cleaning up OCR artifacts with {cleanup_model} ({backend})...")
            entries = cleanup_with_llm(entries, model=cleanup_model, languages=args.languages,
                                     thinking=thinking, backend=backend)
            # Fuzzy dedup: merge consecutive near-duplicate entries the LLM missed
            entries = deduplicate(entries, max_dist=2)
        write_srt(entries, output_path)
        print(f"Written {len(entries)} subtitle entries to {output_path}")


if __name__ == "__main__":
    main()
