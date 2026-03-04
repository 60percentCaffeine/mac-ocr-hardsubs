#!/usr/bin/env python3
"""Extract burned-in subtitles from anime videos into SRT using macOS Vision OCR."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import Vision
from Foundation import NSData


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


def deduplicate(entries):
    """Merge consecutive entries with identical text."""
    if not entries:
        return []

    merged = []
    current_text, current_start, current_end = entries[0]

    for text, start, end in entries[1:]:
        if text == current_text:
            current_end = end
        else:
            merged.append((current_text, current_start, current_end))
            current_text, current_start, current_end = text, start, end

    merged.append((current_text, current_start, current_end))
    return merged


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
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (os.path.splitext(video_path)[0] + ".srt")

    with tempfile.TemporaryDirectory(prefix="ocranime_") as tmpdir:
        print(f"Extracting frames (interval={args.interval}s)...")
        frames = extract_frames(video_path, tmpdir, args.interval, args.crop)
        print(f"Extracted {len(frames)} frames")

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
        write_srt(entries, output_path)
        print(f"Written {len(entries)} subtitle entries to {output_path}")


if __name__ == "__main__":
    main()
