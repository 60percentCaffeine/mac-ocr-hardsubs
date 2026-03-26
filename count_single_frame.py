"""Count single-frame subtitle entries in an SRT file."""

import re
import sys


def parse_timestamp(ts):
    """Parse SRT timestamp to seconds."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def main():
    srt_path = sys.argv[1] if len(sys.argv) > 1 else "ocrtest.srt"
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    frame_dur = 1.0 / fps

    with open(srt_path) as f:
        content = f.read()

    blocks = re.split(r"\n\n+", content.strip())
    single = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        match = re.match(r"(.+) --> (.+)", lines[1])
        if not match:
            continue
        start = parse_timestamp(match.group(1).strip())
        end = parse_timestamp(match.group(2).strip())
        duration = end - start
        if abs(duration - frame_dur) < 1e-3:
            text = "\n".join(lines[2:])
            single.append((lines[0], lines[1], text))

    print(f"FPS: {fps}, frame duration: {frame_dur:.3f}s")
    print(f"Single-frame entries: {len(single)}/{len(blocks)}")
    print()
    for idx, ts, text in single:
        print(f"  #{idx}  {ts}")
        print(f"        {text}")
        print()


if __name__ == "__main__":
    main()
