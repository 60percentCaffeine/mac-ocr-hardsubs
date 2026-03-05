"""Sanity check SRT files after OCR + cleanup pipeline."""
import re
import sys
from pathlib import Path


def count_entries(path):
    text = path.read_text(encoding="utf-8")
    return len(re.findall(r"^\d+$", text, re.MULTILINE))


def check_timestamps_ordered(path):
    text = path.read_text(encoding="utf-8")
    times = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", text)
    issues = []
    for i, (start, end) in enumerate(times, 1):
        if start >= end:
            issues.append(f"  Entry {i}: start ({start}) >= end ({end})")
        if i > 1 and start < times[i - 2][1]:
            issues.append(f"  Entry {i}: overlaps with previous (starts {start}, prev ends {times[i-2][1]})")
    return issues


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "test"
    raw_path = Path(f"{base}.raw.srt")
    clean_path = Path(f"{base}.srt")

    ok = True

    for p in (raw_path, clean_path):
        if not p.exists():
            print(f"FAIL: {p} not found")
            ok = False

    if not ok:
        sys.exit(1)

    raw_count = count_entries(raw_path)
    clean_count = count_entries(clean_path)
    reduction = (1 - clean_count / raw_count) * 100 if raw_count else 0

    print(f"Raw:     {raw_count} entries ({raw_path})")
    print(f"Cleaned: {clean_count} entries ({clean_path})")
    print(f"Reduction: {reduction:.1f}%")

    if raw_count == 0:
        print("FAIL: raw file has 0 entries")
        ok = False
    if clean_count == 0:
        print("FAIL: cleaned file has 0 entries")
        ok = False
    if clean_count > raw_count:
        print("FAIL: cleaned has MORE entries than raw")
        ok = False
    if reduction > 60:
        print(f"WARN: reduction is {reduction:.1f}% (>60%), cleanup may be too aggressive")

    for label, path in [("Raw", raw_path), ("Cleaned", clean_path)]:
        issues = check_timestamps_ordered(path)
        if issues:
            print(f"FAIL: {label} has timestamp issues:")
            for issue in issues:
                print(issue)
            ok = False

    if ok:
        print("PASS")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
