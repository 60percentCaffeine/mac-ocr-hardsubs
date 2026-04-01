#!/usr/bin/env python3
"""Extract burned-in subtitles from anime videos into SRT using OCR."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path


MIN_CLIP_DURATION = 0.5  # seconds — clips shorter than this are dropped
MAX_SUBTITLE_DURATION = 7.0  # seconds — subtitles longer than this are capped


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


PREFILTER_EDGE_THRESHOLD = 2.0  # frames below this edge variance are skipped
FRAMEDIFF_THRESHOLD = 5.0  # frames with mean pixel diff below this reuse prev OCR


def _load_frame_gray(frame_path):
    """Load a frame as a grayscale float32 numpy array."""
    from PIL import Image
    import numpy as np

    return np.array(Image.open(frame_path).convert("L"), dtype=np.float32)


def _frame_has_edges(frame_path):
    """Fast heuristic: check if a frame has enough edges to plausibly contain text.

    Computes mean absolute gradient in both axes. Frames with subtitles have
    sharp edges (text outlines) that produce higher values. Frames below
    PREFILTER_EDGE_THRESHOLD are almost certainly empty (solid/gradient background).
    """
    import numpy as np

    img = _load_frame_gray(frame_path)
    dx = np.abs(np.diff(img, axis=1))
    dy = np.abs(np.diff(img, axis=0))
    return float(np.mean(dx) + np.mean(dy)) >= PREFILTER_EDGE_THRESHOLD


def _frames_are_similar(prev_arr, curr_arr):
    """Check if two frames are similar enough to reuse the previous OCR result."""
    import numpy as np

    return float(np.mean(np.abs(curr_arr - prev_arr))) < FRAMEDIFF_THRESHOLD


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


def ocr_frame_apple(frame_path, languages, fast):
    """Run Apple Vision OCR on a single frame, return recognized text."""
    import Vision
    from Foundation import NSData

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


def _apple_ocr_concat(frame_paths, languages):
    """OCR multiple frames in one Vision call by concatenating vertically.

    Returns a list of text strings, one per input frame.  Uses bounding-box
    Y coordinates to assign lines back to their source frame.  Vision uses
    normalised coordinates with Y=0 at the bottom, so we invert when mapping.
    """
    import Vision
    from Foundation import NSData
    from PIL import Image
    import io

    if not frame_paths:
        return []
    if len(frame_paths) == 1:
        return [ocr_frame_apple(frame_paths[0], languages, fast=False).strip()]

    # Load and normalize widths
    images = [Image.open(p) for p in frame_paths]
    target_w = images[0].width
    resized = []
    frame_heights = []
    for img in images:
        if img.width != target_w:
            ratio = target_w / img.width
            img = img.resize((target_w, int(img.height * ratio)), Image.LANCZOS)
        resized.append(img)
        frame_heights.append(img.height)

    separator_h = 20  # black bar between frames to prevent text bleeding
    total_h = sum(frame_heights) + separator_h * (len(resized) - 1)
    concat = Image.new("RGB", (target_w, total_h))  # black by default
    y_off = 0
    boundaries = []
    for img, fh in zip(resized, frame_heights):
        concat.paste(img, (0, y_off))
        boundaries.append((y_off, y_off + fh))
        y_off += fh + separator_h
    for img in images:
        img.close()

    # Convert concat to PNG bytes in memory for Vision
    buf = io.BytesIO()
    concat.save(buf, format="PNG")
    concat.close()
    png_bytes = buf.getvalue()

    data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
    if data is None:
        return [""] * len(frame_paths)

    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(0)
    request.setRecognitionLanguages_(languages)
    handler.performRequests_error_([request], None)

    results = request.results()
    if not results:
        return [""] * len(frame_paths)

    # Assign each observation to a frame based on bounding box Y
    frame_lines = [[] for _ in frame_paths]
    for obs in results:
        candidate = obs.topCandidates_(1)
        if not candidate:
            continue
        text = candidate[0].string()
        if not text.strip():
            continue
        bbox = obs.boundingBox()
        # Vision Y=0 is bottom; convert to pixel Y from top
        center_y_norm = 1.0 - (bbox.origin.y + bbox.size.height / 2)
        center_y_px = center_y_norm * total_h
        for fi, (ys, ye) in enumerate(boundaries):
            if center_y_px < ye:
                frame_lines[fi].append(text)
                break
        else:
            frame_lines[-1].append(text)

    return ["\n".join(lines) for lines in frame_lines]


def _apple_worker_ocr(task):
    """Worker function: OCR one or more frames with Apple Vision.

    Accepts (chunk_indices, paths, languages).  When multiple paths are given
    the frames are concatenated into a single image for one OCR call.
    """
    chunk_indices, frame_paths, languages = task
    texts = _apple_ocr_concat(frame_paths, languages)
    return chunk_indices, [t.strip() for t in texts]


def _apple_warmup(languages):
    """Warm up a worker by importing Vision framework."""
    import Vision  # noqa: F401
    return True


def detect_text_frames_apple(frames, languages, num_workers=None, batch_size=16):
    """OCR frames using Apple Vision with parallel processes for speed.

    Uses multiprocessing (spawn) so each worker gets its own Vision
    framework instance with full CPU parallelism.  Workers warm up
    while the main process pre-filters frames.

    When batch_size > 1, frames are concatenated into single images so
    each Vision call processes multiple frames at once.
    """
    import numpy as np
    from multiprocessing import get_context

    if num_workers is None:
        num_workers = max(os.cpu_count() or 1, 1)

    # Start worker pool and warm up in background while we pre-filter
    ctx = get_context("spawn")
    pool = ctx.Pool(processes=num_workers)
    warmup_results = [pool.apply_async(_apple_warmup, (languages,)) for _ in range(num_workers)]

    # Pre-filter: skip frames that clearly have no text
    print("Pre-filtering frames...", end="", flush=True)
    candidates = set()
    frame_arrays = {}
    for idx, frame in enumerate(frames):
        arr = _load_frame_gray(frame)
        frame_arrays[idx] = arr
        dx = np.abs(np.diff(arr, axis=1))
        dy = np.abs(np.diff(arr, axis=0))
        if float(np.mean(dx) + np.mean(dy)) >= PREFILTER_EDGE_THRESHOLD:
            candidates.add(idx)
    edge_skipped = len(frames) - len(candidates)
    print(f" {edge_skipped}/{len(frames)} frames skipped (no text edges)")

    # Build list of frames that need OCR (applying frame differencing)
    texts = [""] * len(frames)
    ocr_indices = []
    diff_reused = 0
    last_ocr_arr = None
    for idx in range(len(frames)):
        if idx not in candidates:
            last_ocr_arr = None
        elif last_ocr_arr is not None and _frames_are_similar(last_ocr_arr, frame_arrays[idx]):
            diff_reused += 1
        else:
            ocr_indices.append(idx)
            last_ocr_arr = frame_arrays[idx]

    if diff_reused:
        print(f"Frame differencing will reuse OCR for {diff_reused} frames")

    # Wait for all workers to finish warming up
    for r in warmup_results:
        r.get()

    total_ocr = len(ocr_indices)
    ocr_paths = [str(frames[i]) for i in ocr_indices]

    # Build tasks: chunk frames into batches for concatenated OCR
    bs = max(1, batch_size)
    ocr_tasks = []
    for i in range(0, total_ocr, bs):
        chunk_indices = list(range(i, min(i + bs, total_ocr)))
        chunk_paths = [ocr_paths[j] for j in chunk_indices]
        ocr_tasks.append((chunk_indices, chunk_paths, languages))

    batch_label = f", concat {bs}" if bs > 1 else ""
    print(f"OCR: 0% (0/{total_ocr} frames, {num_workers} workers{batch_label})", end="", flush=True)
    done = 0
    for chunk_indices, chunk_texts in pool.imap_unordered(_apple_worker_ocr, ocr_tasks):
        for ci, text in zip(chunk_indices, chunk_texts):
            texts[ocr_indices[ci]] = text
        done += len(chunk_indices)
        progress = done / total_ocr * 100 if total_ocr else 100
        print(f"\rOCR: {progress:.0f}% ({done}/{total_ocr} frames, {num_workers} workers{batch_label})", end="", flush=True)
    pool.close()
    pool.join()

    # Propagate OCR results to diff-reused frames
    ocr_set = set(ocr_indices)
    last_ocr_text = ""
    last_ocr_arr = None
    for idx in range(len(frames)):
        if idx not in candidates:
            last_ocr_text = ""
            last_ocr_arr = None
        elif idx in ocr_set:
            last_ocr_text = texts[idx]
            last_ocr_arr = frame_arrays[idx]
        else:
            if last_ocr_arr is not None:
                texts[idx] = last_ocr_text

    print()
    if diff_reused:
        print(f"Frame differencing reused OCR for {diff_reused} frames")
    frame_arrays.clear()
    return texts


# ── Chrome Screen AI backend ─────────────────────────────────────────────

_screenai_engine = None


def _load_screenai():
    """Load the Chrome Screen AI native library (cached after first call).

    Model files must already be present at ~/.config/screen_ai/resources/
    (call _ensure_screenai_downloaded() first when running in the main process).
    """
    global _screenai_engine
    if _screenai_engine is not None:
        return _screenai_engine

    import ctypes

    model_dir = Path.home() / ".config" / "screen_ai" / "resources"
    if not model_dir.exists():
        raise RuntimeError(
            "Screen AI model files not found. Run _ensure_screenai_downloaded() first."
        )

    dll_name = "chrome_screen_ai.dll" if sys.platform == "win32" else "libchromescreenai.so"
    dll_mode = os.RTLD_LAZY if hasattr(os, "RTLD_LAZY") else ctypes.DEFAULT_MODE
    lib = ctypes.CDLL(str(model_dir / dll_name), mode=dll_mode)

    # ── ctypes struct definitions (mirrors Skia's SkBitmap) ─────────────
    class SkColorInfo(ctypes.Structure):
        _fields_ = [("fColorSpace", ctypes.c_void_p), ("fColorType", ctypes.c_int32), ("fAlphaType", ctypes.c_int32)]

    class SkISize(ctypes.Structure):
        _fields_ = [("fWidth", ctypes.c_int32), ("fHeight", ctypes.c_int32)]

    class SkImageInfo(ctypes.Structure):
        _fields_ = [("fColorInfo", SkColorInfo), ("fDimensions", SkISize)]

    class SkPixmap(ctypes.Structure):
        _fields_ = [("fPixels", ctypes.c_void_p), ("fRowBytes", ctypes.c_size_t), ("fInfo", SkImageInfo)]

    class SkBitmap(ctypes.Structure):
        _fields_ = [("fPixelRef", ctypes.c_void_p), ("fPixmap", SkPixmap), ("fFlags", ctypes.c_uint32)]

    # ── file-content callbacks (library reads its own weight files) ──────
    @ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_char_p)
    def get_file_content_size(p):
        path = model_dir / p.decode("utf-8")
        return os.path.getsize(path) if path.exists() else 0

    @ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p)
    def get_file_content(p, s, ptr):
        path = model_dir / p.decode("utf-8")
        if path.exists():
            with open(path, "rb") as f:
                ctypes.memmove(ptr, f.read(s), s)

    # ── configure function signatures ───────────────────────────────────
    lib.SetFileContentFunctions.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.InitOCRUsingCallback.restype = ctypes.c_bool
    lib.SetOCRLightMode.argtypes = [ctypes.c_bool]
    lib.PerformOCR.argtypes = [ctypes.POINTER(SkBitmap), ctypes.POINTER(ctypes.c_uint32)]
    lib.PerformOCR.restype = ctypes.c_void_p
    lib.FreeLibraryAllocatedCharArray.argtypes = [ctypes.c_void_p]
    lib.GetMaxImageDimension.restype = ctypes.c_uint32

    lib.SetFileContentFunctions(get_file_content_size, get_file_content)
    lib.InitOCRUsingCallback()
    lib.SetOCRLightMode(False)
    max_pixel_size = lib.GetMaxImageDimension()

    _screenai_engine = {
        "lib": lib,
        "SkBitmap": SkBitmap,
        "max_pixel_size": max_pixel_size,
        # prevent garbage-collection of the callback pointers
        "_cb_size": get_file_content_size,
        "_cb_content": get_file_content,
    }
    print("Chrome Screen AI loaded.", flush=True)
    return _screenai_engine


def _screenai_perform_ocr(engine, img):
    """Run Screen AI OCR on a PIL Image, return parsed protobuf response dict."""
    import ctypes
    from screenai_protos.chrome_screen_ai_pb2 import VisualAnnotation
    from google.protobuf.json_format import MessageToDict

    lib = engine["lib"]
    SkBitmap = engine["SkBitmap"]
    max_px = engine["max_pixel_size"]

    # down-scale if needed
    if any(x > max_px for x in img.size):
        factor = min(max_px / img.width, max_px / img.height)
        img = img.resize((int(img.width * factor), int(img.height * factor)), 3)  # LANCZOS=3

    img_bytes = img.convert("RGBA").tobytes()
    w, h = img.size

    bitmap = SkBitmap()
    bitmap.fPixmap.fPixels = ctypes.cast(ctypes.c_char_p(img_bytes), ctypes.c_void_p)
    bitmap.fPixmap.fRowBytes = w * 4
    bitmap.fPixmap.fInfo.fColorInfo.fColorType = 4
    bitmap.fPixmap.fInfo.fColorInfo.fAlphaType = 1
    bitmap.fPixmap.fInfo.fDimensions.fWidth = w
    bitmap.fPixmap.fInfo.fDimensions.fHeight = h

    output_length = ctypes.c_uint32(0)
    result_ptr = lib.PerformOCR(ctypes.byref(bitmap), ctypes.byref(output_length))
    if not result_ptr:
        return {}

    proto_bytes = ctypes.string_at(result_ptr, output_length.value)
    lib.FreeLibraryAllocatedCharArray(result_ptr)

    annotation = VisualAnnotation()
    annotation.ParseFromString(proto_bytes)
    return MessageToDict(annotation, preserving_proto_field_name=True)


def _screenai_ocr_image(engine, img):
    """Run Screen AI OCR on a PIL Image, return recognized text string."""
    response = _screenai_perform_ocr(engine, img)
    lines = []
    for line in response.get("lines", []):
        text = line.get("utf8_string", "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _screenai_ocr_concat(engine, images):
    """OCR multiple images in one call by concatenating them vertically.

    Returns a list of text strings, one per input image.  Uses bounding-box
    Y coordinates from the protobuf response to assign lines to their source
    frame within the concatenated image.

    If the concatenated height would exceed max_pixel_size, the batch is split
    into sub-batches that each fit within the limit to avoid quality-destroying
    downscale.
    """
    from PIL import Image

    if not images:
        return []
    if len(images) == 1:
        return [_screenai_ocr_image(engine, images[0])]

    max_px = engine["max_pixel_size"]
    target_w = images[0].width

    # Normalize widths and collect heights
    resized = []
    frame_heights = []
    for img in images:
        if img.width != target_w:
            ratio = target_w / img.width
            img = img.resize((target_w, int(img.height * ratio)), 3)
        resized.append(img)
        frame_heights.append(img.height)

    # Split into sub-batches that fit within max_pixel_size
    sub_batches = []  # list of (start_idx, end_idx) into resized/frame_heights
    batch_start = 0
    batch_h = 0
    for i, fh in enumerate(frame_heights):
        if batch_h + fh > max_px and i > batch_start:
            sub_batches.append((batch_start, i))
            batch_start = i
            batch_h = 0
        batch_h += fh
    sub_batches.append((batch_start, len(resized)))

    all_results = [""] * len(images)
    for sb_start, sb_end in sub_batches:
        sb_images = resized[sb_start:sb_end]
        sb_heights = frame_heights[sb_start:sb_end]

        if len(sb_images) == 1:
            all_results[sb_start] = _screenai_ocr_image(engine, sb_images[0])
            continue

        total_h = sum(sb_heights)
        concat = Image.new("RGB", (target_w, total_h))
        y_off = 0
        boundaries = []
        for img, fh in zip(sb_images, sb_heights):
            concat.paste(img, (0, y_off))
            boundaries.append((y_off, y_off + fh))
            y_off += fh

        response = _screenai_perform_ocr(engine, concat)
        concat.close()

        # Assign each line to a frame based on bounding box Y center
        frame_lines = [[] for _ in sb_images]
        for line in response.get("lines", []):
            text = line.get("utf8_string", "").strip()
            if not text:
                continue
            bbox = line.get("bounding_box", {})
            line_y = bbox.get("y", 0) + bbox.get("height", 0) / 2
            for fi, (ys, ye) in enumerate(boundaries):
                if line_y < ye:
                    frame_lines[fi].append(text)
                    break
            else:
                frame_lines[-1].append(text)

        for i, lines in enumerate(frame_lines):
            all_results[sb_start + i] = "\n".join(lines)

    return all_results


def ocr_frame_screenai(frame_path, languages, fast):
    """Run Chrome Screen AI OCR on a single frame, return recognized text."""
    from PIL import Image

    _ensure_screenai_downloaded()
    engine = _load_screenai()
    img = Image.open(frame_path)
    text = _screenai_ocr_image(engine, img)
    img.close()
    return text


def _screenai_worker_ocr(task):
    """Worker function: OCR one or more frames, return (indices, texts).

    Accepts (indices, paths) tuple.  When multiple paths are given the frames
    are concatenated into a single image for one PerformOCR call, and results
    are split back by bounding-box Y coordinate.
    """
    indices, frame_paths = task
    from PIL import Image
    # Suppress noisy native-library logging in workers
    if not getattr(_screenai_worker_ocr, "_stderr_suppressed", False):
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        _screenai_worker_ocr._stderr_suppressed = True
    engine = _load_screenai()
    images = [Image.open(p) for p in frame_paths]
    texts = _screenai_ocr_concat(engine, images)
    for img in images:
        img.close()
    return indices, [t.strip() for t in texts]



def _ensure_screenai_protos():
    """Generate screenai_protos Python bindings if missing."""
    proto_pkg = Path(__file__).parent / "screenai_protos"
    pb2_file = proto_pkg / "chrome_screen_ai_pb2.py"
    proto_file = proto_pkg / "chrome_screen_ai.proto"
    if pb2_file.exists():
        return
    if not proto_file.exists():
        sys.exit("Error: screenai_protos/chrome_screen_ai.proto not found")
    project_root = str(Path(__file__).parent)
    # Try grpc_tools first (pure Python, no system dependency)
    try:
        from grpc_tools import protoc as grpc_protoc
        rc = grpc_protoc.main(["grpc_tools.protoc", f"--proto_path={project_root}",
                               f"--python_out={project_root}",
                               str(proto_file.relative_to(project_root))])
        if rc == 0 and pb2_file.exists():
            return
    except ImportError:
        pass
    # Fall back to system protoc
    protoc_bin = shutil.which("protoc")
    if protoc_bin:
        try:
            subprocess.run([protoc_bin, f"--proto_path={project_root}",
                            f"--python_out={project_root}",
                            str(proto_file.relative_to(project_root))],
                           check=True, capture_output=True)
            if pb2_file.exists():
                return
        except subprocess.CalledProcessError:
            pass
    sys.exit("Error: could not generate screenai protobuf bindings. "
             "Install protoc (e.g. brew install protobuf) or "
             "pip install grpcio-tools, then retry.")


def _ensure_screenai_downloaded():
    """Download Screen AI model files if needed (call before spawning workers)."""
    import platform

    _ensure_screenai_protos()

    model_dir = Path.home() / ".config" / "screen_ai" / "resources"
    if model_dir.exists():
        return

    print("Downloading Chrome Screen AI model files...", flush=True)

    os_name = platform.system().lower()
    arch = platform.machine().lower()
    if os_name == "darwin":
        os_name = "mac"
    if arch in ("x86_64", "amd64"):
        arch = "amd64"
    elif arch in ("aarch64", "arm64"):
        arch = "arm64"
    elif arch in ("x86", "i386", "i686"):
        arch = "386"

    cipd_platform = "linux" if os_name == "linux" else f"{os_name}-{arch}"
    package_name = f"chromium/third_party/screen-ai/{cipd_platform}"
    ensure_content = f"{package_name} latest\n"

    with tempfile.TemporaryDirectory() as td:
        cipd_bin = "cipd.exe" if sys.platform == "win32" else "cipd"
        cipd_path = os.path.join(td, cipd_bin)
        cipd_client_platform = f"{os_name}-{arch}"
        cipd_url = f"https://chrome-infra-packages.appspot.com/client?platform={cipd_client_platform}&version=latest"
        try:
            urllib.request.urlretrieve(cipd_url, cipd_path)
            if sys.platform != "win32":
                os.chmod(cipd_path, 0o755)
        except Exception as exc:
            sys.exit(f"Error: failed to download CIPD client: {exc}")
        target_path = model_dir.parent
        cmd = [cipd_path, "export", "-root", str(target_path), "-ensure-file", "-"]
        try:
            subprocess.run(cmd, input=ensure_content, text=True, check=True)
        except Exception as exc:
            sys.exit(f"Error: failed to download Screen AI files: {exc}")

    print("Screen AI model files downloaded.", flush=True)


def _screenai_warmup(_=None):
    """Warm up a worker by loading the Screen AI engine and suppressing stderr."""
    if not getattr(_screenai_warmup, "_stderr_suppressed", False):
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        _screenai_warmup._stderr_suppressed = True
    _load_screenai()
    return True


def detect_text_frames_screenai(frames, languages, num_workers=None, batch_size=1):
    """OCR frames using Chrome Screen AI with parallel processes for speed.

    Each worker process loads its own copy of the native library to avoid
    thread-safety issues.  Pre-filtering and frame-differencing reduce the
    total number of OCR calls.

    When batch_size > 1, frames are concatenated into single images so each
    PerformOCR call processes multiple frames at once, reducing native library
    call overhead.  Workers are started and warmed up in parallel with
    pre-filtering so the two costly startup phases overlap.
    """
    import numpy as np
    from multiprocessing import get_context

    if num_workers is None:
        num_workers = max(os.cpu_count() or 1, 1)

    # Ensure model files are downloaded before spawning workers
    _ensure_screenai_downloaded()

    # Start worker pool and warm up engines in background while we pre-filter
    ctx = get_context("spawn")
    pool = ctx.Pool(processes=num_workers)
    warmup_results = [pool.apply_async(_screenai_warmup) for _ in range(num_workers)]

    # Pre-filter: skip frames that clearly have no text (runs in main process
    # concurrently with worker engine loading)
    print("Pre-filtering frames...", end="", flush=True)
    candidates = set()
    frame_arrays = {}
    for idx, frame in enumerate(frames):
        arr = _load_frame_gray(frame)
        frame_arrays[idx] = arr
        dx = np.abs(np.diff(arr, axis=1))
        dy = np.abs(np.diff(arr, axis=0))
        if float(np.mean(dx) + np.mean(dy)) >= PREFILTER_EDGE_THRESHOLD:
            candidates.add(idx)
    edge_skipped = len(frames) - len(candidates)
    print(f" {edge_skipped}/{len(frames)} frames skipped (no text edges)")

    # Build list of frames that need OCR (applying frame differencing)
    texts = [""] * len(frames)
    ocr_indices = []
    diff_reused = 0
    last_ocr_arr = None
    for idx in range(len(frames)):
        if idx not in candidates:
            last_ocr_arr = None
        elif last_ocr_arr is not None and _frames_are_similar(last_ocr_arr, frame_arrays[idx]):
            diff_reused += 1
        else:
            ocr_indices.append(idx)
            last_ocr_arr = frame_arrays[idx]

    if diff_reused:
        print(f"Frame differencing will reuse OCR for {diff_reused} frames")

    # Wait for all workers to finish warming up before submitting OCR tasks
    for r in warmup_results:
        r.get()

    total_ocr = len(ocr_indices)
    ocr_paths = [str(frames[i]) for i in ocr_indices]

    # Build tasks: chunk frames into batches for concatenated OCR
    bs = max(1, batch_size)
    ocr_tasks = []
    for i in range(0, total_ocr, bs):
        chunk_indices = list(range(i, min(i + bs, total_ocr)))
        chunk_paths = [ocr_paths[j] for j in chunk_indices]
        ocr_tasks.append((chunk_indices, chunk_paths))

    batch_label = f", concat {bs}" if bs > 1 else ""
    print(f"OCR: 0% (0/{total_ocr} frames, {num_workers} workers{batch_label})", end="", flush=True)
    done = 0
    for task_indices, task_texts in pool.imap_unordered(_screenai_worker_ocr, ocr_tasks):
        for ti, text in zip(task_indices, task_texts):
            texts[ocr_indices[ti]] = text
        done += len(task_indices)
        progress = done / total_ocr * 100 if total_ocr else 100
        print(f"\rOCR: {progress:.0f}% ({done}/{total_ocr} frames, {num_workers} workers{batch_label})", end="", flush=True)
    pool.close()
    pool.join()

    # Propagate OCR results to diff-reused frames
    ocr_set = set(ocr_indices)
    last_ocr_text = ""
    last_ocr_arr = None
    for idx in range(len(frames)):
        if idx not in candidates:
            last_ocr_text = ""
            last_ocr_arr = None
        elif idx in ocr_set:
            last_ocr_text = texts[idx]
            last_ocr_arr = frame_arrays[idx]
        else:
            if last_ocr_arr is not None:
                texts[idx] = last_ocr_text

    print()
    if diff_reused:
        print(f"Frame differencing reused OCR for {diff_reused} frames")
    frame_arrays.clear()
    return texts


def format_timestamp(seconds):
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _tokenize_cjk(text):
    """Split text into tokens: each CJK character is its own token,
    consecutive non-CJK characters form a single token."""
    tokens = []
    buf = []
    for c in text:
        if "\u4E00" <= c <= "\u9FFF" or "\uF900" <= c <= "\uFAFF" or "\U00020000" <= c <= "\U0002A6DF":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(c)
        else:
            buf.append(c)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _edit_distance(a, b):
    """Compute Levenshtein edit distance on CJK-aware tokens.

    Each CJK character (Simplified/Traditional) counts as one token.
    Consecutive non-CJK characters are grouped into a single token.
    This way 'is正在殺時間' and '正在殺時間' differ by 1 (the 'is' run),
    but 'is正在殺is時間' and '正在殺時間' differ by 2.
    """
    ta = _tokenize_cjk(a)
    tb = _tokenize_cjk(b)
    if len(ta) < len(tb):
        ta, tb = tb, ta
    if not tb:
        return len(ta)
    prev = list(range(len(tb) + 1))
    for i, ca in enumerate(ta):
        curr = [i + 1]
        for j, cb in enumerate(tb):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


_OUTER_BRACKET_PAIRS = [
    ("（", "）"), ("(", ")"),
    ("【", "】"), ("《", "》"),
    ("「", "」"), ("『", "』"),
    ("〈", "〉"), ("＜", "＞"),
    ("〔", "〕"), ("｛", "｝"),
    ("﹁", "﹂"), ("﹃", "﹄"),
]


def _strip_outer_parens(text):
    """Strip outer brackets/parentheses for comparison."""
    for opener, closer in _OUTER_BRACKET_PAIRS:
        if text.startswith(opener) and text.endswith(closer):
            return text[1:-1]
    return text


def deduplicate(entries, max_dist=0):
    """Merge consecutive entries with identical (or near-identical) text."""
    if not entries:
        return []

    merged = []
    current_text, current_start, current_end = entries[0]

    for text, start, end in entries[1:]:
        if text == current_text:
            current_end = end
        elif max_dist > 0:
            # Compare with outer parentheses stripped so e.g.
            # "ひ在殺時間" vs "（正在殺時間）" can merge (distance on inner text)
            a = _strip_outer_parens(current_text)
            b = _strip_outer_parens(text)
            if _edit_distance(a, b) <= max_dist:
                # Keep the longer (likely more accurate) text
                if len(text) > len(current_text):
                    current_text = text
                current_end = end
            else:
                merged.append((current_text, current_start, current_end))
                current_text, current_start, current_end = text, start, end
        else:
            merged.append((current_text, current_start, current_end))
            current_text, current_start, current_end = text, start, end

    merged.append((current_text, current_start, current_end))
    return merged


def _texts_similar(a, b):
    """Check if two texts are similar enough to merge a single-frame glitch."""
    # Containment: one text is a substring of the other (handles extra OCR noise)
    if a in b or b in a:
        return True
    # Edit distance with generous threshold
    max_len = max(len(a), len(b))
    threshold = max(2, int(max_len * 0.4))
    if _edit_distance(a, b) <= threshold:
        return True
    return False


def smart_deduplicate(entries, sample_interval):
    """Merge single-frame entries into a neighbor when texts are similar.

    If a subtitle spans only one frame and is similar to the previous or next
    subtitle, it is likely an OCR glitch — merge it into the neighbor.
    """
    if not entries:
        return []

    n = len(entries)
    skip = set()

    # Forward pass: merge single-frame entries into previous
    merged = []
    current_text, current_start, current_end = entries[0]

    for i, (text, start, end) in enumerate(entries[1:], 1):
        duration = round(end - start, 6)
        is_single_frame = abs(duration - sample_interval) < 1e-4
        if is_single_frame and _texts_similar(text, current_text):
            current_end = end
            skip.add(i)
            continue
        merged.append((current_text, current_start, current_end))
        current_text, current_start, current_end = text, start, end

    merged.append((current_text, current_start, current_end))

    # Backward pass: merge remaining single-frame entries into next
    result = []
    i = 0
    while i < len(merged):
        text, start, end = merged[i]
        duration = round(end - start, 6)
        is_single_frame = abs(duration - sample_interval) < 1e-4
        if is_single_frame and i + 1 < len(merged):
            next_text = merged[i + 1][1]
            next_entry_text = merged[i + 1][0]
            if _texts_similar(text, next_entry_text):
                # Extend next entry's start to absorb this one
                merged[i + 1] = (merged[i + 1][0], start, merged[i + 1][2])
                i += 1
                continue
        result.append((text, start, end))
        i += 1

    return result




# ---------------------------------------------------------------------------
# Rules-based cleanup (no LLM)
# ---------------------------------------------------------------------------

# Characters that legitimately appear in CJK subtitles.  Anything outside
# this set is considered OCR noise and stripped from each line.
_SUBTITLE_WHITELIST_RE = re.compile(
    r"["
    r"\u0020-\u007E"       # ASCII (letters, digits, basic punctuation)
    r"\u00C0-\u024F"       # Latin Extended (accented letters)
    r"\u2000-\u206F"       # General Punctuation (—, –, ', ', ", ", …, etc.)
    r"\u2150-\u218F"       # Number Forms (fractions)
    r"\u2190-\u21FF"       # Arrows
    r"\u22EF"              # ⋯ midline ellipsis
    r"\u2500-\u257F"       # Box Drawing
    r"\u3000-\u303F"       # CJK Symbols & Punctuation (　、。〈〉《》「」『』【】〜)
    r"\u3040-\u309F"       # Hiragana
    r"\u30A0-\u30FF"       # Katakana
    r"\u4E00-\u9FFF"       # CJK Unified Ideographs
    r"\uF900-\uFAFF"       # CJK Compatibility Ideographs
    r"\uFE30-\uFE4F"       # CJK Compatibility Forms
    r"\uFF00-\uFFEF"       # Halfwidth & Fullwidth Forms (！＂＃, ０-９, Ａ-Ｚ, etc.)
    r"\U00020000-\U0002A6DF"  # CJK Unified Ideographs Extension B (rare chars)
    r"]"
)

# Hiragana U+3040-309F, Katakana U+30A0-30FF
_KANA_RE = re.compile(r"[\u3040-\u30FF]")


def _is_mostly_kana(text):
    """Return True if text is predominantly Japanese kana/punctuation."""
    stripped = re.sub(r"[\s.,!?。、！？…⋯\-()（）「」『』【】]", "", text)
    if not stripped:
        return True
    kana_count = sum(1 for c in stripped if "\u3040" <= c <= "\u30FF")
    return kana_count / len(stripped) >= 0.5


def _strip_non_whitelist(text):
    """Remove characters not in the subtitle whitelist."""
    return "".join(c for c in text if _SUBTITLE_WHITELIST_RE.match(c))


def _fix_punctuation(text):
    """Fix common OCR punctuation errors."""
    # Normalize various ellipsis forms to ⋯
    text = re.sub(r"\.{2,}", "⋯", text)
    text = re.sub(r"。{2,}", "⋯", text)
    text = re.sub(r"…+", "⋯", text)
    # .0, .00, …00 etc. at end → ⋯  (Screen AI reads ⋯ as .00 sometimes)
    text = re.sub(r"[.…⋯]0+$", "⋯", text)
    # 。0 or trailing 0 at end of sentence → ⋯
    text = re.sub(r"。0", "⋯", text)
    # •。 or ·。 or stray dot combinations → ⋯
    text = re.sub(r"[•·]。", "⋯", text)
    text = re.sub(r"[•·]{2,}", "⋯", text)
    # Strip trailing punctuation after ⋯ (e.g. ⋯。 ⋯. ⋯, → ⋯)
    text = re.sub(r"⋯[.。,，、;；!！?？]+", "⋯", text)
    # Normalize curly quotes: treat any pair of U+201C/U+201D (in any combo)
    # as matched quotes — OCR often uses the same codepoint for both.
    curly_count = text.count("\u201C") + text.count("\u201D")
    if curly_count == 2:
        # Matched pair — normalize first to U+201C, second to U+201D
        first = True
        chars = []
        for c in text:
            if c in "\u201C\u201D":
                chars.append("\u201C" if first else "\u201D")
                first = False
            else:
                chars.append(c)
        text = "".join(chars)
    elif curly_count == 1:
        # Stray single curly quote as hesitation marker → ⋯
        text = text.replace("\u201C", "⋯").replace("\u201D", "⋯")
    # Stray single straight quote as hesitation marker (e.g. 我"我 → 我⋯我)
    if text.count('"') == 1:
        text = text.replace('"', "⋯")

    # Normalize half-width punctuation to full-width in CJK context
    # Comma: CJK,CJK or CJK, CJK → CJK，CJK
    text = re.sub(
        r"(?<=[\u3000-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF]),\s?(?=[\u3000-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF])",
        "，",
        text,
    )
    # Parentheses containing CJK → full-width
    text = re.sub(
        r"\(([^)]*[\u3000-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF][^)]*)\)",
        r"（\1）",
        text,
    )
    # Straight quotes containing CJK → fullwidth quotes
    text = re.sub(
        r"\"([^\"]*[\u3000-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF][^\"]*?)\"",
        r"「\1」",
        text,
    )
    # Curly quotes containing CJK → fullwidth quotes
    text = re.sub(
        r"\u201C([^\u201D]*[\u3000-\u9FFF\uF900-\uFAFF\U00020000-\U0002A6DF][^\u201D]*?)\u201D",
        r"「\1」",
        text,
    )
    # Strip orphan brackets/parens without matching counterpart
    for opener, closer in _OUTER_BRACKET_PAIRS:
        if closer in text and opener not in text:
            text = text.replace(closer, "")
        if opener in text and closer not in text:
            text = text.replace(opener, "")
    return text


def _clean_line(line):
    """Clean a single subtitle line. Returns cleaned line or None to remove."""
    line = line.strip()
    if not line:
        return None

    # Strip characters outside the subtitle whitelist (Arabic, Cyrillic, etc.)
    line = _strip_non_whitelist(line).strip()
    if not line:
        return None

    # Pure kana line → remove
    if _is_mostly_kana(line):
        return None

    # Fix punctuation
    line = _fix_punctuation(line)

    return line if line else None


def cleanup_rules(entries, languages=None):
    """Clean OCR artifacts using deterministic rules. Returns filtered entries."""
    if not entries:
        return entries

    if languages is None:
        languages = ["zh-Hant"]

    result = []
    for text, start, end in entries:
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            cleaned = _clean_line(line)
            if cleaned is not None:
                cleaned_lines.append(cleaned)

        if not cleaned_lines:
            continue

        new_text = "\n".join(cleaned_lines)
        result.append((new_text, start, end))

    return result




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
                if new_end - start <= MAX_SUBTITLE_DURATION:
                    result[best] = (text, start, new_end)

    return result


def cap_durations(entries, max_duration=MAX_SUBTITLE_DURATION):
    """Clamp subtitle durations to a maximum length."""
    result = []
    for text, start, end in entries:
        if end - start > max_duration:
            end = start + max_duration
        result.append((text, start, end))
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
    parser.add_argument("video", nargs="+", help="Input video file path(s)")
    parser.add_argument(
        "--scan-backend",
        default="screenai",
        choices=["apple", "screenai"],
        help="OCR backend: apple (macOS Vision) or screenai (Chrome Screen AI) (default: screenai)",
    )
    parser.add_argument("-o", "--output", help="Output SRT path (default: <video>.srt), or output folder when processing multiple files")
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
        default="rules",
        choices=["none", "rules"],
        help="Cleanup backend (default: rules): rules (local, no LLM), none (skip)",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=None,
        help="Batch size for apple/screenai backends (default: 16 for apple, 64 for screenai)",
    )
    parser.add_argument(
        "--scan-threads",
        type=int,
        default=None,
        help="Number of worker processes for apple/screenai backends (default: 1)",
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
        if scan_only_incompatible:
            parser.error(
                f"--scan-only is incompatible with: {', '.join(scan_only_incompatible)}"
            )

    multi = len(args.video) > 1
    if multi and args.output is not None and not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    # Validate all video files exist before starting
    for video_path in args.video:
        if not os.path.isfile(video_path):
            print(f"Error: video file not found: {video_path}", file=sys.stderr)
            sys.exit(1)

    for file_idx, video_path in enumerate(args.video):
        if multi:
            print(f"\n{'='*60}")
            print(f"[{file_idx + 1}/{len(args.video)}] {video_path}")
            print(f"{'='*60}")

        if multi and args.output is not None:
            basename = os.path.splitext(os.path.basename(video_path))[0] + ".srt"
            output_path = os.path.join(args.output, basename)
        else:
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

            # OCR all frames
            batch_size = args.scan_batch_size
            if args.scan_backend == "screenai":
                ocr_texts = detect_text_frames_screenai(frames, args.languages, num_workers=args.scan_threads, batch_size=batch_size or 64)
            elif args.scan_backend == "apple":
                ocr_texts = detect_text_frames_apple(frames, args.languages, num_workers=args.scan_threads, **({"batch_size": batch_size} if batch_size is not None else {}))
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
                continue

            # Build entries from OCR results within detected clips
            raw_entries = []
            for s, e in clips:
                for idx in range(s, e + 1):
                    timestamp = idx * sample_interval
                    text = ocr_texts[idx]
                    if text:
                        raw_entries.append((text, timestamp, timestamp + sample_interval))

            entries = deduplicate(raw_entries)
            before_smart = len(entries)
            entries = smart_deduplicate(entries, sample_interval)
            smart_merged = before_smart - len(entries)
            if smart_merged:
                print(f"Smart dedup merged {smart_merged} single-frame entries")
            if args.cleanup_backend == "rules":
                # Save raw (pre-cleanup) SRT alongside the final one
                raw_path = os.path.splitext(output_path)[0] + ".raw.srt"
                write_srt(entries, raw_path)
                print(f"Written {len(entries)} raw subtitle entries to {raw_path}")

                print("Cleaning up OCR artifacts with rules-based cleanup...")
                entries = cleanup_rules(entries, languages=args.languages)
                # Fuzzy dedup: merge consecutive near-duplicate entries the cleanup missed
                entries = deduplicate(entries, max_dist=2)
                entries = cap_durations(entries)
                # Ensure every frame in detected clips is covered by a subtitle
                entries = fill_clip_gaps(entries, clips, sample_interval)
            write_srt(entries, output_path)
            print(f"Written {len(entries)} subtitle entries to {output_path}")


if __name__ == "__main__":
    main()
