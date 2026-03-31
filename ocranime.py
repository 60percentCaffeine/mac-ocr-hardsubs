#!/usr/bin/env python3
"""Extract burned-in subtitles from anime videos into SRT using OCR."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

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


def detect_text_frames(frames, languages, scan_backend="apple"):
    """OCR all frames, return list of recognized text (empty string if no text).

    Uses accurate mode because fast mode doesn't reliably detect CJK text.
    Results are cached so the accurate OCR pass can reuse them.

    Two cheap pre-filters avoid expensive OCR calls:
    1. Edge variance — skip frames that clearly have no text.
    2. Frame differencing — reuse the previous OCR result when consecutive
       frames are nearly identical (same subtitle held across frames).
    """
    import numpy as np

    ocr_fn = ocr_frame_dotsocr if scan_backend == "dotsocr" else ocr_frame_apple

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

    # Pre-load the model before starting progress display
    if scan_backend == "dotsocr":
        _load_dotsocr()

    texts = []
    diff_reused = 0
    last_ocr_text = ""
    last_ocr_arr = None
    for idx, frame in enumerate(frames):
        if idx not in candidates:
            texts.append("")
            last_ocr_text = ""
            last_ocr_arr = None
        elif last_ocr_arr is not None and _frames_are_similar(last_ocr_arr, frame_arrays[idx]):
            texts.append(last_ocr_text)
            diff_reused += 1
        else:
            text = ocr_fn(frame, languages, fast=False).strip()
            texts.append(text)
            last_ocr_text = text
            last_ocr_arr = frame_arrays[idx]
        progress = (idx + 1) / len(frames) * 100
        print(f"\rOCR: {progress:.0f}%", end="", flush=True)
    print()
    if diff_reused:
        print(f"Frame differencing reused OCR for {diff_reused} frames")
    # Free frame arrays
    frame_arrays.clear()
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


# ── DotsMOCR (GPU) backend ─────────────────────────────────────────────

_dotsocr_model = None
_dotsocr_processor = None


def _load_dotsocr():
    """Load the DotsMOCR model and processor (cached after first call)."""
    global _dotsocr_model, _dotsocr_processor
    if _dotsocr_model is not None:
        return _dotsocr_model, _dotsocr_processor

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    model_id = os.environ.get("DOTSOCR_MODEL", "rednote-hilab/dots.mocr")

    print(f"Loading DotsMOCR model ({model_id})...", flush=True)
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        sys.exit(
            "Error: flash-attn is required for the dotsocr backend but is not installed.\n"
            "Install it with: poetry run pip install flash-attn --no-build-isolation"
        )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "vision_config") and attn_impl != "flash_attention_2":
        config.vision_config.attn_implementation = attn_impl
    _dotsocr_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _dotsocr_processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )
    print("DotsMOCR model loaded.", flush=True)
    return _dotsocr_model, _dotsocr_processor


def ocr_frame_dotsocr(frame_path, languages, fast):
    """Run DotsMOCR OCR on a single frame using GPU, return recognized text."""
    from qwen_vl_utils import process_vision_info

    model, processor = _load_dotsocr()

    prompt = "Extract the text content from this image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(frame_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    import logging
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip() if output_text else ""


# ── DotsMOCR 4-bit quantized (GPU) backend ───────────────────────────────

_dots4bit_model = None
_dots4bit_processor = None


def _load_dots4bit():
    """Load the DotsMOCR model in 4-bit quantization (cached after first call)."""
    global _dots4bit_model, _dots4bit_processor
    if _dots4bit_model is not None:
        return _dots4bit_model, _dots4bit_processor

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    model_id = os.environ.get("DOTSOCR_MODEL", "rednote-hilab/dots.mocr")

    print(f"Loading DotsMOCR 4-bit model ({model_id})...", flush=True)

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
        print("flash-attn not found, using SDPA attention", flush=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "vision_config") and attn_impl != "flash_attention_2":
        config.vision_config.attn_implementation = attn_impl
    _dots4bit_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        attn_implementation=attn_impl,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Cast vision tower to bfloat16 to match processor output dtype
    if hasattr(_dots4bit_model, "vision_tower"):
        _dots4bit_model.vision_tower = _dots4bit_model.vision_tower.to(torch.bfloat16)
    _dots4bit_processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True
    )
    print("DotsMOCR 4-bit model loaded.", flush=True)
    return _dots4bit_model, _dots4bit_processor


def ocr_batch_dots4bit(frame_paths, languages, fast):
    """Run DotsMOCR 4-bit OCR on a batch of frames, return list of recognized texts."""
    from qwen_vl_utils import process_vision_info

    model, processor = _load_dots4bit()
    prompt = "Extract the text content from this image."

    all_texts = []
    all_image_inputs = []
    for frame_path in frame_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(frame_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        all_texts.append(text)
        all_image_inputs.extend(image_inputs)

    inputs = processor(
        text=all_texts,
        images=all_image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    import logging
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
    generated_ids = model.generate(**inputs, max_new_tokens=512)

    results = []
    for i, in_ids in enumerate(inputs.input_ids):
        out_ids = generated_ids[i][len(in_ids):]
        decoded = processor.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append(decoded.strip())
    return results


def ocr_frame_dots4bit(frame_path, languages, fast):
    """Run DotsMOCR 4-bit OCR on a single frame, return recognized text."""
    results = ocr_batch_dots4bit([frame_path], languages, fast)
    return results[0] if results else ""


DOTS4BIT_BATCH_SIZE = 64


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


def _screenai_ocr_image(engine, img):
    """Run Screen AI OCR on a PIL Image, return recognized text string."""
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
        return ""

    proto_bytes = ctypes.string_at(result_ptr, output_length.value)
    lib.FreeLibraryAllocatedCharArray(result_ptr)

    annotation = VisualAnnotation()
    annotation.ParseFromString(proto_bytes)
    response = MessageToDict(annotation, preserving_proto_field_name=True)

    lines = []
    for line in response.get("lines", []):
        text = line.get("utf8_string", "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def ocr_frame_screenai(frame_path, languages, fast):
    """Run Chrome Screen AI OCR on a single frame, return recognized text."""
    from PIL import Image

    _ensure_screenai_downloaded()
    engine = _load_screenai()
    img = Image.open(frame_path)
    text = _screenai_ocr_image(engine, img)
    img.close()
    return text


def _screenai_worker_ocr(frame_path):
    """Worker function: OCR a single frame path, return text.

    Each worker lazily initialises its own Screen AI engine on first call
    (via the module-level _load_screenai cache).
    """
    from PIL import Image
    # Suppress noisy native-library logging in workers
    if not getattr(_screenai_worker_ocr, "_stderr_suppressed", False):
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        _screenai_worker_ocr._stderr_suppressed = True
    engine = _load_screenai()
    img = Image.open(frame_path)
    text = _screenai_ocr_image(engine, img)
    img.close()
    return text.strip()


def _ensure_screenai_downloaded():
    """Download Screen AI model files if needed (call before spawning workers)."""
    import platform

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


def detect_text_frames_screenai(frames, languages, num_workers=None):
    """OCR frames using Chrome Screen AI with parallel processes for speed.

    Each worker process loads its own copy of the native library to avoid
    thread-safety issues.  Pre-filtering and frame-differencing reduce the
    total number of OCR calls.
    """
    import numpy as np
    from multiprocessing import get_context

    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8)

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

    # Ensure model files are downloaded before spawning workers
    _ensure_screenai_downloaded()

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

    total_ocr = len(ocr_indices)
    ocr_paths = [str(frames[i]) for i in ocr_indices]

    # Use 'spawn' context so each worker starts fresh (no inherited ctypes state)
    ctx = get_context("spawn")
    print(f"OCR: 0% (0/{total_ocr} frames, {num_workers} workers)", end="", flush=True)
    with ctx.Pool(processes=num_workers) as pool:
        results = []
        for i, text in enumerate(pool.imap(_screenai_worker_ocr, ocr_paths)):
            results.append(text)
            progress = (i + 1) / total_ocr * 100 if total_ocr else 100
            print(f"\rOCR: {progress:.0f}% ({i + 1}/{total_ocr} frames, {num_workers} workers)", end="", flush=True)

    for i, idx in enumerate(ocr_indices):
        texts[idx] = results[i]

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


def detect_text_frames_batched(frames, languages, batch_size=DOTS4BIT_BATCH_SIZE):
    """OCR frames using batched 4-bit DotsMOCR for faster throughput.

    Same pre-filtering as detect_text_frames but collects candidate frames
    into batches for parallel GPU inference.
    """
    import numpy as np

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

    # Pre-load the model before starting progress display
    _load_dots4bit()

    # Build list of frames that need OCR (applying frame differencing)
    texts = [""] * len(frames)
    ocr_indices = []  # indices that need actual OCR
    diff_reused = 0
    last_ocr_text = ""
    last_ocr_arr = None
    for idx in range(len(frames)):
        if idx not in candidates:
            last_ocr_text = ""
            last_ocr_arr = None
        elif last_ocr_arr is not None and _frames_are_similar(last_ocr_arr, frame_arrays[idx]):
            texts[idx] = last_ocr_text
            diff_reused += 1
            # Don't update last_ocr_arr — keep comparing against the original OCR'd frame
        else:
            ocr_indices.append(idx)
            last_ocr_text = ""  # placeholder, will be filled after batch OCR
            last_ocr_arr = frame_arrays[idx]

    if diff_reused:
        print(f"Frame differencing will reuse OCR for {diff_reused} frames")

    # Process OCR indices in batches
    total_ocr = len(ocr_indices)
    processed = 0
    for batch_start in range(0, total_ocr, batch_size):
        batch_indices = ocr_indices[batch_start : batch_start + batch_size]
        batch_paths = [frames[i] for i in batch_indices]
        batch_results = ocr_batch_dots4bit(batch_paths, languages, fast=False)
        for i, idx in enumerate(batch_indices):
            texts[idx] = batch_results[i]
        processed += len(batch_indices)
        progress = processed / total_ocr * 100 if total_ocr else 100
        print(f"\rOCR: {progress:.0f}% ({processed}/{total_ocr} frames)", end="", flush=True)

    # Second pass: propagate OCR results to diff-reused frames
    # (because we didn't know the OCR text at diff-reuse time)
    ocr_set = set(ocr_indices)
    last_ocr_text = ""
    last_ocr_arr = None
    for idx in range(len(frames)):
        if idx not in candidates:
            last_ocr_text = ""
            last_ocr_arr = None
        elif idx in ocr_set:
            # This was an OCR'd frame — update tracking
            last_ocr_text = texts[idx]
            last_ocr_arr = frame_arrays[idx]
        else:
            # This was a diff-reused frame — use the last OCR text
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
    resp = urllib.request.urlopen(req, timeout=300)
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
    resp = urllib.request.urlopen(req, timeout=300)
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


def call_mlx(messages, model="mlx-community/Qwen3-4B-4bit", temperature=0.3):
    """Call a local MLX model via mlx-lm. Returns assistant content."""
    from mlx_lm import load, generate

    mlx_model, tokenizer = _load_mlx(model)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    response = generate(
        mlx_model,
        tokenizer,
        prompt=prompt,
        max_tokens=2048,
        sampler=sampler,
        verbose=False,
    )
    return strip_thinking(response)


_mlx_cache = {}


def _load_mlx(model):
    """Load and cache an MLX model+tokenizer."""
    if model not in _mlx_cache:
        from mlx_lm import load

        print(f"Loading MLX model ({model})...", flush=True)
        _mlx_cache[model] = load(model)
        print("MLX model loaded.", flush=True)
    return _mlx_cache[model]


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

    result = subprocess.run(
        cmd,
        input=user_prompt,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI failed (exit {result.returncode}): {result.stderr}"
        )

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
    # .0, .00 etc. at end → ⋯  (Screen AI reads ⋯ as .00 sometimes)
    text = re.sub(r"\.0+$", "⋯", text)
    # 。0 or trailing 0 at end of sentence → ⋯
    text = re.sub(r"。0", "⋯", text)
    # •。 or ·。 or stray dot combinations → ⋯
    text = re.sub(r"[•·]。", "⋯", text)
    text = re.sub(r"[•·]{2,}", "⋯", text)
    # Stray quotes before repeated chars (hesitation pattern)
    # e.g. 我"我 or 我"。我 → 我⋯我
    text = re.sub(r'(["\u201C\u201D])([。]?)', lambda m: "⋯", text)

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
    # Strip orphan closing brackets/parens without matching opener
    text = re.sub(r"^】", "", text)
    text = re.sub(r"】$", "", text) if "【" not in text else text
    if "(" not in text:
        text = text.rstrip(")")
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


def cleanup_with_llm(
    entries,
    model="qwen3:8b-q4_K_M",
    batch_size=30,
    languages=None,
    thinking=True,
    backend="ollama",
    retry=False,
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
    elif backend == "mlx":
        call_llm_fn = call_mlx
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

        attempt = 0
        while True:
            attempt += 1
            try:
                response = call_llm_fn(messages, model=model, temperature=temperature)
            except Exception as e:
                print(
                    f"  Batch {batch_idx + 1}: LLM call error (attempt {attempt}): {e}",
                    file=sys.stderr,
                    flush=True,
                )
                if not retry:
                    sys.exit(1)
                time.sleep(min(attempt * 2, 30))
                continue

            result = parse_cleanup_response(response, batch)
            if result is None:
                print(
                    f"  Batch {batch_idx + 1}: response unparseable (attempt {attempt})",
                    file=sys.stderr,
                    flush=True,
                )
                if not retry:
                    sys.exit(1)
                print("  Retrying...", file=sys.stderr, flush=True)
                time.sleep(min(attempt * 2, 30))
                continue

            break

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
    parser.add_argument(
        "--scan-backend",
        required=True,
        choices=["apple", "dotsocr", "dots4bit", "screenai"],
        help="OCR backend: apple (macOS Vision), dotsocr (DotsMOCR bf16), dots4bit (DotsMOCR 4-bit quantized + batched), or screenai (Chrome Screen AI)",
    )
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
        choices=["none", "rules", "ollama", "openrouter", "claude", "mlx"],
        help="Cleanup backend: rules (local, no LLM), ollama/openrouter/claude/mlx (LLM), none (skip)",
    )
    parser.add_argument(
        "--cleanup-model",
        default=None,
        help="Model for cleanup (default: qwen3:8b-q4_K_M for ollama, "
        "qwen/qwen3-235b-a22b-2507 for openrouter, "
        "claude-sonnet-4-6 for claude, "
        "mlx-community/Qwen3-4B-4bit for mlx)",
    )
    parser.add_argument(
        "--cleanup-reasoning",
        type=int,
        choices=[0, 1],
        default=None,
        help="Enable (1) or disable (0) thinking/reasoning for cleanup model",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry endlessly on LLM cleanup errors instead of aborting",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=DOTS4BIT_BATCH_SIZE,
        help=f"Batch size for dots4bit backend (default: {DOTS4BIT_BATCH_SIZE})",
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
        if args.cleanup_backend not in ("none", "rules") and args.cleanup_reasoning is None:
            parser.error(
                "--cleanup-reasoning is required when --cleanup-backend is an LLM backend"
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

        # OCR all frames
        if args.scan_backend == "dots4bit":
            ocr_texts = detect_text_frames_batched(frames, args.languages, batch_size=args.scan_batch_size)
        elif args.scan_backend == "screenai":
            ocr_texts = detect_text_frames_screenai(frames, args.languages)
        else:
            ocr_texts = detect_text_frames(frames, args.languages, scan_backend=args.scan_backend)
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
        before_smart = len(entries)
        entries = smart_deduplicate(entries, sample_interval)
        smart_merged = before_smart - len(entries)
        if smart_merged:
            print(f"Smart dedup merged {smart_merged} single-frame entries")
        if args.cleanup_backend not in ("none", None):
            # Save raw (pre-cleanup) SRT alongside the final one
            raw_path = os.path.splitext(output_path)[0] + ".raw.srt"
            write_srt(entries, raw_path)
            print(f"Written {len(entries)} raw subtitle entries to {raw_path}")

            backend = args.cleanup_backend
            if backend == "rules":
                print("Cleaning up OCR artifacts with rules-based cleanup...")
                entries = cleanup_rules(entries, languages=args.languages)
            else:
                default_models = {
                    "openrouter": "qwen/qwen3-235b-a22b-2507",
                    "claude": "claude-sonnet-4-6",
                    "ollama": "qwen3:8b-q4_K_M",
                    "mlx": "mlx-community/Qwen3-4B-4bit",
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
                    retry=args.retry,
                )
            # Fuzzy dedup: merge consecutive near-duplicate entries the LLM missed
            entries = deduplicate(entries, max_dist=2)
            # Ensure every frame in detected clips is covered by a subtitle
            entries = fill_clip_gaps(entries, clips, sample_interval)
        write_srt(entries, output_path)
        print(f"Written {len(entries)} subtitle entries to {output_path}")


if __name__ == "__main__":
    main()
