import base64
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from prompt import build_floor_plan_prompt, build_wall_plan_prompt, build_composition_prompt
from validation import (
    VALIDATION_ENABLED,
    VALIDATION_MAX_ATTEMPTS,
    VALIDATION_CANDIDATES_PER_ATTEMPT,
    ValidationResult,
    validate_spatial_accuracy,
    validate_wall_accuracy,
    build_correction_notes,
    pick_best_candidate,
)

logger = logging.getLogger(__name__)

MODEL = "gpt-image-2"
_DEFAULT_WALL_H = 2.8  # metres


# ── URL / path utilities ────────────────────────────────────────────────────

def sanitize_url(u: str) -> str:
    from urllib.parse import urlsplit, urlunsplit, quote
    u = (u or "").strip()
    if not u:
        return u
    parts = urlsplit(u)
    path = quote(parts.path, safe="/:%")
    query = quote(parts.query, safe="=&%:,")
    return urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def normalize_hex(hex_str: str) -> str:
    if not hex_str:
        raise ValueError("hex_color is missing/empty")
    s = hex_str.strip().lower().lstrip("#")
    if not re.fullmatch(r"[0-9a-f]{6}", s):
        raise ValueError(f"Invalid hex color: {hex_str}")
    return f"#{s}"


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    s = normalize_hex(hex_str)[1:]
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _guess_mime(b: bytes) -> str:
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if b.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    head = b[:512].lstrip()
    if head.startswith(b"<svg") or head.startswith(b"<?xml") or b"<svg" in head:
        raise ValueError(
            "SVG images are not supported by the OpenAI API. "
            "Provide a real product photo URL (PNG, JPEG, or WebP)."
        )
    if head.startswith(b"<!doctype") or head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or head.startswith(b"<HTML"):
        raise ValueError(
            "The URL returned an HTML webpage, not an image. "
            "Please provide a direct image URL (ending in .png, .jpg, .webp) "
            "instead of a product page URL."
        )
    return "image/png"


def download_bytes(url_or_path: str, timeout: int = 60, max_retries: int = 3) -> bytes:
    # Handle local filesystem paths
    if os.path.exists(url_or_path):
        with open(url_or_path, "rb") as f:
            return f.read()

    url = sanitize_url(url_or_path)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "image/jpeg,image/png,image/webp,image/*;q=0.8,*/*;q=0.5",
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.SSLError,
        ):
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            raise


def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert("RGBA")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _sanitize_image_bytes(raw: bytes) -> bytes:
    """Convert any downloaded image to a clean RGB PNG that OpenAI can accept.

    Raises ValueError if the bytes cannot be parsed as a valid image,
    rather than silently sending garbage to the API.
    """
    # Early check: detect non-image content before trying to parse
    _guess_mime(raw)  # raises ValueError for SVG, HTML, etc.

    try:
        img = Image.open(BytesIO(raw))
        img.load()  # force full decode to catch truncated/corrupt files
        # Convert problematic modes (CMYK, palette, LA, etc.) to RGB
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        return pil_to_png_bytes(img)
    except ValueError:
        raise  # re-raise our own content-type errors
    except Exception as exc:
        raise ValueError(
            f"Could not process image: {exc}. "
            f"Make sure the URL points directly to a valid image file (PNG, JPEG, or WebP)."
        )


# ── OpenAI API ──────────────────────────────────────────────────────────────

def _openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("FALLBACK_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in environment.")
    return key


def _openai_product_placement_edit(
    prompt: str,
    image_bytes_list: List[bytes],
    model: str = MODEL,
    size: str = "1024x1024",
) -> bytes:
    if not image_bytes_list:
        raise ValueError("At least one image is required.")

    files = []
    for idx, image_bytes in enumerate(image_bytes_list, start=1):
        mime = _guess_mime(image_bytes)
        ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "png")
        files.append(("image[]", (f"source_{idx}.{ext}", image_bytes, mime)))

    logger.info("Prompt length: %d chars, %d image(s)", len(prompt), len(image_bytes_list))
    response = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {_openai_api_key()}"},
        data={"model": model, "prompt": prompt, "n": "1", "size": size},
        files=files,
        timeout=300,
    )
    if not response.ok:
        error_body = response.text[:2000]
        logger.error("OpenAI edits API error %s: %s", response.status_code, error_body)
        raise RuntimeError(f"OpenAI API error {response.status_code}: {error_body}")
    payload = response.json()
    return base64.b64decode(payload["data"][0]["b64_json"])


# ── Floor plan generation (text → image, no base image required) ─────────────

def _openai_generate(prompt: str, size: str = "1024x1024") -> bytes:
    """Call the gpt-image-2 generations endpoint (no base image needed)."""
    body = {
        "model": MODEL,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    logger.info("Generation request body (excluding prompt): model=%s size=%s", MODEL, size)
    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={
            "Authorization": f"Bearer {_openai_api_key()}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120,
    )
    if not response.ok:
        logger.error("OpenAI generations API error %s: %s", response.status_code, response.text)
        response.raise_for_status()
    item = response.json()["data"][0]
    if "b64_json" in item:
        return base64.b64decode(item["b64_json"])
    return download_bytes(item["url"])


def _make_blank_canvas(size: str = "1024x1024") -> bytes:
    w, h = (int(x) for x in size.split("x"))
    img = Image.new("RGB", (w, h), "white")
    return pil_to_png_bytes(img)


# ── Floor plan guide generation ─────────────────────────────────────────────

_GUIDE_GRAYS = [
    "#A0A0A0", "#787878", "#B8B8B8", "#909090",
    "#686868", "#C8C8C8", "#808080", "#989898",
]


def _assign_guide_colors(products: list) -> list:
    """Return a copy of products with a neutral gray auto-assigned where missing.

    Uses neutral grays instead of saturated colors so guide rectangle colors
    don't bleed into the rendered product appearance.
    """
    result = []
    for i, p in enumerate(products):
        p = dict(p)
        p["hex_color"] = _GUIDE_GRAYS[i % len(_GUIDE_GRAYS)]
        result.append(p)
    return result


def _get_font(size: int = 12) -> ImageFont.FreeTypeFont:
    """Try to load a TrueType font at the given size, fall back to default."""
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _make_floor_plan_guide(payload: dict, canvas_size: int = 1024) -> bytes:
    """
    Programmatically draw a 2D top-down floor plan at exact scale:
    - Room walls and architectural openings
    - Each product's floor footprint as a coloured filled rectangle
    - Product ID labels and dimension annotations inside each rectangle
    - Front-edge markers showing product orientation
    - Room dimension labels along edges and a compass indicator

    This gives the model visual ground truth for both position and size,
    so no geometry needs to appear in the text prompt.
    """
    room = payload.get("room", {})
    room_w = float(room.get("width") or 0)
    room_h = float(room.get("length") or 0)
    if not room_w or not room_h:
        return _make_blank_canvas(f"{canvas_size}x{canvas_size}")

    # Equal ppm for x and y — no distortion
    margin = 50
    usable = canvas_size - 2 * margin
    ppm = usable / max(room_w, room_h)

    rw_px = int(round(room_w * ppm))
    rh_px = int(round(room_h * ppm))
    ox = (canvas_size - rw_px) // 2
    oy = (canvas_size - rh_px) // 2

    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Floor fill
    draw.rectangle([ox, oy, ox + rw_px, oy + rh_px], fill=(240, 228, 210))

    # 25%/50%/75% grid lines on both axes
    grid_color = (210, 200, 185)
    for pct in (0.25, 0.50, 0.75):
        gx = ox + int(round(rw_px * pct))
        gy = oy + int(round(rh_px * pct))
        draw.line([(gx, oy), (gx, oy + rh_px)], fill=grid_color, width=1)
        draw.line([(ox, gy), (ox + rw_px, gy)], fill=grid_color, width=1)

    # Wall thickness
    wall_cfg = room.get("walls", {})
    wall_t = max(6, int(round(float(wall_cfg.get("thickness") or 0.12) * ppm)))
    wall_fill = (55, 55, 55)

    # Solid walls (draw as four rectangles)
    draw.rectangle([ox - wall_t, oy - wall_t, ox + rw_px + wall_t, oy], fill=wall_fill)          # north
    draw.rectangle([ox - wall_t, oy + rh_px, ox + rw_px + wall_t, oy + rh_px + wall_t], fill=wall_fill)  # south
    draw.rectangle([ox - wall_t, oy - wall_t, ox, oy + rh_px + wall_t], fill=wall_fill)          # west
    draw.rectangle([ox + rw_px, oy - wall_t, ox + rw_px + wall_t, oy + rh_px + wall_t], fill=wall_fill)  # east

    # Openings — cut gaps in walls and label them
    opening_label_font = _get_font(11)
    for o in payload.get("openings", []):
        wall = (o.get("wall") or "").lower()
        pos = float(o.get("position_from_left") or 0)
        ow = float(o.get("width") or 0)
        otype = (o.get("type") or "").lower()
        gap_fill = (210, 180, 140) if otype == "door" else (140, 185, 220)
        outline_color = (180, 140, 90) if otype == "door" else (100, 150, 200)

        pos_px = int(round(pos * ppm))
        ow_px = int(round(ow * ppm))

        # Draw the gap (wider than wall thickness for visibility)
        extra = max(4, wall_t // 2)
        if wall == "north":
            r = [ox + pos_px, oy - wall_t - extra, ox + pos_px + ow_px, oy + extra]
        elif wall == "south":
            r = [ox + pos_px, oy + rh_px - extra, ox + pos_px + ow_px, oy + rh_px + wall_t + extra]
        elif wall == "west":
            r = [ox - wall_t - extra, oy + pos_px, ox + extra, oy + pos_px + ow_px]
        elif wall == "east":
            r = [ox + rw_px - extra, oy + pos_px, ox + rw_px + wall_t + extra, oy + pos_px + ow_px]
        else:
            continue

        draw.rectangle(r, fill=gap_fill, outline=outline_color, width=2)

        # Label inside the gap
        lbl = "DOOR" if otype == "door" else "WINDOW"
        text_color = (100, 70, 30) if otype == "door" else (30, 70, 130)
        bbox_lbl = draw.textbbox((0, 0), lbl, font=opening_label_font)
        lw, lh = bbox_lbl[2] - bbox_lbl[0], bbox_lbl[3] - bbox_lbl[1]
        cx = (r[0] + r[2]) / 2
        cy = (r[1] + r[3]) / 2
        # Only draw label if it fits in both dimensions
        if lw < abs(r[2] - r[0]) - 2 and lh < abs(r[3] - r[1]) - 2:
            draw.text((cx - lw / 2, cy - lh / 2), lbl, fill=text_color, font=opening_label_font)

    # Fonts for labels
    label_font = _get_font(13)
    dim_font = _get_font(11)
    room_label_font = _get_font(14)
    number_font = _get_font(18)

    _FACING_MAP = {0: "N", 90: "E", 180: "S", 270: "W"}

    # Product footprints — neutral gray rectangles with number labels
    for prod_idx, p in enumerate(payload.get("products", [])):
        dims = p.get("dimensions", {})
        dim_w = float(dims.get("width") or dims.get("diameter") or 0)
        dim_d = float(dims.get("depth") or dims.get("diameter") or dim_w)
        pos_d = p.get("position", {})
        rot = int(p.get("rotation_y") or p.get("rotation_degrees") or 0)
        cx_m = float(pos_d.get("x") or 0)
        cy_m = float(pos_d.get("y") or 0)

        plan_w, plan_d = (dim_d, dim_w) if rot in (90, 270) else (dim_w, dim_d)

        cx_px = ox + cx_m * ppm
        cy_px = oy + cy_m * ppm
        hw = plan_w * ppm / 2
        hd = plan_d * ppm / 2

        x0 = max(float(ox), cx_px - hw)
        y0 = max(float(oy), cy_px - hd)
        x1 = min(float(ox + rw_px), cx_px + hw)
        y1 = min(float(oy + rh_px), cy_px + hd)

        try:
            r, g, b = hex_to_rgb(p.get("hex_color") or "#888888")
        except Exception:
            r, g, b = 136, 136, 136

        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(0, 0, 0), width=2)

        # ── Product ID label centered inside the rectangle ──
        pid = p.get("id", "")
        if pid:
            # Pick contrasting text color (white on dark, black on light)
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = (255, 255, 255) if luma < 140 else (0, 0, 0)
            rect_cx = (x0 + x1) / 2
            rect_cy = (y0 + y1) / 2 - 7  # nudge up to leave room for dim text
            bbox = draw.textbbox((0, 0), pid, font=label_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # Only draw if text fits inside the rectangle
            if tw < (x1 - x0) - 4 and th < (y1 - y0) - 4:
                draw.text((rect_cx - tw / 2, rect_cy - th / 2), pid, fill=text_color, font=label_font)

        # ── Dimension annotation below the ID ──
        dim_text = f"{dim_w:.1f}\u00d7{dim_d:.1f}m"
        bbox_d = draw.textbbox((0, 0), dim_text, font=dim_font)
        dtw, dth = bbox_d[2] - bbox_d[0], bbox_d[3] - bbox_d[1]
        dim_cx = (x0 + x1) / 2
        dim_cy = (y0 + y1) / 2 + 7  # nudge down below ID text
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (255, 255, 255) if luma < 140 else (0, 0, 0)
        if dtw < (x1 - x0) - 4 and dth < (y1 - y0) - 4:
            draw.text((dim_cx - dtw / 2, dim_cy - dth / 2), dim_text, fill=text_color, font=dim_font)

        # ── Front-edge marker (thick line on the edge the product faces) ──
        rot_norm = rot % 360
        facing = _FACING_MAP.get(rot_norm, "")
        marker_color = (0, 0, 0)
        marker_w = 3
        if facing == "N":
            draw.line([(x0, y0), (x1, y0)], fill=marker_color, width=marker_w)
        elif facing == "S":
            draw.line([(x0, y1), (x1, y1)], fill=marker_color, width=marker_w)
        elif facing == "E":
            draw.line([(x1, y0), (x1, y1)], fill=marker_color, width=marker_w)
        elif facing == "W":
            draw.line([(x0, y0), (x0, y1)], fill=marker_color, width=marker_w)

        # ── Percentage coordinate labels at top-left corner of product ──
        if rw_px > 0 and rh_px > 0:
            pct_x = int(round((cx_px - ox) / rw_px * 100))
            pct_y = int(round((cy_px - oy) / rh_px * 100))
            pct_label = f"{pct_x}%,{pct_y}%"
            pct_font = _get_font(9)
            pct_bbox = draw.textbbox((0, 0), pct_label, font=pct_font)
            pct_tw = pct_bbox[2] - pct_bbox[0]
            pct_th = pct_bbox[3] - pct_bbox[1]
            # Draw just above the top-left corner
            draw.text(
                (x0, y0 - pct_th - 2), pct_label,
                fill=(80, 80, 80), font=pct_font,
            )

        # ── Circled number label for product-to-image mapping ──
        num_label = str(prod_idx + 1)
        num_bbox = draw.textbbox((0, 0), num_label, font=number_font)
        num_tw = num_bbox[2] - num_bbox[0]
        num_th = num_bbox[3] - num_bbox[1]
        circle_r = max(num_tw, num_th) // 2 + 4
        circle_cx = x0 + circle_r + 3
        circle_cy = y0 + circle_r + 3
        draw.ellipse(
            [circle_cx - circle_r, circle_cy - circle_r,
             circle_cx + circle_r, circle_cy + circle_r],
            fill=(255, 255, 255), outline=(0, 0, 0), width=2,
        )
        draw.text(
            (circle_cx - num_tw / 2, circle_cy - num_th / 2),
            num_label, fill=(0, 0, 0), font=number_font,
        )

    # ── Room dimension labels ──
    # Width label along the bottom edge
    width_text = f"{room_w:.1f}m"
    bbox_w = draw.textbbox((0, 0), width_text, font=room_label_font)
    wtw = bbox_w[2] - bbox_w[0]
    draw.text(
        (ox + rw_px / 2 - wtw / 2, oy + rh_px + wall_t + 6),
        width_text, fill=(40, 40, 40), font=room_label_font,
    )

    # Length label along the left edge (drawn horizontally for readability)
    length_text = f"{room_h:.1f}m"
    bbox_l = draw.textbbox((0, 0), length_text, font=room_label_font)
    ltw, lth = bbox_l[2] - bbox_l[0], bbox_l[3] - bbox_l[1]
    draw.text(
        (ox - wall_t - ltw - 6, oy + rh_px / 2 - lth / 2),
        length_text, fill=(40, 40, 40), font=room_label_font,
    )

    # ── Compass "N" indicator at the top center ──
    n_text = "N"
    bbox_n = draw.textbbox((0, 0), n_text, font=room_label_font)
    ntw = bbox_n[2] - bbox_n[0]
    n_x = ox + rw_px / 2 - ntw / 2
    n_y = oy - wall_t - 20
    draw.text((n_x, n_y), n_text, fill=(40, 40, 40), font=room_label_font)
    # Small arrow pointing up below the N
    arrow_cx = ox + rw_px / 2
    arrow_top = n_y + 14
    draw.line([(arrow_cx, arrow_top), (arrow_cx, arrow_top + 8)], fill=(40, 40, 40), width=2)
    draw.polygon(
        [(arrow_cx - 3, arrow_top + 2), (arrow_cx + 3, arrow_top + 2), (arrow_cx, arrow_top - 2)],
        fill=(40, 40, 40),
    )

    return pil_to_png_bytes(img)


def _generate_isometric_with_validation(
    payload: dict,
    image_bytes_list: List[bytes],
    guide_bytes: bytes,
    size: str = "1024x1024",
    image_order: Optional[List[str]] = None,
    has_guide: bool = False,
) -> Tuple[bytes, List[Dict]]:
    """Generate isometric view with spatial validation and retry loop.

    Returns (best_isometric_bytes, attempt_metrics_list).
    """
    attempt_metrics: List[Dict] = []
    best_iso: Optional[bytes] = None
    best_validation: Optional[ValidationResult] = None
    correction_notes = ""

    max_attempts = VALIDATION_MAX_ATTEMPTS if VALIDATION_ENABLED else 1
    candidates_per = VALIDATION_CANDIDATES_PER_ATTEMPT if VALIDATION_ENABLED else 1

    for attempt in range(1, max_attempts + 1):
        logger.info("Isometric attempt %d/%d", attempt, max_attempts)

        # Build prompt (with correction notes on retries)
        iso_prompt = build_floor_plan_prompt(
            payload,
            image_order=image_order,
            has_guide=has_guide,
            view="isometric",
            correction_notes=correction_notes,
        )
        logger.info("Isometric prompt %d chars (attempt %d)", len(iso_prompt), attempt)

        # Generate N candidates (parallel if N > 1)
        def _gen_one(_idx: int) -> bytes:
            return _openai_product_placement_edit(iso_prompt, image_bytes_list, size=size)

        if candidates_per > 1:
            with ThreadPoolExecutor(max_workers=min(candidates_per, 4)) as pool:
                futures = [pool.submit(_gen_one, i) for i in range(candidates_per)]
                candidate_bytes_list = [f.result() for f in futures]
        else:
            candidate_bytes_list = [_gen_one(0)]

        # Validate each candidate
        candidates_with_scores: List[Tuple[bytes, ValidationResult]] = []
        for idx, cand_bytes in enumerate(candidate_bytes_list):
            if VALIDATION_ENABLED:
                vr = validate_spatial_accuracy(cand_bytes, guide_bytes, payload)
                logger.info(
                    "Candidate %d score=%.3f passed=%s", idx + 1, vr.score, vr.passed,
                )
            else:
                vr = ValidationResult(score=1.0, passed=True)
            candidates_with_scores.append((cand_bytes, vr))

        # Pick best candidate this attempt
        attempt_best_bytes, attempt_best_vr = pick_best_candidate(candidates_with_scores)

        attempt_metrics.append({
            "attempt": attempt,
            "candidates": candidates_per,
            "best_score": attempt_best_vr.score,
            "passed": attempt_best_vr.passed,
            "details": attempt_best_vr.to_dict(),
        })

        # Track overall best across all attempts
        if best_validation is None or attempt_best_vr.score > best_validation.score:
            best_iso = attempt_best_bytes
            best_validation = attempt_best_vr

        # If passed, stop retrying
        if attempt_best_vr.passed:
            logger.info("Validation passed on attempt %d (score=%.3f)", attempt, attempt_best_vr.score)
            break

        # Build correction notes for next attempt
        if attempt < max_attempts:
            correction_notes = build_correction_notes(attempt_best_vr)
            logger.info("Validation failed (score=%.3f), retrying with corrections", attempt_best_vr.score)

    return best_iso, attempt_metrics


def _generate_views_sequential(
    payload: dict,
    image_bytes_list: List[bytes],
    size: str = "1024x1024",
    image_order: Optional[List[str]] = None,
    has_guide: bool = False,
    guide_bytes: Optional[bytes] = None,
) -> Tuple[Dict[str, bytes], List[Dict]]:
    """Generate top-down (isometric) view only."""
    if guide_bytes is not None and VALIDATION_ENABLED:
        iso_bytes, attempt_metrics = _generate_isometric_with_validation(
            payload, image_bytes_list,
            guide_bytes=guide_bytes,
            size=size,
            image_order=image_order,
            has_guide=has_guide,
        )
    else:
        iso_prompt = build_floor_plan_prompt(
            payload, image_order=image_order, has_guide=has_guide, view="isometric"
        )
        logger.info("Generating top view (no validation), prompt %d chars", len(iso_prompt))
        iso_bytes = _openai_product_placement_edit(iso_prompt, image_bytes_list, size=size)
        attempt_metrics = []

    return {"isometric": iso_bytes}, attempt_metrics


def _generate_wall_elevation_with_validation(
    payload: dict,
    image_bytes_list: List[bytes],
    guide_bytes: bytes,
    size: str = "1024x1024",
    image_order: Optional[List[str]] = None,
    has_guide: bool = False,
) -> Tuple[bytes, List[Dict]]:
    """Generate wall elevation with spatial validation and retry loop.

    Mirrors _generate_isometric_with_validation:
    - Up to VALIDATION_MAX_ATTEMPTS retries
    - VALIDATION_CANDIDATES_PER_ATTEMPT candidates per attempt (parallel if >1)
    - GPT-4o Vision scores each candidate via validate_wall_accuracy
    - Correction notes injected into prompt on each retry
    - Best candidate across all attempts is returned
    """
    attempt_metrics: List[Dict] = []
    best_elevation: Optional[bytes] = None
    best_validation: Optional[ValidationResult] = None
    correction_notes = ""

    max_attempts = VALIDATION_MAX_ATTEMPTS if VALIDATION_ENABLED else 1
    candidates_per = VALIDATION_CANDIDATES_PER_ATTEMPT if VALIDATION_ENABLED else 1

    for attempt in range(1, max_attempts + 1):
        logger.info("Wall elevation attempt %d/%d", attempt, max_attempts)

        wall_prompt = build_wall_plan_prompt(
            payload,
            image_order=image_order,
            has_guide=has_guide,
            correction_notes=correction_notes,
        )
        logger.info("Wall prompt %d chars (attempt %d)", len(wall_prompt), attempt)

        def _gen_one(_idx: int) -> bytes:
            return _openai_product_placement_edit(wall_prompt, image_bytes_list, size=size)

        if candidates_per > 1:
            with ThreadPoolExecutor(max_workers=min(candidates_per, 4)) as pool:
                futures = [pool.submit(_gen_one, i) for i in range(candidates_per)]
                candidate_bytes_list = [f.result() for f in futures]
        else:
            candidate_bytes_list = [_gen_one(0)]

        candidates_with_scores: List[Tuple[bytes, ValidationResult]] = []
        for idx, cand_bytes in enumerate(candidate_bytes_list):
            if VALIDATION_ENABLED:
                vr = validate_wall_accuracy(cand_bytes, guide_bytes, payload)
                logger.info(
                    "Wall candidate %d score=%.3f passed=%s openings_ok=%s",
                    idx + 1, vr.score, vr.passed, vr.openings_correct,
                )
            else:
                vr = ValidationResult(score=1.0, passed=True)
            candidates_with_scores.append((cand_bytes, vr))

        attempt_best_bytes, attempt_best_vr = pick_best_candidate(candidates_with_scores)

        attempt_metrics.append({
            "attempt": attempt,
            "candidates": candidates_per,
            "best_score": attempt_best_vr.score,
            "passed": attempt_best_vr.passed,
            "details": attempt_best_vr.to_dict(),
        })

        if best_validation is None or attempt_best_vr.score > best_validation.score:
            best_elevation = attempt_best_bytes
            best_validation = attempt_best_vr

        if attempt_best_vr.passed:
            logger.info("Wall validation passed on attempt %d (score=%.3f)", attempt, attempt_best_vr.score)
            break

        if attempt < max_attempts:
            correction_notes = build_correction_notes(attempt_best_vr)
            logger.info("Wall validation failed (score=%.3f), retrying with corrections", attempt_best_vr.score)

    return best_elevation, attempt_metrics


def generate_floor_plan(payload: dict, size: str = "1024x1024") -> Tuple[Dict[str, bytes], List[Dict]]:
    """
    Generate top-down (isometric) render from a structured JSON payload.
    Returns ({"isometric": bytes}, attempt_metrics).
    """
    _validate_and_clamp_layout(payload)
    products = payload.get("products", [])
    products_with_images = [(p["id"], p["image_url"]) for p in products if p.get("image_url")]

    if products_with_images:
        image_order = [pid for pid, _ in products_with_images]
        payload["products"] = _assign_guide_colors(payload.get("products", []))
        guide_bytes = _make_floor_plan_guide(payload)
        image_bytes_list = [guide_bytes]
        for pid, url in products_with_images:
            image_bytes_list.append(_sanitize_image_bytes(download_bytes(url)))
        return _generate_views_sequential(
            payload, image_bytes_list, size=size,
            image_order=image_order, has_guide=True,
            guide_bytes=guide_bytes,
        )

    # No product images — isometric only via text generation
    prompt = build_floor_plan_prompt(payload, view="isometric")
    logger.info("Floor plan generations prompt (%d chars):\n%s", len(prompt), prompt)
    return {"isometric": _openai_generate(prompt, size=size)}, []


# ── Surface Editor → floor-plan payload converter ───────────────────────────

def _batch_to_fp_payload(
    cleaned_products: List[Dict],
    room_dimensions: Dict,
    presets: Dict,
    openings: Optional[List[Dict]] = None,
) -> tuple:
    """
    Convert Surface Editor batch items into a floor-plan-style payload so that
    build_floor_plan_prompt can be reused for the Surface Editor floor flow.
    Returns (payload_dict, image_order_list).
    """
    unit = room_dimensions.get("unit", "m")
    w = float(room_dimensions.get("width") or 0)
    h = float(room_dimensions.get("height") or 0)

    pr = presets or {}
    flooring_material = pr.get("flooring_material", "light oak")
    flooring_type     = pr.get("flooring_type", "wood")
    wall_color        = pr.get("wall_color", "#F3EFE8")
    lighting_type     = pr.get("lighting", "natural_daylight")
    room_type         = pr.get("room_type", "")
    decor_style       = pr.get("decor_style", "")

    products = []
    image_order = []
    seen_ids = {}
    for item in cleaned_products:
        raw_id = item.get("product_name") or item.get("category") or str(item.get("product_id", "")) or "product"
        seen_ids[raw_id] = seen_ids.get(raw_id, 0) + 1
        pid = raw_id if seen_ids[raw_id] == 1 else f"{raw_id}_{seen_ids[raw_id]}"
        dims_obj = item.get("dimensions")
        if isinstance(dims_obj, dict):
            # Structured metres object sent from canvas (already rotation-swapped)
            dim_w = float(dims_obj.get("width") or 1.0)
            dim_d = float(dims_obj.get("depth") or dims_obj.get("height") or dim_w)
        else:
            dims_str = (item.get("dims") or (dims_obj if isinstance(dims_obj, str) else "") or "").strip()
            dims_nums = re.findall(r"[\d.]+", dims_str) if dims_str else []
            dim_w = float(dims_nums[0]) if len(dims_nums) >= 1 else 1.0
            dim_d = float(dims_nums[1]) if len(dims_nums) >= 2 else dim_w

        # Extract real product height (for 3D rendering); fall back to 0.85m
        if isinstance(dims_obj, dict):
            dim_h = float(dims_obj.get("height") or 0) or 0.85
        else:
            dim_h = 0.85

        x_m = item.get("x_m")
        y_m = item.get("y_m")

        entry = {
            "id": pid,
            "hex_color": item.get("hex_color", ""),
            "image_url": item.get("image_url", ""),
            "dimensions": {"width": dim_w, "depth": dim_d, "height": dim_h},
            "position": {
                "x": x_m if x_m is not None else round(w / 2, 3),
                "y": y_m if y_m is not None else round(h / 2, 3),
                "z": 0,
            },
            "rotation_y": item.get("rotation") or 0,
        }
        if item.get("category"):
            entry["category"] = item["category"]
        products.append(entry)
        image_order.append(pid)

    payload = {
        "unit": unit,
        "room": {
            "width": w,
            "length": h,
            "height": 2.8,
            "flooring": {"type": flooring_type, "material": flooring_material},
            "walls": {"color": wall_color},
        },
        "openings": openings or [],
        "products": products,
        "lighting": {"type": lighting_type, "sun_direction": "north", "intensity": 0.85},
        "camera_views": [{"name": "isometric_top_view", "fov": 35}],
    }
    if room_type:
        payload["scene_id"] = room_type
    if decor_style:
        payload["decor_style"] = decor_style

    return payload, image_order


# ── Wall elevation guide + payload ──────────────────────────────────────────

def _make_wall_elevation_guide(payload: dict, canvas_size: int = 1024) -> bytes:
    """Draw a 2D front-elevation guide for a wall surface at exact scale.

    IMAGE 1 for wall generation: wall rectangle, openings (doors/windows),
    product footprints as numbered gray rectangles — mirrors _make_floor_plan_guide.
    """
    wall = payload.get("wall", {})
    wall_w = float(wall.get("width") or 0)
    wall_h = float(wall.get("height") or _DEFAULT_WALL_H)
    if not wall_w or not wall_h:
        return _make_blank_canvas(f"{canvas_size}x{canvas_size}")

    margin = 60
    usable = canvas_size - 2 * margin
    ppm = usable / max(wall_w, wall_h)

    ww_px = int(round(wall_w * ppm))
    wh_px = int(round(wall_h * ppm))
    ox = (canvas_size - ww_px) // 2
    oy = (canvas_size - wh_px) // 2  # oy = ceiling edge; oy+wh_px = floor edge

    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Wall surface background + subtle plaster lines
    draw.rectangle([ox, oy, ox + ww_px, oy + wh_px], fill=(245, 242, 238))
    draw.line([(ox, oy + wh_px), (ox + ww_px, oy + wh_px)], fill=(180, 170, 160), width=3)  # floor
    for y_line in range(oy + 40, oy + wh_px, 40):
        draw.line([(ox, y_line), (ox + ww_px, y_line)], fill=(235, 230, 225), width=1)

    # Thin diagram outline only — NOT a thick border that the AI might render as molding
    draw.rectangle([ox, oy, ox + ww_px, oy + wh_px], outline=(160, 155, 148), width=2)

    label_font = _get_font(13)
    dim_font = _get_font(11)
    number_font = _get_font(18)
    opening_label_font = _get_font(11)

    # Openings (position_from_left = left edge in metres; y from floor)
    for o in payload.get("openings", []):
        otype = (o.get("type") or "").lower()
        pos_left = float(o.get("position_from_left") or 0)
        ow = float(o.get("width") or 0)
        oh = float(o.get("height") or (2.2 if otype == "door" else 0.9))
        sill = float(o.get("sill_height") or 0)

        x0_px = ox + int(round(pos_left * ppm))
        x1_px = ox + int(round((pos_left + ow) * ppm))

        if otype == "door":
            # Door from floor up to oh metres
            y0_px = oy + wh_px - int(round(oh * ppm))
            y1_px = oy + wh_px
            gap_fill, outline_color = (210, 180, 140), (180, 140, 90)
            lbl, text_color = "DOOR", (100, 70, 30)
        else:
            # Window from sill to sill+oh
            y0_px = oy + wh_px - int(round((sill + oh) * ppm))
            y1_px = oy + wh_px - int(round(sill * ppm))
            gap_fill, outline_color = (140, 185, 220), (100, 150, 200)
            lbl, text_color = "WINDOW", (30, 70, 130)

        draw.rectangle([x0_px, y0_px, x1_px, y1_px], fill=gap_fill, outline=outline_color, width=2)
        bbox = draw.textbbox((0, 0), lbl, font=opening_label_font)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((x0_px + x1_px) / 2 - lw / 2, (y0_px + y1_px) / 2 - lh / 2),
                  lbl, fill=text_color, font=opening_label_font)

    # Product footprints — neutral gray rectangles with number labels
    for prod_idx, p in enumerate(payload.get("products", [])):
        dims = p.get("dimensions", {})
        dim_w = float(dims.get("width") or 1.0)
        dim_h = float(dims.get("height") or dims.get("depth") or 0.85)
        rot = int(p.get("rotation_y") or 0) % 360
        # Swap w/h when product is rotated 90°/270° (tipped sideways on the wall)
        draw_w, draw_h = (dim_h, dim_w) if rot in (90, 270) else (dim_w, dim_h)
        pos_d = p.get("position", {})
        cx_m = float(pos_d.get("x") or 0)       # centre from left edge
        cy_m = float(pos_d.get("y") or 0)        # centre height from floor

        cx_px = ox + cx_m * ppm
        cy_px = oy + wh_px - cy_m * ppm          # canvas: floor is oy+wh_px
        hw = draw_w * ppm / 2
        hh = draw_h * ppm / 2

        x0 = max(float(ox), cx_px - hw)
        y0 = max(float(oy), cy_px - hh)
        x1 = min(float(ox + ww_px), cx_px + hw)
        y1 = min(float(oy + wh_px), cy_px + hh)

        try:
            r, g, b = hex_to_rgb(p.get("hex_color") or "#888888")
        except Exception:
            r, g, b = 136, 136, 136

        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(0, 0, 0), width=2)

        luma = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (255, 255, 255) if luma < 140 else (0, 0, 0)

        pid = p.get("id", "")
        if pid:
            bbox = draw.textbbox((0, 0), pid, font=label_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if tw < (x1 - x0) - 4 and th < (y1 - y0) - 4:
                draw.text(((x0 + x1) / 2 - tw / 2, (y0 + y1) / 2 - th / 2 - 7),
                          pid, fill=text_color, font=label_font)

        dim_text = f"{draw_w:.1f}×{draw_h:.1f}m" + (f" [{rot}°]" if rot else "")
        bbox_d = draw.textbbox((0, 0), dim_text, font=dim_font)
        dtw, dth = bbox_d[2] - bbox_d[0], bbox_d[3] - bbox_d[1]
        if dtw < (x1 - x0) - 4 and dth < (y1 - y0) - 4:
            draw.text(((x0 + x1) / 2 - dtw / 2, (y0 + y1) / 2 - dth / 2 + 7),
                      dim_text, fill=text_color, font=dim_font)

        # Circled number
        num_label = str(prod_idx + 1)
        num_bbox = draw.textbbox((0, 0), num_label, font=number_font)
        num_tw = num_bbox[2] - num_bbox[0]
        num_th = num_bbox[3] - num_bbox[1]
        circle_r = max(num_tw, num_th) // 2 + 4
        circle_cx = x0 + circle_r + 3
        circle_cy = y0 + circle_r + 3
        draw.ellipse(
            [circle_cx - circle_r, circle_cy - circle_r,
             circle_cx + circle_r, circle_cy + circle_r],
            fill=(255, 255, 255), outline=(0, 0, 0), width=2,
        )
        draw.text((circle_cx - num_tw / 2, circle_cy - num_th / 2),
                  num_label, fill=(0, 0, 0), font=number_font)

        # Vertical dashed center-line through product (x-anchor visual)
        dash_len, gap_len = 6, 4
        y_cur = float(oy)
        while y_cur < oy + wh_px:
            y_end = min(y_cur + dash_len, float(oy + wh_px))
            draw.line([(cx_px, y_cur), (cx_px, y_end)], fill=(80, 80, 80), width=1)
            y_cur += dash_len + gap_len

        # X-position label below the wall border (ruler annotation)
        # Stagger alternating rows so overlapping labels don't collide
        x_pct = round(cx_m / wall_w * 100) if wall_w else 0
        x_label = f"x={cx_m:.2f}m ({x_pct}%)"
        bbox_xl = draw.textbbox((0, 0), x_label, font=dim_font)
        xl_w = bbox_xl[2] - bbox_xl[0]
        row_offset = (prod_idx % 2) * 16
        label_y = oy + wh_px + 8 + row_offset
        draw.text((cx_px - xl_w / 2, label_y), x_label, fill=(80, 40, 160), font=dim_font)

    # Dimension labels
    room_label_font = _get_font(14)
    width_text = f"{wall_w:.1f}m"
    bbox_w = draw.textbbox((0, 0), width_text, font=room_label_font)
    wtw = bbox_w[2] - bbox_w[0]
    draw.text((ox + ww_px / 2 - wtw / 2, oy + wh_px + 6),
              width_text, fill=(40, 40, 40), font=room_label_font)

    height_text = f"{wall_h:.1f}m"
    bbox_h = draw.textbbox((0, 0), height_text, font=room_label_font)
    htw, hth = bbox_h[2] - bbox_h[0], bbox_h[3] - bbox_h[1]
    draw.text((ox - htw - 8, oy + wh_px / 2 - hth / 2),
              height_text, fill=(40, 40, 40), font=room_label_font)

    return pil_to_png_bytes(img)


def _batch_to_wall_payload(
    cleaned_products: List[Dict],
    room_dimensions: Dict,
    presets: Dict,
    openings: Optional[List[Dict]] = None,
) -> tuple:
    """Convert Surface Editor batch items into a wall-elevation payload.

    Mirrors _batch_to_fp_payload but for front-facing wall renders.
    y_m from the canvas is distance from the TOP (ceiling); we convert to
    distance from the BOTTOM (floor) so the coordinate system matches the
    wall elevation guide (y=0 at floor, y=wall_h at ceiling).
    """
    unit = room_dimensions.get("unit", "m")
    wall_w = float(room_dimensions.get("width") or 0)
    wall_h = float(room_dimensions.get("height") or _DEFAULT_WALL_H)

    pr = presets or {}
    wall_color = pr.get("wall_color", "#F3EFE8")
    lighting_type = pr.get("lighting", "natural_daylight")

    products = []
    image_order = []
    seen_ids: Dict[str, int] = {}

    for item in cleaned_products:
        raw_id = (
            item.get("product_name")
            or item.get("category")
            or str(item.get("product_id", ""))
            or "product"
        )
        seen_ids[raw_id] = seen_ids.get(raw_id, 0) + 1
        pid = raw_id if seen_ids[raw_id] == 1 else f"{raw_id}_{seen_ids[raw_id]}"

        dims_obj = item.get("dimensions")
        if isinstance(dims_obj, dict):
            dim_w = float(dims_obj.get("width") or 1.0)
            dim_h = float(dims_obj.get("height") or dims_obj.get("depth") or dim_w)
        else:
            dims_str = (
                item.get("dims")
                or (dims_obj if isinstance(dims_obj, str) else "")
                or ""
            ).strip()
            dims_nums = re.findall(r"[\d.]+", dims_str) if dims_str else []
            dim_w = float(dims_nums[0]) if len(dims_nums) >= 1 else 1.0
            dim_h = float(dims_nums[1]) if len(dims_nums) >= 2 else dim_w

        x_m = item.get("x_m")
        # Canvas y_m = distance from ceiling (top of canvas).
        # Convert to y_from_floor = distance from floor.
        y_m_from_ceiling = item.get("y_m")
        if y_m_from_ceiling is not None:
            y_from_floor = round(wall_h - float(y_m_from_ceiling), 3)
        else:
            y_from_floor = round(dim_h / 2, 3)  # rest on floor by default

        if x_m is None:
            x_m = round(wall_w / 2, 3)

        entry = {
            "id": pid,
            "hex_color": item.get("hex_color", ""),
            "image_url": item.get("image_url", ""),
            "dimensions": {"width": dim_w, "height": dim_h},
            "position": {"x": round(float(x_m), 3), "y": y_from_floor},
            "rotation_y": item.get("rotation") or 0,
        }
        if item.get("category"):
            entry["category"] = item["category"]
        products.append(entry)
        image_order.append(pid)

    payload = {
        "unit": unit,
        "wall": {"width": wall_w, "height": wall_h, "color": wall_color},
        "openings": openings or [],
        "products": products,
        "lighting": {"type": lighting_type},
    }
    return payload, image_order


def _validate_and_clamp_wall_layout(payload: dict) -> dict:
    """Clamp every product position so its footprint stays inside the wall."""
    wall = payload.get("wall", {})
    wall_w = float(wall.get("width") or 0)
    wall_h = float(wall.get("height") or _DEFAULT_WALL_H)
    if not wall_w or not wall_h:
        return payload

    for p in payload.get("products", []):
        dims = p.get("dimensions", {})
        pw = float(dims.get("width") or 0)
        ph = float(dims.get("height") or 0)
        rot = int(p.get("rotation_y") or 0) % 360
        eff_w, eff_h = (ph, pw) if rot in (90, 270) else (pw, ph)
        half_w, half_h = eff_w / 2, eff_h / 2

        pos = p.get("position", {})
        x = float(pos.get("x") or 0)
        y = float(pos.get("y") or 0)

        clamped_x = max(half_w, min(wall_w - half_w, x))
        clamped_y = max(half_h, min(wall_h - half_h, y))

        if abs(clamped_x - x) > 0.001 or abs(clamped_y - y) > 0.001:
            logger.warning(
                "Wall layout: product %s clamped from (%.3f, %.3f) to (%.3f, %.3f) "
                "to stay within wall %.1f × %.1f",
                p.get("id", "?"), x, y, clamped_x, clamped_y, wall_w, wall_h,
            )
            pos["x"] = round(clamped_x, 3)
            pos["y"] = round(clamped_y, 3)

    return payload


def _validate_and_clamp_layout(payload: dict) -> dict:
    """Clamp every product position so its footprint stays inside the room.

    Modifies `payload` in-place and returns it for convenience.
    Logs a warning for every coordinate that was corrected.
    """
    room = payload.get("room", {})
    room_w = float(room.get("width") or 0)
    room_l = float(room.get("length") or 0)
    if not room_w or not room_l:
        return payload

    for p in payload.get("products", []):
        dims = p.get("dimensions", {})
        pw = float(dims.get("width") or 0)
        pd = float(dims.get("depth") or 0)
        rot = int(p.get("rotation_y") or 0) % 360

        # After rotation, effective footprint
        eff_w, eff_d = (pd, pw) if rot in (90, 270) else (pw, pd)
        half_w = eff_w / 2
        half_d = eff_d / 2

        pos = p.get("position", {})
        x = float(pos.get("x") or 0)
        y = float(pos.get("y") or 0)

        clamped_x = max(half_w, min(room_w - half_w, x))
        clamped_y = max(half_d, min(room_l - half_d, y))

        if abs(clamped_x - x) > 0.001 or abs(clamped_y - y) > 0.001:
            logger.warning(
                "Layout validation: product %s clamped from (%.3f, %.3f) to (%.3f, %.3f) "
                "to stay within room %.1f x %.1f",
                p.get("id", "?"), x, y, clamped_x, clamped_y, room_w, room_l,
            )
            pos["x"] = round(clamped_x, 3)
            pos["y"] = round(clamped_y, 3)

    return payload


# ── Main generation function ────────────────────────────────────────────────

def generate_product_placement(
    products: List[Dict],
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
    generation_type: str = "floor",
    openings: Optional[List[Dict]] = None,
) -> Tuple[Dict[str, bytes], List[Dict]]:
    """Generate product placement renders for floor or wall surfaces.

    Returns (view_images_dict, attempt_metrics).
    Floor: views = {"isometric"}  (top view only)
    Wall:  views = {"elevation"}
    """
    if not products:
        raise ValueError("products list is empty.")

    # Normalize products
    cleaned_products = []
    for p in products:
        if hasattr(p, "model_dump"):
            p = p.model_dump()
        hc = normalize_hex(p.get("hex_color") or p.get("color", ""))
        img_url = p.get("image_url")
        if not img_url:
            raise ValueError(f"Missing image_url for product_id={p.get('product_id')}")
        cleaned = dict(p)
        cleaned["hex_color"] = hc
        cleaned["image_url"] = img_url
        cleaned_products.append(cleaned)

    # ── Floor: programmatic guide + JSON geometry, top view only ───────────
    if generation_type == "floor":
        fp_payload, image_order = _batch_to_fp_payload(
            cleaned_products, room_dimensions or {}, presets or {},
            openings=openings or [],
        )
        _validate_and_clamp_layout(fp_payload)
        fp_payload["products"] = _assign_guide_colors(fp_payload.get("products", []))
        logger.info("Surface Editor floor: %d product(s)", len(cleaned_products))
        guide_bytes = _make_floor_plan_guide(fp_payload)
        image_bytes_list = [guide_bytes]
        for p in cleaned_products:
            image_bytes_list.append(_sanitize_image_bytes(download_bytes(p["image_url"])))
        return _generate_views_sequential(
            fp_payload, image_bytes_list,
            image_order=image_order, has_guide=True,
            guide_bytes=guide_bytes,
        )

    # ── Wall: programmatic guide + JSON geometry + validation retry loop ─────
    wall_payload, image_order = _batch_to_wall_payload(
        cleaned_products, room_dimensions or {}, presets or {},
        openings=openings or [],
    )
    _validate_and_clamp_wall_layout(wall_payload)
    wall_payload["products"] = _assign_guide_colors(wall_payload.get("products", []))
    logger.info("Surface Editor wall: %d product(s)", len(cleaned_products))
    guide_bytes = _make_wall_elevation_guide(wall_payload)
    image_bytes_list = [guide_bytes]
    for p in cleaned_products:
        image_bytes_list.append(_sanitize_image_bytes(download_bytes(p["image_url"])))

    if VALIDATION_ENABLED:
        elevation_bytes, attempt_metrics = _generate_wall_elevation_with_validation(
            wall_payload, image_bytes_list,
            guide_bytes=guide_bytes,
            image_order=image_order,
            has_guide=True,
        )
    else:
        prompt = build_wall_plan_prompt(wall_payload, image_order=image_order, has_guide=True)
        logger.info("Wall elevation prompt %d chars (no validation)", len(prompt))
        elevation_bytes = _openai_product_placement_edit(prompt, image_bytes_list)
        attempt_metrics = []

    return {"elevation": elevation_bytes}, attempt_metrics


def generate_room_composition(
    floor_image_url: str,
    wall_image_urls: List[Dict],
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
    base_dir: Optional[str] = None,
) -> Dict[str, bytes]:
    """Generate isometric compositions from two opposite corners.

    Returns {"sw": bytes, "ne": bytes}:
      sw — camera at SW corner looking NE; shows NORTH + EAST walls.
      ne — camera at NE corner looking SW; shows SOUTH + WEST walls.

    floor_image_url: server-relative path like "/outputs/xxx_isometric.png"
    wall_image_urls: list of {url, label, width_m, height_m} — label must be a compass
                     direction ("north", "south", "east", "west").
    """
    root = Path(base_dir) if base_dir else Path(__file__).parent

    def _resolve(url: str) -> bytes:
        path = root / url.lstrip("/")
        if path.exists() and path.is_file():
            return _sanitize_image_bytes(path.read_bytes())
        if url.startswith("http://") or url.startswith("https://"):
            return _sanitize_image_bytes(download_bytes(url))
        raise FileNotFoundError(f"Image not found at local path: {path}")

    floor_bytes = _resolve(floor_image_url)

    # Build compass → bytes mapping from the provided wall elevation images
    wall_map: Dict[str, bytes] = {}
    for wi in wall_image_urls:
        label = (wi.get("label") or "").lower().strip()
        if label:
            wall_map[label] = _resolve(wi["url"])

    # Each corner view sees exactly 2 walls; order defines IMAGE 2, IMAGE 3
    _CORNER_WALLS = {
        "sw": ["north", "east"],
        "ne": ["south", "west"],
    }

    def _gen_corner(corner: str) -> tuple:
        ordered = _CORNER_WALLS[corner]
        corner_labels = [w for w in ordered if w in wall_map]
        images = [floor_bytes] + [wall_map[w] for w in corner_labels]
        prompt = build_composition_prompt(
            room_dimensions=room_dimensions or {},
            wall_labels=corner_labels,
            presets=presets or {},
            n_walls=len(corner_labels),
            corner=corner,
        )
        logger.info(
            "Composition %s corner: walls=%s prompt=%d chars images=%d",
            corner, corner_labels, len(prompt), len(images),
        )
        return corner, _openai_product_placement_edit(prompt, images)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_gen_corner, c) for c in ("sw", "ne")]
        results: Dict[str, bytes] = {}
        for f in as_completed(futures):
            corner_name, img_bytes = f.result()
            results[corner_name] = img_bytes

    return results
