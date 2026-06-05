import base64
import copy
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from prompt import (
    build_floor_plan_prompt, build_wall_plan_prompt,
    build_dollhouse_shell_prompt, build_dollhouse_add_wall_prompt,
    build_geometric_furnish_prompt, build_dollhouse_oneshot_prompt,
    build_camera_view_config,
    dollhouse_view_walls,
)
import geometry as geo_mod
from validation import (
    VALIDATION_ENABLED,
    VALIDATION_MAX_ATTEMPTS,
    VALIDATION_CANDIDATES_PER_ATTEMPT,
    COMPOSITION_VALIDATION_ENABLED,
    COMPOSITION_MAX_ATTEMPTS,
    COMPOSITION_CANDIDATES_PER_ATTEMPT,
    ValidationResult,
    validate_spatial_accuracy,
    validate_wall_accuracy,
    validate_composition_accuracy,
    build_correction_notes,
    pick_best_candidate,
)

logger = logging.getLogger(__name__)

MODEL = "gpt-image-2"
_DEFAULT_WALL_H = 4.0  # metres


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


# ── Gemini API (composition / Nano Banana) ──────────────────────────────────
# The composite is the only image the user sees, so it runs on Gemini's image model
# (gemini-3-pro-image by default). Nano Banana fuses multiple reference images in a
# single call while preserving each subject's identity, and edits surgically without
# re-drawing the rest — which is exactly what robust compositing + self-correction need.

GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image")
# Backends for the two opposite dollhouse views:
#   oneshot   – (DEFAULT) one call per view: floor + all four wall elevations sent together with
#               an explicit Image→compass map in the prompt; the model removes the near wall and
#               renders the cutaway in a single pass. Fewest model calls (lowest latency).
#   geometric – guide-anchored: a computed structure guide locks the camera/walls, the model
#               furnishes the floor, then paints each wall one elevation at a time.
#   gemini    – fully generative shell-from-floor + per-wall edits.
#   openai    – the gemini pipeline on gpt-image-2.
COMPOSITION_BACKEND = os.getenv("COMPOSITION_BACKEND", "oneshot").strip().lower()
COMPOSITION_ASPECT_RATIO = os.getenv("COMPOSITION_ASPECT_RATIO", "4:3")
COMPOSITION_IMAGE_SIZE = os.getenv("COMPOSITION_IMAGE_SIZE", "2K")


def _gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY in environment.")
    return key


def _extract_gemini_image(payload: Dict) -> bytes:
    for cand in (payload.get("candidates") or []):
        for part in ((cand.get("content") or {}).get("parts") or []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    cands = payload.get("candidates") or []
    finish = cands[0].get("finishReason") if cands else None
    raise RuntimeError(
        f"Gemini returned no image (finishReason={finish}): {str(payload)[:1000]}"
    )


def _gemini_image_edit(
    prompt: str,
    image_bytes_list: List[bytes],
    model: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    image_size: Optional[str] = None,
) -> bytes:
    """Generate/edit an image with a Gemini image model via the REST API.

    Accepts multiple reference images in one call (floor + each wall elevation, or a
    prior composite for refinement). Returns raw image bytes.
    """
    model = model or GEMINI_IMAGE_MODEL
    aspect_ratio = aspect_ratio or COMPOSITION_ASPECT_RATIO
    image_size = image_size or COMPOSITION_IMAGE_SIZE

    parts: List[Dict] = [{"text": prompt}]
    for image_bytes in image_bytes_list:
        parts.append({
            "inline_data": {
                "mime_type": _guess_mime(image_bytes),
                "data": base64.b64encode(image_bytes).decode("ascii"),
            }
        })

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio, "imageSize": image_size},
        },
    }

    logger.info(
        "Gemini image call: model=%s prompt_len=%d images=%d aspect=%s size=%s",
        model, len(prompt), len(image_bytes_list), aspect_ratio, image_size,
    )
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": _gemini_api_key(), "Content-Type": "application/json"},
        json=body,
        timeout=300,
    )
    if not response.ok:
        error_body = response.text[:2000]
        logger.error("Gemini API error %s: %s", response.status_code, error_body)
        raise RuntimeError(f"Gemini API error {response.status_code}: {error_body}")
    return _sanitize_image_bytes(_extract_gemini_image(response.json()))


def _compose_edit(prompt: str, image_bytes_list: List[bytes]) -> bytes:
    """Route a composition image edit to the configured backend (gemini | openai)."""
    if COMPOSITION_BACKEND == "openai":
        return _openai_product_placement_edit(prompt, image_bytes_list)
    return _gemini_image_edit(prompt, image_bytes_list)


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


def _make_plain_wall(wall_color: str = "#F3EFE8", size: Tuple[int, int] = (1024, 768)) -> bytes:
    """A solid wall-colour elevation, used as a placeholder for a wall that has no design so the
    fixed [floor, N, S, E, W] image order stays intact in the single-shot composition call."""
    try:
        rgb = hex_to_rgb(wall_color)
    except Exception:
        rgb = (243, 239, 232)
    return pil_to_png_bytes(Image.new("RGB", size, rgb))


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

    # Wall thickness
    wall_cfg = room.get("walls", {})
    wall_t = max(6, int(round(float(wall_cfg.get("thickness") or 0.12) * ppm)))
    wall_fill = (55, 55, 55)

    # Check for polygon room shape (non-rectangular rooms with >4 walls)
    room_polygon_m = room.get("polygon")  # list of [x_m, y_m] vertices
    if room_polygon_m and isinstance(room_polygon_m, list) and len(room_polygon_m) >= 3:
        # Convert polygon vertices from metres to pixel coordinates
        poly_pts = []
        for pt in room_polygon_m:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                px = ox + float(pt[0]) * ppm
                py = oy + float(pt[1]) * ppm
                poly_pts.append((px, py))
        if len(poly_pts) >= 3:
            # Floor fill — polygon
            draw.polygon(poly_pts, fill=(240, 228, 210))
            # Wall outlines — thick lines along each polygon edge
            for i in range(len(poly_pts)):
                p1 = poly_pts[i]
                p2 = poly_pts[(i + 1) % len(poly_pts)]
                draw.line([p1, p2], fill=wall_fill, width=wall_t)
        else:
            # Malformed polygon — fall back to rectangle
            draw.rectangle([ox, oy, ox + rw_px, oy + rh_px], fill=(240, 228, 210))
            draw.rectangle([ox - wall_t, oy - wall_t, ox + rw_px + wall_t, oy], fill=wall_fill)
            draw.rectangle([ox - wall_t, oy + rh_px, ox + rw_px + wall_t, oy + rh_px + wall_t], fill=wall_fill)
            draw.rectangle([ox - wall_t, oy - wall_t, ox, oy + rh_px + wall_t], fill=wall_fill)
            draw.rectangle([ox + rw_px, oy - wall_t, ox + rw_px + wall_t, oy + rh_px + wall_t], fill=wall_fill)
    else:
        # Standard rectangular room
        draw.rectangle([ox, oy, ox + rw_px, oy + rh_px], fill=(240, 228, 210))

        # 25%/50%/75% grid lines on both axes
        grid_color = (210, 200, 185)
        for pct in (0.25, 0.50, 0.75):
            gx = ox + int(round(rw_px * pct))
            gy = oy + int(round(rh_px * pct))
            draw.line([(gx, oy), (gx, oy + rh_px)], fill=grid_color, width=1)
            draw.line([(ox, gy), (ox + rw_px, gy)], fill=grid_color, width=1)

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

        # For polygon rooms: skip rectangular clamping, render at actual coordinates
        # (frontend already constrains products inside the polygon)
        if room_polygon_m and isinstance(room_polygon_m, list) and len(room_polygon_m) >= 3:
            x0 = cx_px - hw
            y0 = cy_px - hd
            x1 = cx_px + hw
            y1 = cy_px + hd
        else:
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


_COMPASS_FLIP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _mirror_payload_180(payload: dict) -> dict:
    """Return a deep copy of the floor-plan payload with all product positions,
    rotations, and opening positions mirrored 180° around the room centre.
    Used to build the Floor B prompt so the text geometry matches the
    180°-rotated guide image exactly — no contradictory signals.
    """
    mirrored = copy.deepcopy(payload)
    room = mirrored.get("room", {})
    room_w = float(room.get("width") or 0)
    room_d = float(room.get("length") or 0)

    # Mirror polygon room vertices if present
    polygon = mirrored.get("room", {}).get("polygon")
    if polygon and isinstance(polygon, list):
        mirrored["room"]["polygon"] = [
            [round(room_w - float(pt[0]), 3), round(room_d - float(pt[1]), 3)]
            for pt in polygon if isinstance(pt, (list, tuple)) and len(pt) >= 2
        ]

    for product in mirrored.get("products", []):
        pos = product.get("position", {})
        if pos:
            pos["x"] = round(room_w - float(pos.get("x") or 0), 3)
            pos["y"] = round(room_d - float(pos.get("y") or 0), 3)
        product["rotation_y"] = (float(product.get("rotation_y") or 0) + 180) % 360

    for opening in mirrored.get("openings", []):
        # Flip compass wall (supports both "wall" and "compass" field names)
        wall = (opening.get("wall") or opening.get("compass") or "").lower()
        flipped = _COMPASS_FLIP.get(wall, wall)
        if "wall" in opening:
            opening["wall"] = flipped
        if "compass" in opening:
            opening["compass"] = flipped

        # Mirror position_from_left: new = wall_length - old_pos - opening_width
        opening_w = float(opening.get("width") or opening.get("width_m") or 0)
        pos_left = float(opening.get("position_from_left") or 0)
        wall_len = room_w if wall in ("north", "south") else room_d
        opening["position_from_left"] = round(max(0.0, wall_len - pos_left - opening_w), 3)

    return mirrored


def _generate_views_sequential(
    payload: dict,
    image_bytes_list: List[bytes],
    size: str = "1024x1024",
    image_order: Optional[List[str]] = None,
    has_guide: bool = False,
    guide_bytes: Optional[bytes] = None,
) -> Tuple[Dict[str, bytes], List[Dict]]:
    """Generate two isometric floor renders: Floor A (standard) and Floor B (rotated guide).

    Floor B is produced by rotating the guide image 180° and re-running the same edit call.
    This gives the dollhouse compositor a perspective-matched floor reference for each
    cutaway view (front uses Floor A, back uses Floor B).
    """
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

    # Floor B: rotate guide 180° AND mirror product positions in the payload so the
    # text geometry and visual guide are fully consistent (no contradictory signals).
    iso_back_bytes: Optional[bytes] = None
    if guide_bytes is not None:
        try:
            rotated_guide = pil_to_png_bytes(Image.open(BytesIO(guide_bytes)).rotate(180))
            back_images = [rotated_guide] + image_bytes_list[1:]
            mirrored_payload = _mirror_payload_180(payload)
            back_prompt = build_floor_plan_prompt(
                mirrored_payload, image_order=image_order, has_guide=has_guide, view="isometric"
            )
            logger.info("Generating back floor view: rotated guide + mirrored payload")
            iso_back_bytes = _openai_product_placement_edit(back_prompt, back_images, size=size)
        except Exception as exc:
            logger.warning("Back floor render failed, falling back to front: %s", exc)
            iso_back_bytes = iso_bytes

    result: Dict[str, bytes] = {"isometric": iso_bytes}
    if iso_back_bytes is not None:
        result["isometric_back"] = iso_back_bytes
    return result, attempt_metrics


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

    room_dict = {
        "width": w,
        "length": h,
        "height": _DEFAULT_WALL_H,
        "flooring": {"type": flooring_type, "material": flooring_material},
        "walls": {"color": wall_color},
    }
    # Pass through polygon vertices for non-rectangular rooms
    room_polygon = room_dimensions.get("polygon")
    if room_polygon and isinstance(room_polygon, list) and len(room_polygon) >= 3:
        room_dict["polygon"] = room_polygon

    payload = {
        "unit": unit,
        "room": room_dict,
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


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Ray-casting point-in-polygon test for [[x, y], ...] vertex list."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _validate_and_clamp_layout(payload: dict) -> dict:
    """Clamp every product position so its footprint stays inside the room.

    Modifies `payload` in-place and returns it for convenience.
    Logs a warning for every coordinate that was corrected.
    For polygon rooms: check if product center is inside polygon; if not, clamp to
    bounding box as fallback with warning log.
    """
    room = payload.get("room", {})
    room_w = float(room.get("width") or 0)
    room_l = float(room.get("length") or 0)
    if not room_w or not room_l:
        return payload

    room_polygon = room.get("polygon")
    has_polygon = (
        room_polygon
        and isinstance(room_polygon, list)
        and len(room_polygon) >= 3
    )

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

        if has_polygon:
            # Polygon room: check if center is inside polygon
            if not _point_in_polygon(x, y, room_polygon):
                # Fallback: clamp to bounding box
                clamped_x = max(half_w, min(room_w - half_w, x))
                clamped_y = max(half_d, min(room_l - half_d, y))
                logger.warning(
                    "Layout validation (polygon): product %s center (%.3f, %.3f) outside polygon, "
                    "clamped to bbox (%.3f, %.3f)",
                    p.get("id", "?"), x, y, clamped_x, clamped_y,
                )
                pos["x"] = round(clamped_x, 3)
                pos["y"] = round(clamped_y, 3)
        else:
            # Rectangular room: standard bounding box clamping
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
    openings: Optional[List[Dict]] = None,
    base_dir: Optional[str] = None,
    floor_image_url_back: Optional[str] = None,
) -> Dict[str, bytes]:
    """Generate TWO opposite open-box (dollhouse) isometric room renders.

    - "front": near SOUTH wall removed; shows NORTH (back) + WEST (left) + EAST (right).
    - "back":  180°-opposite view, near NORTH wall removed; shows SOUTH (back) +
               EAST (left) + WEST (right) — so the wall missing from "front" is visible.
    Together the two views reveal all four walls. Each wall's products are rendered only
    on that wall (never moved between walls).

    Returns {"front": bytes, "back": bytes}.

    floor_image_url:      server-relative path for the front-view floor render.
    floor_image_url_back: optional server-relative path for the back-view floor render
                          (generated from a 180°-rotated guide). Falls back to
                          floor_image_url when not provided.
    wall_image_urls: list of {url, compass|label, wall_id, width_m, height_m}. Each wall
                     is placed by compass direction ("north"/"south"/"east"/"west"),
                     read from "compass" (preferred) or "label".
    openings:        AUTHORITATIVE per-wall architectural openings (ground truth):
                     [{compass, type, position_from_left(0..1), width_m, sill_height}].
                     Used to (a) tell the shell exactly which walls are blank vs. have
                     windows/doors (stops hallucinated windows) and (b) give the QA
                     validator a real reference so it can catch invented openings.
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
    floor_bytes_back = _resolve(floor_image_url_back) if floor_image_url_back else floor_bytes
    if floor_bytes_back is floor_bytes or floor_bytes_back == floor_bytes:
        logger.warning("Composition: floor_bytes_back is identical to floor_bytes — both views use same floor render")
    else:
        logger.info("Composition: floor_bytes=%d bytes, floor_bytes_back=%d bytes (distinct)", len(floor_bytes), len(floor_bytes_back))

    # Build compass → bytes mapping from the provided wall elevation images.
    # Prefer the explicit "compass" field; fall back to "label" for compatibility.
    _VALID_COMPASS = {"north", "south", "east", "west"}
    wall_map: Dict[str, bytes] = {}
    for wi in wall_image_urls:
        compass = (wi.get("compass") or wi.get("label") or "").lower().strip()
        if compass not in _VALID_COMPASS:
            logger.warning(
                "Composition: skipping wall id=%s with non-compass label %r "
                "(expected north/south/east/west)",
                wi.get("wall_id"), compass,
            )
            continue
        wall_map[compass] = _resolve(wi["url"])

    # Group authoritative openings by compass so each wall knows exactly what it has.
    openings_by_wall: Dict[str, List[Dict]] = {}
    for op in (openings or []):
        compass = (op.get("compass") or op.get("label") or "").lower().strip()
        if compass in _VALID_COMPASS:
            openings_by_wall.setdefault(compass, []).append(op)
    logger.info(
        "Composition openings ground truth: %s",
        {k: len(v) for k, v in openings_by_wall.items()} or "none",
    )

    def _assemble_once(view: str, visible_designed: List[str], correction_notes: str) -> bytes:
        # Sequential pipeline (avoids multi-image confusion / hallucination):
        #   1. Build an EMPTY dollhouse shell from the floor (locks camera + furniture +
        #      architectural openings). BOTH views build their shell from the floor render, so
        #      the furniture is always present in each — the opposite corners come from the
        #      shell prompt's camera, not from rotating one composite into the other.
        #   2. Add each designed wall's content ONE wall at a time — the model only ever sees a
        #      single wall elevation per call, so it cannot swap walls, leak a product onto the
        #      wrong wall, or invent content. Runs on the configured backend (Gemini edits are
        #      surgical, so each step changes only the targeted wall).
        shell_prompt = build_dollhouse_shell_prompt(
            view, room_dimensions=room_dimensions or {}, presets=presets or {},
            openings_by_wall=openings_by_wall,
        )
        logger.info("Composition %s: building empty dollhouse shell from floor", view)
        current = _compose_edit(shell_prompt, [floor_bytes])

        for compass in visible_designed:
            add_prompt = build_dollhouse_add_wall_prompt(
                view, compass,
                room_dimensions=room_dimensions or {},
                presets=presets or {},
                correction_notes=correction_notes,
            )
            logger.info("Composition %s: adding %s wall content", view, compass)
            current = _compose_edit(add_prompt, [current, wall_map[compass]])
        return current

    def _assemble_geometric(view: str, visible_designed: List[str], correction_notes: str) -> bytes:
        # GUIDE-ANCHORED backend. The room's structure (camera, the exact two walls, floor diamond,
        # and openings) is COMPUTED and rendered as a flat geometry guide. The model only ever
        # renders ONTO that locked structure, so it cannot invent an extra wall, mirror an
        # elevation, or render the two views from the same angle — but, unlike the old homography
        # warp, every pixel is photorealistically rendered (no stickered/flat walls, no seam
        # speckle, no floor-strip bleed). Pipeline:
        #   1. Compute the open-box geometry for this view and render a flat GEOMETRY GUIDE
        #      (correct camera + the two far walls + floor diamond + opening rectangles).
        #   2. AI furnishes the guide's floor from the floor render — the guide locks the camera
        #      and wall planes, so this is limited to placing furniture; walls stay bare.
        #   3. Add each wall's content ONE wall at a time: the model sees the furnished render plus
        #      a single flat elevation and paints that elevation's decor onto the matching (already
        #      visible) wall plane, foreshortened and lit to match. Single-elevation-per-call +
        #      the guide-locked left/right wall planes prevent any wall swap or product leak.
        guide_w, guide_h = 1024, 768  # 4:3, matches COMPOSITION_ASPECT_RATIO
        geo = geo_mod.compute_dollhouse_geometry(
            room_dimensions or {}, view, guide_w, guide_h,
            openings_by_wall=openings_by_wall,
        )
        guide_img = geo_mod.render_guide(
            geo, guide_w, guide_h,
            wall_color=(presets or {}).get("wall_color", "#F3EFE8"),
        )
        furnish_prompt = build_geometric_furnish_prompt(
            view, room_dimensions=room_dimensions or {}, presets=presets or {},
            openings_by_wall=openings_by_wall,
        )
        logger.info("Composition %s (geometric): furnishing computed guide from floor", view)
        current = _compose_edit(furnish_prompt, [pil_to_png_bytes(guide_img), floor_bytes])

        # Paint each wall's decor onto its (guide-locked) plane, one elevation at a time.
        for compass in visible_designed:
            add_prompt = build_dollhouse_add_wall_prompt(
                view, compass,
                room_dimensions=room_dimensions or {},
                presets=presets or {},
                correction_notes=correction_notes,
            )
            logger.info("Composition %s (geometric): adding %s wall content onto locked plane", view, compass)
            current = _compose_edit(add_prompt, [current, wall_map[compass]])
        return current

    def _run_qa_loop(view: str, assemble_fn, visible_order: List[str]) -> bytes:
        # Validated best-of-N SELECTION. The composite is the only image the user sees, so it
        # gets a GPT-4o vision QA gate (vs. the floor render, each wall elevation, and the
        # authoritative openings). We generate independent candidates (in parallel) and KEEP THE
        # HIGHEST-SCORING one. We deliberately do NOT run whole-image "refine" edits: in practice
        # those diverged (degraded the composite, e.g. 0.65 → 0.05) instead of converging, so
        # selecting the best of several fresh samples beats trying to correct a bad one.
        visible_designed = [c for c in visible_order if c in wall_map]
        wall_items = [(c, wall_map[c]) for c in visible_designed]

        if not visible_designed or not COMPOSITION_VALIDATION_ENABLED:
            best = assemble_fn("")
            logger.info("Composition %s view done (no QA): walls=%s", view, visible_designed)
            return best

        max_attempts = max(1, COMPOSITION_MAX_ATTEMPTS)
        candidates_per = max(1, COMPOSITION_CANDIDATES_PER_ATTEMPT)
        best_bytes: Optional[bytes] = None
        best_vr: Optional[ValidationResult] = None

        def _validate(cand: bytes) -> ValidationResult:
            return validate_composition_accuracy(
                cand, floor_bytes, wall_items, view,
                visible_walls=list(visible_order),
                openings_by_wall=openings_by_wall,
            )

        def _make_candidate() -> Tuple[bytes, ValidationResult]:
            cand = assemble_fn("")
            return cand, _validate(cand)

        for attempt in range(1, max_attempts + 1):
            # Each attempt is a fresh batch of independent best-of-N candidates (parallel).
            if candidates_per > 1:
                with ThreadPoolExecutor(max_workers=min(candidates_per, 4)) as pool:
                    cand_scored: List[Tuple[bytes, ValidationResult]] = list(
                        pool.map(lambda _: _make_candidate(), range(candidates_per))
                    )
            else:
                cand_scored = [_make_candidate()]

            for _cand_bytes, vr in cand_scored:
                logger.info(
                    "Composition %s attempt %d/%d: score=%.3f passed=%s missing=%d extra/migrated=%d",
                    view, attempt, max_attempts, vr.score, vr.passed,
                    vr.missing_items, vr.extra_items,
                )

            attempt_bytes, attempt_vr = pick_best_candidate(cand_scored)
            if best_vr is None or attempt_vr.score > best_vr.score:
                best_bytes, best_vr = attempt_bytes, attempt_vr

            if best_vr.passed:
                break

        logger.info(
            "Composition %s view done: walls=%s best_score=%.3f",
            view, visible_designed, best_vr.score if best_vr else -1.0,
        )
        return best_bytes

    def _ordered_wall_images(removed_wall: str) -> tuple:
        # Returns (images, blank_compass).
        # Uses CameraViewConfig as the single source of truth for image slot order.
        # Removed wall always gets a blank placeholder — sending its elevation lets
        # the model migrate that content to a blank visible-wall slot.
        cfg = build_camera_view_config(removed_wall)
        wall_color = (presets or {}).get("wall_color", "#F3EFE8")
        placeholder: Optional[bytes] = None
        out: List[bytes] = []
        blank_compass: List[str] = []
        for c in cfg.image_order:
            b = wall_map.get(c)
            if b is None or c == removed_wall:
                if placeholder is None:
                    placeholder = _make_plain_wall(wall_color)
                if c != removed_wall:
                    logger.info("Composition oneshot: %s wall has no elevation, using plain placeholder", c)
                    blank_compass.append(c)
                b = placeholder
            out.append(b)
        return out, blank_compass

    def _assemble_oneshot(removed_wall: str, floor: bytes) -> bytes:
        images_list, blank_compass = _ordered_wall_images(removed_wall)
        prompt = build_dollhouse_oneshot_prompt(removed_wall, blank_compass=blank_compass or None)
        images = [floor] + images_list
        logger.info("Composition oneshot: remove %s wall, %d images, blank: %s",
                    removed_wall, len(images), blank_compass or "none")
        return _compose_edit(prompt, images)

    # ── Compose the two opposite cutaway views. A GPT-4o vision QA gate keeps the best of several
    # fresh candidates (best-of-N selection). ──
    if COMPOSITION_BACKEND == "oneshot":
        logger.info("Composition backend: ONESHOT (floor + four walls in a single call per view)")
        # "front" removes the SOUTH wall (camera outside south, North is the focal back wall);
        # "back" removes the NORTH wall (camera outside north). Together they reveal all walls.
        front_cfg = build_camera_view_config("south")
        back_cfg = build_camera_view_config("north")
        front_fn = lambda: _run_qa_loop("front", lambda n: _assemble_oneshot("south", floor_bytes), front_cfg.visible_walls)
        back_fn  = lambda: _run_qa_loop("back",  lambda n: _assemble_oneshot("north", floor_bytes_back), back_cfg.visible_walls)
    elif COMPOSITION_BACKEND == "geometric":
        logger.info("Composition backend: GEOMETRIC (computed structure guide + guided photoreal fill)")
        front_designed = [c for c in dollhouse_view_walls("front")[1] if c in wall_map]
        back_designed = [c for c in dollhouse_view_walls("back")[1] if c in wall_map]
        front_fn = lambda: _run_qa_loop("front", lambda n: _assemble_geometric("front", front_designed, n), dollhouse_view_walls("front")[1])
        back_fn = lambda: _run_qa_loop("back", lambda n: _assemble_geometric("back", back_designed, n), dollhouse_view_walls("back")[1])
    else:
        front_designed = [c for c in dollhouse_view_walls("front")[1] if c in wall_map]
        back_designed = [c for c in dollhouse_view_walls("back")[1] if c in wall_map]
        front_fn = lambda: _run_qa_loop("front", lambda n: _assemble_once("front", front_designed, n), dollhouse_view_walls("front")[1])
        back_fn = lambda: _run_qa_loop("back", lambda n: _assemble_once("back", back_designed, n), dollhouse_view_walls("back")[1])

    # The two views are fully independent — run them concurrently so total wall-clock is one
    # view's latency, not two. (Each view internally also parallelises its best-of-N candidates.)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_front = pool.submit(front_fn)
        fut_back = pool.submit(back_fn)
        front_bytes = fut_front.result()
        back_bytes = fut_back.result()
    return {"front": front_bytes, "back": back_bytes}
