import base64
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw

from prompt import build_placement_prompt, build_floor_plan_prompt

logger = logging.getLogger(__name__)

MODEL = "gpt-image-2"
COLOR_TOL = 45
MIN_AREA = 300


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
    head = b[:256].lstrip()
    if head.startswith(b"<svg") or head.startswith(b"<?xml") or b"<svg" in head:
        raise ValueError(
            "SVG images are not supported by the OpenAI API. "
            "Provide a real product photo URL (PNG, JPEG, or WebP)."
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

    response = requests.post(
        "https://api.openai.com/v1/images/edits",
        headers={"Authorization": f"Bearer {_openai_api_key()}"},
        data={"model": model, "prompt": prompt, "n": "1", "size": size},
        files=files,
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    return base64.b64decode(payload["data"][0]["b64_json"])


# ── Color mask utilities ────────────────────────────────────────────────────

def extract_connected_components(binary: np.ndarray) -> List[np.ndarray]:
    H, W = binary.shape
    visited = np.zeros((H, W), dtype=bool)
    comps = []

    for y in range(H):
        for x in range(W):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            m = np.zeros((H, W), dtype=bool)
            ys, xs = zip(*coords)
            m[np.array(ys), np.array(xs)] = True
            comps.append(m)

    return comps


def mask_bool_to_L(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def guide_to_color_masks(
    guide_img: Image.Image,
    target_colors_rgb: Dict[str, Tuple[int, int, int]],
    tol: int = COLOR_TOL,
    min_area: int = MIN_AREA,
) -> Dict[str, np.ndarray]:
    rgb = np.array(guide_img.convert("RGB"))
    out: Dict[str, np.ndarray] = {}

    for hex_color, (r, g, b) in target_colors_rgb.items():
        binary = (
            (np.abs(rgb[..., 0] - r) <= tol) &
            (np.abs(rgb[..., 1] - g) <= tol) &
            (np.abs(rgb[..., 2] - b) <= tol)
        )
        comps = extract_connected_components(binary)
        comps = [m for m in comps if int(m.sum()) >= min_area]
        if not comps:
            continue
        merged = np.zeros(binary.shape, dtype=bool)
        for m in comps:
            merged |= m
        out[hex_color] = merged

    return out


def lock_all_except_color_regions(
    prev_base: Image.Image,
    gen_img: Image.Image,
    color_masks: Dict[str, np.ndarray],
    editable_hex_colors: List[str],
) -> Image.Image:
    prev_base = prev_base.convert("RGBA")
    gen_img = gen_img.convert("RGBA")
    if prev_base.size != gen_img.size:
        gen_img = gen_img.resize(prev_base.size, Image.Resampling.LANCZOS)

    out = gen_img.copy()
    editable_set = {normalize_hex(c) for c in editable_hex_colors}

    for hex_color, m in color_masks.items():
        if normalize_hex(hex_color) in editable_set:
            continue
        out.paste(prev_base, (0, 0), mask_bool_to_L(m))

    union = np.zeros(next(iter(color_masks.values())).shape, dtype=bool)
    for m in color_masks.values():
        union |= m
    out.paste(prev_base, (0, 0), mask_bool_to_L(~union))

    return out


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

_GUIDE_COLORS = [
    "#E05C5C", "#3DA8C6", "#4BAF7A", "#D4A017",
    "#9B59B6", "#E67E22", "#16A085", "#C0392B",
]


def _assign_guide_colors(products: list) -> list:
    """Return a copy of products with hex_color auto-assigned where missing."""
    result = []
    for i, p in enumerate(products):
        p = dict(p)
        if not p.get("hex_color"):
            p["hex_color"] = _GUIDE_COLORS[i % len(_GUIDE_COLORS)]
        result.append(p)
    return result


def _make_floor_plan_guide(payload: dict, canvas_size: int = 1024) -> bytes:
    """
    Programmatically draw a 2D top-down floor plan at exact scale:
    - Room walls and architectural openings
    - Each product's floor footprint as a coloured filled rectangle

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

    # Wall thickness
    wall_cfg = room.get("walls", {})
    wall_t = max(6, int(round(float(wall_cfg.get("thickness") or 0.12) * ppm)))
    wall_fill = (55, 55, 55)

    # Solid walls (draw as four rectangles)
    draw.rectangle([ox - wall_t, oy - wall_t, ox + rw_px + wall_t, oy], fill=wall_fill)          # north
    draw.rectangle([ox - wall_t, oy + rh_px, ox + rw_px + wall_t, oy + rh_px + wall_t], fill=wall_fill)  # south
    draw.rectangle([ox - wall_t, oy - wall_t, ox, oy + rh_px + wall_t], fill=wall_fill)          # west
    draw.rectangle([ox + rw_px, oy - wall_t, ox + rw_px + wall_t, oy + rh_px + wall_t], fill=wall_fill)  # east

    # Openings — cut gaps in walls
    for o in payload.get("openings", []):
        wall = (o.get("wall") or "").lower()
        pos = float(o.get("position_from_left") or 0)
        ow = float(o.get("width") or 0)
        otype = (o.get("type") or "").lower()
        gap_fill = (240, 228, 210) if otype == "door" else (180, 215, 240)

        pos_px = int(round(pos * ppm))
        ow_px = int(round(ow * ppm))

        if wall == "north":
            draw.rectangle([ox + pos_px, oy - wall_t, ox + pos_px + ow_px, oy], fill=gap_fill)
        elif wall == "south":
            draw.rectangle([ox + pos_px, oy + rh_px, ox + pos_px + ow_px, oy + rh_px + wall_t], fill=gap_fill)
        elif wall == "west":
            draw.rectangle([ox - wall_t, oy + pos_px, ox, oy + pos_px + ow_px], fill=gap_fill)
        elif wall == "east":
            draw.rectangle([ox + rw_px, oy + pos_px, ox + rw_px + wall_t, oy + pos_px + ow_px], fill=gap_fill)

    # Product footprints — coloured rectangles at exact position and size
    for p in payload.get("products", []):
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

        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(0, 0, 0), width=1)

    return pil_to_png_bytes(img)


_FLOOR_VIEWS = ["isometric", "front", "corner"]


def _generate_views_sequential(
    payload: dict,
    image_bytes_list: List[bytes],
    size: str = "1024x1024",
    image_order: Optional[List[str]] = None,
    has_guide: bool = False,
) -> Dict[str, bytes]:
    """
    Two-step generation:
      1. Generate isometric from JSON + product photos (ground truth layout).
      2. Generate front + corner in parallel using the isometric result as IMAGE 1,
         so both derived views are anchored to the same committed layout.
    """
    # ── Step 1: isometric ────────────────────────────────────────────────────
    iso_prompt = build_floor_plan_prompt(
        payload, image_order=image_order, has_guide=has_guide, view="isometric"
    )
    logger.info("Generating isometric view, prompt %d chars", len(iso_prompt))
    iso_bytes = _openai_product_placement_edit(iso_prompt, image_bytes_list, size=size)

    # ── Step 2: front + corner derived from isometric ────────────────────────
    # IMAGE 1 = isometric result; IMAGE 2+ = same product reference photos
    derived_images = [iso_bytes] + image_bytes_list[1:]

    def _gen_derived(view: str) -> tuple:
        prompt = build_floor_plan_prompt(
            payload, image_order=image_order, view=view, iso_base=True
        )
        logger.info("Generating %s view from isometric, prompt %d chars", view, len(prompt))
        return view, _openai_product_placement_edit(prompt, derived_images, size=size)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_gen_derived, v): v for v in ["front", "corner"]}
        results = {"isometric": iso_bytes}
        for future in as_completed(futures):
            view_name, img_bytes = future.result()
            results[view_name] = img_bytes
    return results


def generate_floor_plan(payload: dict, size: str = "1024x1024") -> Dict[str, bytes]:
    """
    Generate isometric, front, and corner renders in parallel from a structured JSON payload.
    Returns {"isometric": bytes, "front": bytes, "corner": bytes}.
    """
    products = payload.get("products", [])
    products_with_images = [(p["id"], p["image_url"]) for p in products if p.get("image_url")]

    if products_with_images:
        image_order = [pid for pid, _ in products_with_images]
        image_bytes_list = [_make_blank_canvas(size)]
        for pid, url in products_with_images:
            image_bytes_list.append(download_bytes(url))
        return _generate_views_sequential(payload, image_bytes_list, size=size, image_order=image_order)

    # No product images — isometric only via text generation
    prompt = build_floor_plan_prompt(payload, view="isometric")
    logger.info("Floor plan generations prompt (%d chars):\n%s", len(prompt), prompt)
    return {"isometric": _openai_generate(prompt, size=size)}


# ── Surface Editor → floor-plan payload converter ───────────────────────────

def _batch_to_fp_payload(
    cleaned_products: List[Dict],
    room_dimensions: Dict,
    presets: Dict,
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
    for item in cleaned_products:
        pid = item.get("hex_color", str(item.get("product_id", "")))
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

        x_m = item.get("x_m")
        y_m = item.get("y_m")

        entry = {
            "id": pid,
            "hex_color": item.get("hex_color", ""),
            "image_url": item.get("image_url", ""),
            "dimensions": {"width": dim_w, "depth": dim_d, "height": 0},
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
        "openings": [],
        "products": products,
        "lighting": {"type": lighting_type, "sun_direction": "north", "intensity": 0.85},
        "camera_views": [{"name": "isometric_top_view", "fov": 35}],
    }
    if room_type:
        payload["scene_id"] = room_type
    if decor_style:
        payload["decor_style"] = decor_style

    return payload, image_order


# ── Main generation function ────────────────────────────────────────────────

def generate_product_placement(
    base_image_url: str,
    guide_image_url: str,
    products: List[Dict],
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
    generation_type: str = "floor",
    color_tol: int = 60,
    min_area: int = 200,
    batch_size: int = 20,
) -> bytes:
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

    # ── Floor type: JSON geometry + appearance prompt, 3 views in parallel ──────
    if generation_type == "floor":
        fp_payload, image_order = _batch_to_fp_payload(
            cleaned_products, room_dimensions or {}, presets or {}
        )
        logger.info("Surface Editor floor: %d product(s)", len(cleaned_products))
        image_bytes_list = [_make_blank_canvas()]
        for p in cleaned_products:
            image_bytes_list.append(download_bytes(p["image_url"]))
        return _generate_views_sequential(fp_payload, image_bytes_list, image_order=image_order)

    # ── Wall type: original guide-blob batch approach ─────────────────────────
    guide_bytes = download_bytes(guide_image_url)
    base_bytes = download_bytes(base_image_url)
    base_img = pil_from_bytes(base_bytes)
    guide_img = pil_from_bytes(guide_bytes)

    unique_hex = sorted({p["hex_color"] for p in cleaned_products})
    target_colors_rgb = {hc: hex_to_rgb(hc) for hc in unique_hex}

    color_masks = guide_to_color_masks(
        guide_img=guide_img,
        target_colors_rgb=target_colors_rgb,
        tol=color_tol,
        min_area=min_area,
    )
    for hc in [hc for hc in unique_hex if hc not in color_masks]:
        try:
            extra = guide_to_color_masks(
                guide_img=guide_img,
                target_colors_rgb={hc: hex_to_rgb(hc)},
                tol=color_tol + 20,
                min_area=max(10, min_area - 50),
            )
            if hc in extra:
                color_masks[hc] = extra[hc]
        except Exception:
            pass

    current_base = base_img
    i = 0
    while i < len(cleaned_products):
        batch = cleaned_products[i:i + batch_size]
        batch_items, batch_urls = [], []
        for p in batch:
            hc = p["hex_color"]
            if hc not in color_masks:
                logger.warning("No color mask for hex %s — skipping product %s", hc, p.get("product_id"))
                continue
            batch_items.append({
                "hex_color": hc,
                "product_id": p.get("product_id"),
                "dims": p.get("dims") or p.get("dimensions"),
                "rotation": p.get("rotation") or 0,
                "image_url": p.get("image_url"),
                "x_m": p.get("x_m"),
                "y_m": p.get("y_m"),
            })
            batch_urls.append(p["image_url"])

        if batch_items:
            prompt = build_placement_prompt(
                batch_items=batch_items,
                room_dimensions=room_dimensions,
                presets=presets,
                generation_type="wall",
            )
            image_bytes_list = [pil_to_png_bytes(current_base), guide_bytes]
            for url in batch_urls:
                image_bytes_list.append(download_bytes(url))
            current_base = pil_from_bytes(
                _openai_product_placement_edit(prompt, image_bytes_list)
            )

        i += len(batch)

    return pil_to_png_bytes(current_base)
