import base64
import logging
import os
import re
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from prompt import build_placement_prompt

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

    base_bytes = download_bytes(base_image_url)
    guide_bytes = download_bytes(guide_image_url)
    base_img = pil_from_bytes(base_bytes)
    guide_img = pil_from_bytes(guide_bytes)

    # Normalize products + collect unique hex colors
    cleaned_products = []
    unique_hex = set()

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
        unique_hex.add(hc)

    unique_hex = sorted(unique_hex)
    target_colors_rgb = {hc: hex_to_rgb(hc) for hc in unique_hex}

    # Build color masks
    color_masks = guide_to_color_masks(
        guide_img=guide_img,
        target_colors_rgb=target_colors_rgb,
        tol=color_tol,
        min_area=min_area,
    )

    # Retry missing colors with relaxed tolerances
    missing = [hc for hc in unique_hex if hc not in color_masks]
    for hc in missing:
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
    iter_no = 1

    while i < len(cleaned_products):
        batch = cleaned_products[i:i + batch_size]
        batch_items = []
        batch_urls = []
        editable_colors = []

        for p in batch:
            hc = p["hex_color"]
            if hc not in color_masks:
                logger.warning("No color mask found for hex %s — skipping product %s", hc, p.get("product_id"))
                continue
            editable_colors.append(hc)
            batch_items.append({
                "hex_color": hc,
                "product_id": p.get("product_id"),
                "dims": p.get("dims") or p.get("dimensions"),
                "rotation": p.get("rotation") or 0,
                "image_url": p.get("image_url"),
            })
            batch_urls.append(p["image_url"])

        if not batch_items:
            i += len(batch)
            iter_no += 1
            continue

        prompt = build_placement_prompt(
            batch_items=batch_items,
            room_dimensions=room_dimensions,
            presets=presets,
            generation_type=generation_type,
        )

        image_bytes_list = [pil_to_png_bytes(current_base), guide_bytes]
        for url in batch_urls:
            image_bytes_list.append(download_bytes(url))

        generated_bytes = _openai_product_placement_edit(
            prompt=prompt,
            image_bytes_list=image_bytes_list,
        )
        gen_img = pil_from_bytes(generated_bytes)
        current_base = gen_img

        i += len(batch)
        iter_no += 1

    return pil_to_png_bytes(current_base)
