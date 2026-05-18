import json
import re
from typing import Dict, List, Optional


def _parse_room_wh(room_dimensions: Dict) -> tuple:
    rd = room_dimensions or {}
    try:
        w = float(rd.get("width") or 0)
        h = float(rd.get("height") or 0)
    except (ValueError, TypeError):
        w = h = 0.0
    if w > 0 and h > 0:
        return w, h
    raw = str(rd.get("raw") or "")
    nums = re.findall(r"[\d.]+", raw)
    if len(nums) >= 2:
        return float(nums[1]), float(nums[0])
    return 0.0, 0.0


_OUTPUT_PX = 1024


def _build_scale_lines(batch_items: List[Dict], room_w_m: float, room_h_m: float) -> tuple:
    ppm_x = _OUTPUT_PX / room_w_m if room_w_m > 0 else 0.0
    ppm_y = _OUTPUT_PX / room_h_m if room_h_m > 0 else 0.0
    room_area_m2 = room_w_m * room_h_m if (room_w_m and room_h_m) else 0.0
    total_prod_area = 0.0
    lines = []

    for idx, item in enumerate(batch_items, start=1):
        hc = item.get("hex_color", "")
        pid = item.get("product_id", "")
        dims_str = (item.get("dims") or item.get("dimensions") or "").strip()
        rotation = int(item.get("rotation") or 0)

        scale_line = ""
        dims_nums = re.findall(r"[\d.]+", dims_str) if dims_str else []
        if len(dims_nums) >= 2 and room_w_m and room_h_m:
            dim1_m = float(dims_nums[0])
            dim2_m = float(dims_nums[1])
            if rotation in (90, 270):
                prod_w_m, prod_h_m = dim1_m, dim2_m
            else:
                prod_w_m, prod_h_m = dim2_m, dim1_m
            w_pct = prod_w_m / room_w_m * 100
            h_pct = prod_h_m / room_h_m * 100
            w_px = max(4, int(round(prod_w_m * ppm_x)))
            h_px = max(4, int(round(prod_h_m * ppm_y)))
            prod_area = dim1_m * dim2_m
            total_prod_area += prod_area
            area_pct = prod_area / room_area_m2 * 100 if room_area_m2 else 0
            scale_line = (
                f"Size: ~{w_pct:.0f}% of room width × ~{h_pct:.0f}% of room height "
                f"= approximately {w_px}×{h_px} px in the {_OUTPUT_PX}×{_OUTPUT_PX} output "
                f"(covers only {area_pct:.1f}% of the total floor area). "
            )
        elif dims_str:
            scale_line = f"Physical size: {dims_str}. "

        if rotation:
            rot_line = (
                f"Orientation: rotate {rotation}° clockwise from the product's natural top-down view. "
                f"The px dimensions above already account for this rotation. "
            )
        else:
            rot_line = "Orientation: natural top-down view (0° rotation). "

        lines.append(
            f"{idx}. PRODUCT_IMAGE {idx} (id={pid}): "
            f"{scale_line}"
            f"{rot_line}"
            f"Place ONLY in GUIDE region matching hex {hc}. "
            f"Do not substitute a different product type."
        )

    return lines, total_prod_area, room_area_m2


def build_floor_placement_prompt(
    batch_items: List[Dict],
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
) -> str:
    rd = room_dimensions or {}
    room_w_m, room_h_m = _parse_room_wh(rd)

    placement_lines, total_prod_area, room_area_m2 = _build_scale_lines(
        batch_items, room_w_m, room_h_m
    )

    if room_area_m2 and total_prod_area:
        covered_pct = total_prod_area / room_area_m2 * 100
        empty_pct = 100 - covered_pct
        scale_header = (
            f"SCALE ANCHOR: all {len(batch_items)} products together cover only "
            f"{covered_pct:.1f}% of the room's {room_area_m2:.0f} m² floor. "
            f"The remaining {empty_pct:.0f}% of the floor MUST be visible as empty space. "
            f"Do NOT enlarge any product to fill empty floor.\n"
        )
    else:
        scale_header = ""

    placement_guide = scale_header + "\n".join(placement_lines)

    unit = (rd.get("unit") or "").strip() if isinstance(rd, dict) else ""
    width = (str(rd.get("width")).strip() if isinstance(rd, dict) and rd.get("width") is not None else "")
    height = (str(rd.get("height")).strip() if isinstance(rd, dict) and rd.get("height") is not None else "")
    raw = (rd.get("raw") or "").strip() if isinstance(rd, dict) else ""

    if raw:
        room_dim_sentence = (
            f"The room spans approximately {raw}"
            + (f" ({unit})." if unit else ".")
        )
    elif unit and width and height:
        room_dim_sentence = f"The room spans approximately {width} × {height} {unit}."
    else:
        room_dim_sentence = (
            "Use wall boundaries in BASE_IMAGE to infer room footprint if explicit room size is missing."
        )

    pr = presets or {}
    presets_line = ""
    if isinstance(pr, dict) and any(pr.get(k) for k in ("room_type", "decor_style", "theme", "color_palette")):
        parts = []
        if pr.get("room_type"):
            parts.append(f"room type {pr['room_type']}")
        if pr.get("decor_style"):
            parts.append(f"decor style {pr['decor_style']}")
        if pr.get("theme"):
            parts.append(f"theme {pr['theme']}")
        if pr.get("color_palette"):
            palette = pr["color_palette"]
            if isinstance(palette, list):
                palette = ", ".join(str(c) for c in palette)
            parts.append(f"accent palette hex {palette}")
        presets_line = (
            "Room preset (apply clearly to finishes, floor material mood, lighting — never replacing product shapes): "
            + "; ".join(parts)
            + "."
        )

    n = len(batch_items)
    return f"""## Task
Produce ONE photorealistic top bird-eye 3D isometric interior render of the entire room (full footprint in frame, not a tight crop).
Place each listed product in its assigned GUIDE region at the exact stated size and rotation. Keep a spacious, open layout.
{presets_line}

## Room dimensions
{room_dim_sentence}

## Products to place — every field is binding
Attached images: IMAGE 1 = BASE floor plan, IMAGE 2 = GUIDE blobs, then PRODUCT_IMAGES in order (image 3 = product 1, image 4 = product 2, ...).
{placement_guide}

## How to read the attached images
- **IMAGE 1 - BASE**: ground truth for room shape, walls, and every existing opening. Copy it exactly.
- **IMAGE 2 - GUIDE**: flat colored blobs showing WHERE each product goes. Each hex color = one product slot.
- **IMAGES 3+ - PRODUCTS**: the exact product appearance to render in each slot, in the same order as the list above.

## Scale rules
- Each product entry states its final rendered size as a **percentage of room width x room height**. Those figures are mathematically derived from the physical dimensions and must be respected precisely.
- A product at 8% of room width is a small object -- show generous empty floor around it.
- Never upscale any product to fill empty space or make the room look fuller.
- Inter-product spacing must match the guide: blobs far apart on the guide stay far apart in the render.

## Rotation rules
- Each product entry states its orientation in degrees clockwise from the product image's natural top-down view.
- 0 degrees = natural top-down orientation as shown in the product image.
- 90 degrees CW = rotate that view 90 degrees clockwise; the object's longer axis now points left-right instead of up-down.
- Apply the rotation before placing in the guide region. The % width/height figures already account for the rotation.

## Photoreal quality
Soft drop-shadows beneath each piece, realistic material textures (fabric, wood, metal), natural top-down ambient light.
No sticker or cutout look. Colored guide marks must vanish -- show natural floor (wood/tile/stone) where they were.

## HARD CONSTRAINTS -- all are mandatory, violation = failure
1. **No new architectural elements** -- do NOT add windows, doors, skylights, columns, arches, or any structural feature not already visible in IMAGE 1. This is the single most critical rule.
2. **No removal of existing architecture** -- do NOT erase or relocate walls, doors, or windows that appear in IMAGE 1.
3. **Exactly {n} products** -- no extra chairs, tables, rugs, lamps, plants, vases, cushions, or accessories of any kind beyond the {n} listed items.
4. **No product substitution** -- every slot must show the product from its designated PRODUCT_IMAGE; do not replace with a generic alternative.
5. **No text or graphics** overlaid on the render -- no dimension lines, labels, arrows, or watermarks.
6. **Same framing as BASE** -- identical room footprint and layout, with a consistent top bird-eye 3D isometric camera angle.

## Data
{json.dumps(batch_items, ensure_ascii=False, indent=2)}

Output: a single photorealistic realistic-looking top bird-eye 3D isometric render, no overlays, no on-image text.
""".strip()


def build_wall_placement_prompt(
    batch_items: List[Dict],
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
) -> str:
    rd = room_dimensions or {}
    room_w_m, room_h_m = _parse_room_wh(rd)

    placement_lines, total_prod_area, room_area_m2 = _build_scale_lines(
        batch_items, room_w_m, room_h_m
    )

    if room_area_m2 and total_prod_area:
        covered_pct = total_prod_area / room_area_m2 * 100
        scale_header = (
            f"SCALE ANCHOR: all {len(batch_items)} products together cover only "
            f"{covered_pct:.1f}% of the {room_area_m2:.1f} m² wall surface. "
            f"The remaining {100 - covered_pct:.0f}% MUST be visible as empty wall.\n"
        )
    else:
        scale_header = ""

    placement_guide = scale_header + "\n".join(placement_lines)

    unit = (rd.get("unit") or "m").strip()
    width = str(rd.get("width") or "")
    height = str(rd.get("height") or "7")
    if width:
        room_dim_sentence = f"The wall is approximately {width} × {height} {unit} (width × height)."
    else:
        room_dim_sentence = "Use wall boundaries in BASE_IMAGE to infer wall dimensions."

    pr = presets or {}
    presets_line = ""
    if any(pr.get(k) for k in ("room_type", "decor_style", "theme", "color_palette")):
        parts = []
        if pr.get("room_type"):
            parts.append(f"room type {pr['room_type']}")
        if pr.get("decor_style"):
            parts.append(f"decor style {pr['decor_style']}")
        if pr.get("theme"):
            parts.append(f"theme {pr['theme']}")
        if pr.get("color_palette"):
            palette = pr["color_palette"]
            if isinstance(palette, list):
                palette = ", ".join(str(c) for c in palette)
            parts.append(f"accent palette {palette}")
        presets_line = (
            "Room preset (apply to wall finish, paint color, lighting mood): "
            + "; ".join(parts) + "."
        )

    n = len(batch_items)
    return f"""## Task
Produce ONE photorealistic front-facing wall elevation render showing the full wall surface from floor to ceiling.
Place each listed product/furniture piece in its assigned GUIDE region at the exact stated size.
The result should look like a professional interior design elevation drawing rendered as a photorealistic scene.
{presets_line}

## Wall dimensions
{room_dim_sentence}

## Products to place — every field is binding
Attached images: IMAGE 1 = BASE wall surface (black & white), IMAGE 2 = GUIDE color blobs, then PRODUCT_IMAGES in order.
{placement_guide}

## How to read the attached images
- IMAGE 1 - BASE: the wall surface outline (width × height). This defines the wall boundaries exactly.
- IMAGE 2 - GUIDE: flat colored blobs showing WHERE each product is placed on the wall. Each hex = one product slot.
- IMAGES 3+ - PRODUCTS: the exact product appearance to render in each slot, in the same order as the list above.

## Placement rules
- Products sit on the floor against the wall, or hang/mount on the wall surface, depending on their type.
- Furniture (sofas, chairs, tables) should appear to rest on the floor in front of the wall.
- Wall-mounted items (art, shelves, TVs, mirrors) should appear mounted on the wall surface.
- All placement positions are determined by the GUIDE blobs — follow them exactly.
- Keep the wall surface visible and realistic (paint, texture) in empty areas.

## Scale rules
- Each product entry states its rendered size as a percentage of wall width × wall height. Respect precisely.
- Never upscale any product to fill empty space.
- Maintain realistic proportions relative to a 7m tall wall.

## Photoreal quality
Soft shadows, realistic material textures, natural front-facing ambient lighting.
No sticker or cutout look. Colored guide marks must vanish — show natural wall surface where they were.

## HARD CONSTRAINTS
1. Exactly {n} products — no extra accessories, decorations, or furniture beyond the {n} listed items.
2. No product substitution — every slot must show its designated PRODUCT_IMAGE.
3. No text, labels, dimension lines, or watermarks overlaid on the render.
4. Wall boundaries must match BASE exactly — do not add doors, windows, or openings not in IMAGE 1.
5. Front-facing view only — no top-down, no fisheye, no angled perspective.

## Data
{json.dumps(batch_items, ensure_ascii=False, indent=2)}

Output: a single photorealistic front-facing wall elevation render, no overlays, no on-image text.
""".strip()


def build_placement_prompt(
    batch_items: List[Dict],
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
    generation_type: str = "floor",
) -> str:
    if generation_type == "wall":
        return build_wall_placement_prompt(batch_items, room_dimensions, presets)
    return build_floor_placement_prompt(batch_items, room_dimensions, presets)
