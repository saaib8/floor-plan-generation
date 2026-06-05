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
        rotation = int(item.get("rotation") or 0)
        x_m = item.get("x_m")
        y_m = item.get("y_m")

        # Parse dimensions — may be a dict (structured metres) or a string
        dims_raw = item.get("dimensions")
        if isinstance(dims_raw, dict):
            dim1_m = float(dims_raw.get("width") or 0)
            dim2_m = float(dims_raw.get("depth") or dims_raw.get("height") or dim1_m)
            dims_str = f"{dim1_m} x {dim2_m}"
        else:
            dims_str = (item.get("dims") or (dims_raw if isinstance(dims_raw, str) else "") or "").strip()
            dims_parsed = re.findall(r"[\d.]+", dims_str) if dims_str else []
            dim1_m = float(dims_parsed[0]) if len(dims_parsed) >= 1 else 0
            dim2_m = float(dims_parsed[1]) if len(dims_parsed) >= 2 else dim1_m

        scale_line = ""
        if dim1_m and dim2_m and room_w_m and room_h_m:
            if rotation in (90, 270):
                prod_w_m, prod_h_m = dim2_m, dim1_m
            else:
                prod_w_m, prod_h_m = dim1_m, dim2_m
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

        pos_pct_str = ""
        if x_m is not None and y_m is not None and room_w_m and room_h_m:
            pct_x = x_m / room_w_m * 100
            pct_y = y_m / room_h_m * 100
            pos_pct_str = (
                f"Center position: {pct_x:.0f}% from left edge, {pct_y:.0f}% from top edge "
                f"({100 - pct_x:.0f}% from right edge, {100 - pct_y:.0f}% from bottom edge). "
            )

        lines.append(
            f"{idx}. PRODUCT_IMAGE {idx} (id={pid}): "
            f"{scale_line}"
            f"{pos_pct_str}"
            f"{rot_line}"
            f"Place ONLY in GUIDE region matching hex {hc}. "
            f"Do not substitute a different product type."
        )

    return lines, total_prod_area, room_area_m2


# ── Legacy batch-placement helpers (kept only for reference, unused) ──────────
def _build_floor_placement_prompt_legacy(
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
        room_dim_sentence = (
            f"The room spans approximately {width} × {height} {unit}.\n"
            f"Coordinate system: origin (0,0) is the TOP-LEFT corner of the surface. "
            f"x increases rightward (0=left edge, {width}=right edge). "
            f"y increases downward (0=top edge, {height}=bottom edge). "
            f"Units: {unit}."
        )
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
7. **Never alter product orientation** -- render every product at exactly the rotation_y stated in the data. Do NOT rotate, flip, or reorient any product even if a different angle looks more natural or correct to you. The stated rotation is authoritative.

## Data
{json.dumps(batch_items, ensure_ascii=False, indent=2)}

Output: a single photorealistic realistic-looking top bird-eye 3D isometric render, no overlays, no on-image text.
""".strip()


def _build_wall_placement_prompt_legacy(
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


def _build_derived_view_prompt(
    payload: dict,
    image_order: Optional[List[str]] = None,
    view: str = "front",
) -> str:
    """Prompt for front/corner views when IMAGE 1 is the isometric render."""
    room = payload.get("room", {})
    fl = room.get("flooring", {})
    walls_cfg = room.get("walls", {})
    lighting = payload.get("lighting", {})

    # Appearance (same as main prompt)
    app: List[str] = ["Ultra-realistic architectural visualization"]
    lt = (lighting.get("type") or "natural_daylight").replace("_", " ")
    if "natural" in lt.lower() or "daylight" in lt.lower():
        app.append("warm natural daylight, soft ambient light from windows")
    floor_bits = [fl.get("material", ""), fl.get("type", "")]
    floor_str = " ".join(b for b in floor_bits if b).strip()
    if floor_str:
        app.append(f"{floor_str} flooring")
    wall_color = walls_cfg.get("color", "")
    if wall_color:
        app.append(f"walls painted {wall_color}")
    app += ["soft drop-shadows", "realistic material textures (fabric, wood, metal)",
            "no sticker or cutout look", "editorial photography quality"]
    appearance = ", ".join(app) + "."

    view_lines = {
        "front": (
            "Eye-level interior perspective — camera at 1.2 m height outside the SOUTH wall "
            "looking straight NORTH, full room width in frame, all furniture fully visible, no cropping."
        ),
        "corner": (
            "Wide-angle 3D corner perspective — camera pulled far back and elevated outside the SW corner "
            "looking diagonally toward the NE corner. Entire room visible, all four walls, "
            "every furniture piece in frame. Do not zoom in."
        ),
    }
    view_line = view_lines[view]

    # Image reference block
    img_order = image_order or []
    ref_lines = [
        "IMAGE 1 is a top-down isometric render of the room — it is the ground truth for "
        "furniture layout, product identities, positions, sizes, and room dimensions. "
        "Re-render this exact scene from the camera angle described above.",
    ]
    if img_order:
        ref_lines.append("Product reference photos — match materials and appearance exactly:")
        for i, pid in enumerate(img_order, start=1):
            ref_lines.append(f"  IMAGE {i + 1} = product id={pid}")
    refs = "\n".join(ref_lines)

    # Openings description for derived views
    openings = payload.get("openings", [])
    openings_line = ""
    if openings:
        o_parts = []
        for o in openings:
            otype = (o.get("type") or "opening").lower()
            wall = (o.get("wall") or "?").upper()
            ow = float(o.get("width") or 0)
            o_parts.append(f"{otype} on {wall} wall ({ow:.1f}m wide)")
        openings_line = (
            "OPENINGS visible in IMAGE 1 that MUST appear in this view: "
            + ", ".join(o_parts) + ". "
            "Render each as a realistic architectural element (door frame/panel, window glass/frame)."
        )

    n = len(payload.get("products", []))
    constraints = (
        f"HARD CONSTRAINTS: "
        f"(1) Render exactly the same {n} furniture item(s) visible in IMAGE 1 — same products, same positions, same layout. "
        f"(2) Match each product's material and finish to its reference photo (IMAGE 2+). "
        f"(3) No extra furniture, accessories, rugs, lamps, or decor not in IMAGE 1. "
        f"(4) No text, labels, dimension lines, or watermarks."
    )

    return "\n\n".join(filter(None, [appearance, view_line, refs, openings_line, constraints])).strip() + (
        "\n\nOutput: one photorealistic render, no overlays, no on-image text."
    )


def _point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute minimum distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5


def _compute_spatial_spec(payload: dict) -> str:
    """Compute explicit wall-gap distances and asymmetry directives for each product.

    Returns a prompt block like:
        SPATIAL PLACEMENT CONSTRAINTS:
        Room shape: wider than deep (9.0m wide x 6.0m deep, ratio 1.5:1), ...
        seating (id=seating): gaps W=0.40m, E=7.50m, ...
        ...
        CRITICAL: Preserve exact spatial asymmetries above. ...

    Returns "" if room data is missing so the caller can safely filter(None, ...).
    """
    room = payload.get("room", {})
    room_w = float(room.get("width") or 0)
    room_l = float(room.get("length") or 0)
    if not room_w or not room_l:
        return ""

    products = payload.get("products", [])
    if not products:
        return ""

    # Detect polygon room
    room_polygon = room.get("polygon")
    is_polygon = (
        room_polygon
        and isinstance(room_polygon, list)
        and len(room_polygon) >= 3
    )

    room_area = room_w * room_l

    if is_polygon:
        n_vertices = len(room_polygon)
        header = (
            f"Room shape: irregular polygon with {n_vertices} vertices "
            f"(bounding box {room_w:.1f}m wide x {room_l:.1f}m deep, "
            f"total bounding area {room_area:.1f} m2). "
            f"The room is NOT rectangular — it has {n_vertices} walls forming an irregular shape."
        )
    else:
        # Room shape description
        if room_w > room_l * 1.05:
            shape = "wider than deep"
        elif room_l > room_w * 1.05:
            shape = "deeper than wide"
        else:
            shape = "roughly square"

        ratio = max(room_w, room_l) / min(room_w, room_l) if min(room_w, room_l) > 0 else 1.0
        header = (
            f"Room shape: {shape} ({room_w:.1f}m wide x {room_l:.1f}m deep, "
            f"ratio {ratio:.1f}:1), total floor area {room_area:.1f} m2."
        )

    lines = ["SPATIAL PLACEMENT CONSTRAINTS:", header]
    any_asymmetric = False
    any_near_wall_gap = False
    wall_touch_threshold = 0.15  # metres — considered "touching" a wall
    near_wall_threshold = 1.0    # metres — close enough that models tend to snap

    _WALL_NAMES = {
        "w": "west (left)",
        "e": "east (right)",
        "n": "north (top/back)",
        "s": "south (bottom/front)",
    }

    for p in products:
        pid = p.get("id", "?")
        cat = (p.get("category") or pid).replace("_", " ")
        dims = p.get("dimensions", {})
        pw = float(dims.get("width") or 0)
        pd = float(dims.get("depth") or 0)
        rot = int(p.get("rotation_y") or 0) % 360

        eff_w, eff_d = (pd, pw) if rot in (90, 270) else (pw, pd)
        half_w = eff_w / 2
        half_d = eff_d / 2

        pos = p.get("position", {})
        cx = float(pos.get("x") or 0)
        cy = float(pos.get("y") or 0)

        footprint_pct = (eff_w * eff_d) / room_area * 100 if room_area else 0
        width_pct = eff_w / room_w * 100 if room_w else 0

        if is_polygon:
            # Polygon room: compute distance to nearest wall segment
            min_dist = float("inf")
            for i in range(len(room_polygon)):
                x1, y1 = float(room_polygon[i][0]), float(room_polygon[i][1])
                x2, y2 = float(room_polygon[(i + 1) % len(room_polygon)][0]), float(room_polygon[(i + 1) % len(room_polygon)][1])
                d = _point_to_segment_distance(cx, cy, x1, y1, x2, y2)
                if d < min_dist:
                    min_dist = d

            # Subtract half-product to get edge-to-wall distance
            nearest_wall_gap = max(0, min_dist - max(half_w, half_d))

            line = (
                f"  {cat} (id={pid}): nearest wall distance={nearest_wall_gap:.2f}m "
                f"(polygon-shaped room). Footprint: {footprint_pct:.1f}% of bounding area."
            )
            lines.append(line)

            touching = []
            near_wall_gaps = []
            if nearest_wall_gap <= wall_touch_threshold:
                touching.append("nearest wall")
            elif nearest_wall_gap <= near_wall_threshold:
                near_wall_gaps.append(("nearest", nearest_wall_gap))

            # Use rectangular gaps for asymmetry detection (still useful for bounding box)
            gap_w = max(0, cx - half_w)
            gap_e = max(0, room_w - cx - half_w)
            gap_n = max(0, cy - half_d)
            gap_s = max(0, room_l - cy - half_d)
            gaps = {"w": gap_w, "e": gap_e, "n": gap_n, "s": gap_s}
        else:
            # Rectangular room: compute N/S/E/W gaps
            gap_w = max(0, cx - half_w)           # west gap
            gap_e = max(0, room_w - cx - half_w)  # east gap
            gap_n = max(0, cy - half_d)           # north gap
            gap_s = max(0, room_l - cy - half_d)  # south gap

            gaps = {"w": gap_w, "e": gap_e, "n": gap_n, "s": gap_s}

            touching = []
            near_wall_gaps = []  # walls that are close but NOT touching — danger zone
            for key, gap in gaps.items():
                if gap <= wall_touch_threshold:
                    touching.append(_WALL_NAMES[key])
                elif gap <= near_wall_threshold:
                    near_wall_gaps.append((key, gap))

            line = (
                f"  {cat} (id={pid}): gaps W={gap_w:.2f}m, E={gap_e:.2f}m, "
                f"N={gap_n:.2f}m, S={gap_s:.2f}m."
            )
            lines.append(line)

        if touching:
            lines.append(
                f"    AGAINST {', '.join(touching)} — keep it flush against "
                f"{'these walls' if len(touching) > 1 else 'this wall'}. "
                f"Footprint: {footprint_pct:.1f}% of floor, {width_pct:.0f}% of room width."
            )

        if near_wall_gaps:
            any_near_wall_gap = True
            for key, gap in near_wall_gaps:
                if key == "nearest":
                    # Polygon room — no cardinal direction available
                    lines.append(
                        f"    GAP to nearest wall = {gap:.2f}m — this gap is INTENTIONAL. "
                        f"Do NOT push this product flush against the wall. "
                        f"Maintain visible floor/space of {gap:.2f}m between the product edge and the wall."
                    )
                else:
                    wall_name = _WALL_NAMES[key]
                    lines.append(
                        f"    GAP to {wall_name} wall = {gap:.2f}m — this gap is INTENTIONAL. "
                        f"Do NOT push this product against the {wall_name} wall. "
                        f"Maintain visible floor/space ({gap:.1f}m ≈ {gap / (room_l if key in ('n', 's') else room_w) * 100:.0f}% of room {'depth' if key in ('n', 's') else 'width'}) between the product edge and the wall."
                    )

        if not touching and not near_wall_gaps:
            lines.append(f"    Free-standing (well away from all walls). Footprint: {footprint_pct:.1f}% of floor, {width_pct:.0f}% of room width.")

        # Asymmetry detection — left/right
        if gap_w > 0.01 and gap_e > 0.01:
            lr_ratio = max(gap_w, gap_e) / min(gap_w, gap_e) if min(gap_w, gap_e) > 0.01 else 999
            if lr_ratio > 1.5:
                closer_wall = "west" if gap_w < gap_e else "east"
                lines.append(f"    Asymmetric left-right: closer to {closer_wall} wall -- do NOT center horizontally.")
                any_asymmetric = True

        # Asymmetry detection — front/back
        if gap_n > 0.01 and gap_s > 0.01:
            fb_ratio = max(gap_n, gap_s) / min(gap_n, gap_s) if min(gap_n, gap_s) > 0.01 else 999
            if fb_ratio > 1.5:
                closer_wall = "north" if gap_n < gap_s else "south"
                lines.append(f"    Asymmetric front-back: closer to {closer_wall} wall -- do NOT center vertically.")
                any_asymmetric = True

    # ── Gap-to-openings (doors/windows) for products on the same wall ──────
    openings = payload.get("openings", [])
    if openings:
        for p in products:
            pid = p.get("id", "?")
            pos = p.get("position", {})
            cx = float(pos.get("x") or 0)
            cy = float(pos.get("y") or 0)
            dims = p.get("dimensions", {})
            pw = float(dims.get("width") or 0)
            pd_val = float(dims.get("depth") or 0)
            rot = int(p.get("rotation_y") or 0) % 360
            eff_w, eff_d = (pd_val, pw) if rot in (90, 270) else (pw, pd_val)

            for o in openings:
                o_wall = (o.get("wall") or "").lower()
                o_pos = float(o.get("position_from_left") or 0)
                o_width = float(o.get("width") or 0)
                o_type = (o.get("type") or "opening").lower()
                o_center = o_pos + o_width / 2

                # Compute distance between product edge and opening center
                dist = None
                axis = ""
                if o_wall in ("north", "south"):
                    dist = abs(cx - o_center)
                    axis = "horizontally"
                elif o_wall in ("west", "east"):
                    dist = abs(cy - o_center)
                    axis = "vertically"

                if dist is not None and dist < max(room_w, room_l):
                    lines.append(
                        f"    {pid} is {dist:.2f}m {axis} from {o_type} on {o_wall} wall — preserve this gap exactly."
                    )

    lines.append("")
    lines.append(
        "CRITICAL: Preserve exact spatial positioning above. Do NOT redistribute or\n"
        "center products to look 'balanced'. Gaps are computed from user input.\n"
        "If a product has a gap to a wall, that gap MUST appear in the render.\n"
        + ("If a product has a gap to a window or door, that gap MUST be preserved.\n" if openings else "")
        + "If a product is flush against a wall, it MUST stay flush.\n"
        "The user placed every item deliberately — do not 'improve' the layout."
    )

    return "\n".join(lines)


def build_floor_plan_prompt(
    payload: dict,
    image_order: Optional[List[str]] = None,
    has_guide: bool = False,
    view: str = "isometric",
    iso_base: bool = False,
    correction_notes: str = "",
) -> str:
    """
    Strict separation of concerns:
      JSON data  → all geometry (positions, dimensions, rotations, layout)
      English    → appearance only (materials, lighting, mood, render quality)

    image_order: product IDs in the order their photos are attached (IMAGE 2, 3, …).
    has_guide:   True when IMAGE 1 is a hand-drawn colour-blob guide (Surface Editor).
    view:        "isometric" | "front" | "corner"
    iso_base:    True for front/corner — IMAGE 1 is the isometric render, not a blank canvas.
    """
    if iso_base and view in ("front", "corner"):
        return _build_derived_view_prompt(payload, image_order=image_order, view=view)

    room = payload.get("room", {})
    fl = room.get("flooring", {})
    walls_cfg = room.get("walls", {})
    lighting = payload.get("lighting", {})

    # Detect polygon room
    room_polygon = room.get("polygon")
    is_polygon_room = (
        room_polygon
        and isinstance(room_polygon, list)
        and len(room_polygon) >= 3
    )
    n_walls = len(room_polygon) if is_polygon_room else 4

    # ── English appearance paragraph — NO geometry, NO coordinates ───────────
    openings = payload.get("openings", [])
    has_openings = len(openings) > 0
    has_windows = any((o.get("type") or "").lower() == "window" for o in openings)

    app: List[str] = ["Ultra-realistic architectural visualization"]

    lt = (lighting.get("type") or "natural_daylight").replace("_", " ")
    if "natural" in lt.lower() or "daylight" in lt.lower():
        if has_windows:
            app.append("warm natural daylight, soft ambient light from windows")
        else:
            app.append("warm soft ambient light, evenly lit interior")
    elif lt.strip():
        app.append(lt)

    floor_bits = [fl.get("material", ""), fl.get("type", "")]
    floor_str = " ".join(b for b in floor_bits if b).strip()
    if floor_str:
        d = fl.get("plank_direction") or fl.get("direction", "")
        app.append(f"{floor_str} flooring" + (f", {d} planks" if d else ""))

    wall_color = walls_cfg.get("color", "")
    if wall_color:
        app.append(f"walls painted {wall_color}")

    for p in payload.get("products", []):
        mat = p.get("material")
        if isinstance(mat, dict):
            app.extend(str(v) for v in mat.values() if v)
        elif isinstance(mat, str) and mat:
            app.append(mat)

    decor = payload.get("decor_style") or ""
    if decor:
        app.append(f"{decor.replace('_', ' ')} interior")

    app += [
        "soft drop-shadows beneath each furniture piece",
        "no sticker or cutout look",
        "clean architectural rendering — do NOT embellish or add decorative elements not in the reference photos",
    ]

    appearance = ", ".join(app) + "."

    # ── Camera / view ─────────────────────────────────────────────────────────
    _VIEW_CAMERAS = {
        "isometric": {"name": "isometric_top_view", "fov": 35},
        "front":     {"name": "front_view",          "fov": 50},
        "corner":    {"name": "corner_view",          "fov": 65},
    }
    _VIEW_LINES = {
        "isometric": (
            "Top-down 3D isometric perspective, full room footprint in frame, no tight crop."
        ),
        "front": (
            "Eye-level interior perspective — camera at 1.2 m height positioned outside "
            "the SOUTH wall looking straight NORTH, full room width in frame, "
            "all furniture fully visible, no cropping."
        ),
        "corner": (
            "Wide-angle 3D corner perspective — camera pulled far back and elevated, "
            "placed outside the SW corner looking diagonally toward the NE corner. "
            "The entire room floor plan must be visible. "
            "Show all four walls and every furniture piece without cropping. "
            "Do not zoom in — keep the whole room in frame."
        ),
    }
    view_line = _VIEW_LINES.get(view, _VIEW_LINES["isometric"])

    # ── Product image references ───────────────────────────────────────────────
    img_order = image_order or []
    products_by_id = {p["id"]: p for p in payload.get("products", [])}
    ref_lines: List[str] = []

    if has_guide and img_order:
        ref_lines.append(
            "IMAGE 1 is a precise 2D floor plan guide showing the room at exact scale:"
        )
        ref_lines.append(
            "- Dark grey border = room walls"
        )
        if has_openings:
            ref_lines.append(
                "- Tan gaps in walls = doors; light blue gaps = windows"
            )
        else:
            ref_lines.append(
                f"- All {n_walls} walls are SOLID with NO gaps — there are NO doors or windows in this room"
            )
        ref_lines.append(
            "- Neutral gray rectangles = exact product footprint positions (position, size, rotation all precise)"
        )
        ref_lines.append(
            "- Each rectangle has a CIRCLED NUMBER in its top-left corner identifying which product goes there:"
        )
        for i, pid in enumerate(img_order, start=1):
            p_info = products_by_id.get(pid, {})
            dims = p_info.get("dimensions", {})
            dw = dims.get("width") or 0
            dd = dims.get("depth") or 0
            dim_str = f", {float(dw):.1f}\u00d7{float(dd):.1f}m" if dw and dd else ""
            ref_lines.append(f"  Rectangle \u24ea{i} = \"{pid}\"{dim_str} \u2192 render using IMAGE {i + 1}")
        ref_lines.append(
            "- Beige background = empty floor; white outside = outside the room"
        )
        ref_lines.append(
            "- Thick black edge on a rectangle = the FRONT of that product"
        )
        ref_lines.append(
            "- The gray color of the rectangles is MEANINGLESS — it is NOT a color hint for the product. "
            "Get each product's color/material ONLY from its reference photo (IMAGE 2, 3, …)."
        )
        ref_lines.append("")
        ref_lines.append(
            "CRITICAL GUIDE RULES:\n"
            "1. Each product's rendered footprint MUST align with its numbered guide rectangle \u2014 same position, same size, same rotation.\n"
            "2. If a rectangle is in the left third of the room in the guide, the product MUST be in the left third in the render.\n"
            "3. Do NOT shift products to look more 'balanced' or 'centered' \u2014 the guide positions are the user's explicit intent.\n"
            "4. Replace gray rectangles with photorealistic furniture. Show bare floor everywhere else.\n"
            "5. If the same product appears multiple times (e.g. two armchairs), render ALL instances with IDENTICAL appearance — same color, same material, same design. Only position/rotation differs."
        )
    elif img_order:
        ref_lines.append("Product reference photos — render each item to exactly match its image:")
        for i, pid in enumerate(img_order, start=1):
            ref_lines.append(f"  IMAGE {i + 1} = product id={pid}")

    refs = "\n".join(ref_lines)

    # ── Product identity fidelity block ───────────────────────────────────────
    identity_lines: List[str] = []
    if img_order:
        identity_lines.append(
            "PRODUCT IDENTITY FIDELITY — EQUALLY IMPORTANT AS PLACEMENT ACCURACY:"
        )
        identity_lines.append(
            "IMPORTANT: Reference photos may contain OTHER objects (chairs at desks, pillows on beds, "
            "rugs under tables, plants, accessories). These are staging props — IGNORE THEM. "
            "From each reference photo, extract and render ONLY the single named product below."
        )
        for i, pid in enumerate(img_order, start=1):
            identity_lines.append(
                f"  IMAGE {i + 1} → render ONLY the \"{pid}\" — copy its exact design, shape, color, "
                f"material, frame, and structure. IGNORE any other furniture, chairs, accessories, "
                f"or objects visible in the same photo — they are staging props, not products to render."
            )
        # Detect duplicates — same base product used multiple times
        base_names = {}
        for pid in img_order:
            base = pid.rstrip("_0123456789")  # "Armchair_2" → "Armchair"
            base_names.setdefault(base, []).append(pid)
        duplicate_lines = []
        for base, pids in base_names.items():
            if len(pids) > 1:
                duplicate_lines.append(
                    f"- {', '.join(pids)} are the SAME product — render ALL of them with IDENTICAL "
                    f"appearance (same color, same fabric, same design). Only their position/rotation differs."
                )

        identity_lines.append(
            "\nIDENTITY RULES:\n"
            "- From each reference photo, render ONLY the single named product. All other objects in the photo are staging — discard them.\n"
            "- If a desk photo shows a chair, render ONLY the desk. If a bed photo shows side tables, render ONLY the bed.\n"
            "- Do NOT change the fabric color, pattern, or material of the named product.\n"
            "- Do NOT redesign the product shape, frame, or structure — copy the named product faithfully.\n"
            "- Do NOT add extra hardware (handles, locks, knobs) to doors or furniture beyond what the reference shows.\n"
            "- The reference photo defines ONLY the appearance of the named product — nothing else from the photo should appear in the render."
        )
        if duplicate_lines:
            identity_lines.append("\nDUPLICATE PRODUCTS — must look identical:")
            identity_lines.extend(duplicate_lines)
    identity_block = "\n".join(identity_lines) if identity_lines else ""

    # ── Geometry block — raw JSON with percentage anchors added ──────────────
    room_w = float(room.get("width") or 0)
    room_l = float(room.get("length") or 0)

    _FACING = {0: "NORTH", 90: "EAST", 180: "SOUTH", 270: "WEST"}

    clean_products = []
    for p in payload.get("products", []):
        cp = {k: v for k, v in p.items() if k not in ("material", "image_url", "hex_color")}
        # Enrich position with x_pct / y_pct so model can cross-check metre vs percentage
        pos = cp.get("position", {})
        if pos and room_w and room_l:
            x, y = float(pos.get("x") or 0), float(pos.get("y") or 0)
            cp["position"] = {
                **pos,
                "x_pct": round(x / room_w * 100, 1),
                "y_pct": round(y / room_l * 100, 1),
            }
        # Translate rotation_y to a compass facing so the model understands orientation
        rot = int(cp.get("rotation_y") or 0) % 360
        cp["facing"] = _FACING.get(rot, f"{rot}deg_CW")
        cp["facing_note"] = (
            f"The FRONT of this product faces {cp['facing']} — "
            f"rotate it {rot}° clockwise from its reference photo orientation."
        )
        clean_products.append(cp)

    # ── Per-product rotation summary (English, before JSON) ──────────────────
    rotation_lines = []
    for p in payload.get("products", []):
        rot = int(p.get("rotation_y") or 0) % 360
        pid = p.get("id", "?")
        cat = (p.get("category") or pid).replace("_", " ")
        facing = _FACING.get(rot, f"{rot}°")
        dims = p.get("dimensions", {})
        w = float(dims.get("width") or 0)
        d = float(dims.get("depth") or 0)
        is_square = abs(w - d) < 0.1
        # Compute effective footprint after rotation
        if rot in (90, 270):
            eff_w, eff_d = d, w  # width/depth swap
        else:
            eff_w, eff_d = w, d
        footprint = f"footprint in room: {eff_w:.1f}m LEFT↔RIGHT × {eff_d:.1f}m TOP↔BOTTOM"

        if rot == 0:
            rotation_lines.append(
                f"  {pid}: 0° — use reference photo orientation as-is. Front faces NORTH. {footprint}."
            )
        elif rot == 90:
            if is_square:
                rotation_lines.append(
                    f"  {pid}: 90° CW — front faces EAST. "
                    f"Product is square ({w:.1f}×{d:.1f}m), rotate it so its front/seat side points RIGHT. {footprint}."
                )
            else:
                long_dir = "LEFT↔RIGHT" if eff_w > eff_d else "TOP↔BOTTOM"
                rotation_lines.append(
                    f"  {pid}: 90° CW — front faces EAST. "
                    f"Longer dimension ({max(eff_w,eff_d):.1f}m) runs {long_dir}. {footprint}."
                )
        elif rot == 180:
            rotation_lines.append(
                f"  {pid}: 180° — completely flipped from reference photo. Front faces SOUTH. {footprint}."
            )
        elif rot == 270:
            if is_square:
                rotation_lines.append(
                    f"  {pid}: 270° CW — front faces WEST. "
                    f"Product is square ({w:.1f}×{d:.1f}m), rotate it so its front/seat side points LEFT. {footprint}."
                )
            else:
                long_dir = "LEFT↔RIGHT" if eff_w > eff_d else "TOP↔BOTTOM"
                rotation_lines.append(
                    f"  {pid}: 270° CW — front faces WEST. "
                    f"Longer dimension ({max(eff_w,eff_d):.1f}m) runs {long_dir}. {footprint}."
                )
        else:
            rotation_lines.append(
                f"  {pid}: {rot}° CW from reference photo. Front faces {facing}. {footprint}."
            )

    rotation_block = (
        "PRODUCT ROTATIONS — apply exactly before placing in the room:\n"
        + "\n".join(rotation_lines)
    ) if rotation_lines else ""

    geometry = {
        "unit": payload.get("unit", "m"),
        "room": payload.get("room", {}),
        "openings": payload.get("openings", []),
        "products": clean_products,
        "camera_views": [_VIEW_CAMERAS.get(view, _VIEW_CAMERAS["isometric"])],
    }
    geometry_block = (
        "Room geometry — positions, dimensions, and rotations are exact; render precisely to these values.\n"
        "Coordinate system: origin (0,0) = NW corner; x increases EAST; y increases SOUTH.\n"
        "Facing: NORTH = toward top of room (y=0 wall); SOUTH = toward bottom (y=max wall); "
        "EAST = toward right (x=max wall); WEST = toward left (x=0 wall).\n"
        + json.dumps(geometry, ensure_ascii=False, indent=2)
    )

    # ── Spatial placement spec (wall gaps, asymmetry directives) ─────────────
    spatial_spec = _compute_spatial_spec(payload)

    # ── Openings spec (doors & windows) ────────────────────────────────────────
    openings = payload.get("openings", [])
    openings_spec = ""
    if openings:
        o_parts = []
        for o in openings:
            otype = (o.get("type") or "opening").lower()
            wall = (o.get("wall") or "?").lower()
            ow = float(o.get("width") or 0)
            o_parts.append(f"{otype} on {wall} wall ({ow:.1f}m wide)")
        openings_spec = (
            f"DOORS & WINDOWS: Render these {len(openings)} opening(s): "
            + ", ".join(o_parts) + ". "
            "In IMAGE 1, tan gaps = doors, light blue gaps = windows. "
            "Render each as a simple, plain architectural element — a basic door frame with a flat panel, "
            "or a plain window with simple glass and frame. Do NOT add decorative details, extra locks, "
            "handles, panels, or hardware beyond a single standard handle. Keep doors and windows minimal and clean."
        )

    # ── Hard constraints ───────────────────────────────────────────────────────
    n = len(payload.get("products", []))
    n_openings = len(openings)
    constraints = (
        f"HARD CONSTRAINTS — every rule is mandatory, violation = failure:\n"
        f"(1) Exactly {n} furniture item(s) in total — no more, no fewer. Do NOT add chairs, stools, rugs, lamps, plants, "
        f"cushions, throws, blankets, or any object from a reference photo's background/staging.\n"
    )
    if n_openings:
        constraints += (
            f"(2) Only render the {n_openings} architectural opening(s) listed in the JSON openings array — no extra doors, windows, or skylights.\n"
        )
    else:
        constraints += (
            f"(2) This room has ZERO openings — do NOT render any doors, windows, or skylights. "
            f"All {n_walls} walls must be completely SOLID with no gaps, no glass, no frames, no openings of any kind.\n"
        )
    constraints += (
        f"(3) PRODUCT IDENTITY: Copy ONLY the named product from each reference photo. Ignore staging props visible in the photo. "
        f"Match the named product's design, colors, and structure exactly — do NOT redesign or restyle.\n"
        f"(4) No text, labels, or watermarks.\n"
        f"(5) The ONLY objects in the render are: the {n} named products + the room itself (walls, floor). Nothing else."
    )
    if n_openings:
        constraints += f"\n(6) Render all {n_openings} opening(s) as simple, plain doors/windows in the correct walls — no extra hardware or decorative details."

    # Room shape description for polygon rooms
    room_shape_block = ""
    if is_polygon_room:
        room_w_val = float(room.get("width") or 0)
        room_l_val = float(room.get("length") or 0)
        room_shape_block = (
            f"ROOM SHAPE: This room is an irregular polygon with {n_walls} vertices — it is NOT a simple rectangle.\n"
            f"The polygon vertices (in metres, origin at bounding-box top-left) are:\n"
            f"  {json.dumps(room_polygon)}\n"
            f"Bounding box: {room_w_val:.1f}m wide x {room_l_val:.1f}m deep.\n"
            f"The room outline must follow this polygon shape exactly. Do NOT render a rectangular room.\n"
            f"The floor plan guide (IMAGE 1) shows the polygon outline — match it precisely."
        )

    return "\n\n".join(filter(None, [appearance, view_line, room_shape_block, rotation_block, refs, identity_block, geometry_block, spatial_spec, correction_notes, openings_spec, constraints])).strip() + (
        "\n\nOutput: one photorealistic render, no overlays, no on-image text."
    )


def build_wall_plan_prompt(
    payload: dict,
    image_order: Optional[List[str]] = None,
    has_guide: bool = False,
    correction_notes: str = "",
) -> str:
    """
    Prompt for wall elevation renders.
    Strict separation of concerns (mirrors build_floor_plan_prompt):
      JSON data  → all geometry (positions, dimensions, openings)
      English    → appearance only (materials, lighting, render quality)

    image_order: product IDs in the order their photos are attached (IMAGE 2, 3, …).
    has_guide:   True when IMAGE 1 is a programmatically-drawn wall elevation guide.
    """
    wall = payload.get("wall", {})
    lighting = payload.get("lighting", {})

    # ── English appearance — NO geometry ─────────────────────────────────────
    app: List[str] = ["Ultra-realistic architectural visualization"]

    lt = (lighting.get("type") or "natural_daylight").replace("_", " ")
    if "natural" in lt.lower() or "daylight" in lt.lower():
        app.append("warm natural daylight, soft ambient light from windows")
    elif lt.strip():
        app.append(lt)

    wall_color = wall.get("color", "")
    if wall_color:
        app.append(f"wall painted {wall_color}")

    for p in payload.get("products", []):
        mat = p.get("material")
        if isinstance(mat, dict):
            app.extend(str(v) for v in mat.values() if v)
        elif isinstance(mat, str) and mat:
            app.append(mat)

    app += [
        "soft drop-shadows beneath each piece",
        "realistic material textures (fabric, wood, metal)",
        "no sticker or cutout look",
        "editorial photography quality",
    ]
    appearance = ", ".join(app) + "."

    # ── View ─────────────────────────────────────────────────────────────────
    view_line = (
        "Front-facing wall elevation — camera at eye level (1.2 m height) perpendicular "
        "to the wall, looking straight at it. Full wall width and height in frame. "
        "Show a strip of floor in front. "
        "Floor-standing furniture rests on the floor in front of the wall; "
        "wall-mounted items (art, TV, shelves) are mounted on the wall surface."
    )

    # ── Per-product rotation summary ─────────────────────────────────────────
    rotation_lines: List[str] = []
    for p in payload.get("products", []):
        rot = int(p.get("rotation_y") or 0) % 360
        if rot == 0:
            continue
        pid = p.get("id", "?")
        dims = p.get("dimensions", {})
        w = float(dims.get("width") or 0)
        h = float(dims.get("height") or 0)
        eff_w = h if rot in (90, 270) else w
        eff_h = w if rot in (90, 270) else h
        rotation_lines.append(
            f"  {pid}: {rot}° CW — product is rotated {rot}° clockwise as viewed from the front. "
            f"Visible footprint on wall: {eff_w:.1f}m wide × {eff_h:.1f}m tall."
        )
    rotation_block = (
        "PRODUCT ROTATIONS — apply exactly before placing on the wall:\n"
        + "\n".join(rotation_lines)
    ) if rotation_lines else ""

    # ── Image references ──────────────────────────────────────────────────────
    img_order = image_order or []
    products_by_id = {p["id"]: p for p in payload.get("products", [])}
    ref_lines: List[str] = []

    if has_guide and img_order:
        ref_lines.append("IMAGE 1 is a precise 2D wall elevation guide drawn at exact physical scale:")
        ref_lines.append("- Dark grey border = wall boundary (exact width × height)")
        ref_lines.append("- Tan rectangles at bottom edge = doors; light blue rectangles = windows")
        ref_lines.append("- GRAY rectangles = exact product footprint positions (different gray shades per product)")
        ref_lines.append("- Each gray rectangle has a CIRCLED NUMBER in its top-left corner identifying which product goes there")
        ref_lines.append("- Thin vertical DASHED LINE through each rectangle = the exact horizontal center of that product")
        ref_lines.append("- Purple label below each dashed line (e.g. 'x=1.65m 30%') = horizontal center distance from LEFT wall edge")
        ref_lines.append("")
        ref_lines.append("PRODUCT PLACEMENT — exact x-position per item (these are binding, not suggestions):")
        wall_w2 = float(wall.get("width") or 1)
        wall_h2 = float(wall.get("height") or 4.0)
        for i, pid in enumerate(img_order, start=1):
            p_info = products_by_id.get(pid, {})
            dims = p_info.get("dimensions", {})
            pos = p_info.get("position", {})
            dw = float(dims.get("width") or 0)
            dh = float(dims.get("height") or dims.get("depth") or 0)
            px = float(pos.get("x") or 0)
            py = float(pos.get("y") or 0)
            x_pct = round(px / wall_w2 * 100, 0) if wall_w2 else 0
            y_pct = round(py / wall_h2 * 100, 0) if wall_h2 else 0
            side = "LEFT third" if x_pct < 34 else ("CENTER" if x_pct < 67 else "RIGHT third")
            pos_str = (
                f"center at x={px:.2f}m from LEFT edge ({x_pct:.0f}% — {side} of wall), "
                f"y={py:.2f}m from floor ({y_pct:.0f}% up the wall)"
            ) if px or py else ""
            dim_str = f"{dw:.1f}×{dh:.1f}m" if dw and dh else ""
            ref_lines.append(
                f"  ⓪{i} \"{pid}\" [{dim_str}] — {pos_str} — render using IMAGE {i + 1}"
            )
        ref_lines.append("")
        ref_lines.append(
            "CRITICAL POSITIONING RULES — violation = failure:\n"
            "1. Each product's LEFT-RIGHT position MUST match the dashed center line in the guide and the x-value listed above.\n"
            "2. Do NOT cluster products near windows or doors — place them at the exact x-position shown, even if that leaves them isolated.\n"
            "3. Do NOT move products to create a 'balanced' composition — the user's explicit placement is the only valid arrangement.\n"
            "4. Replace gray guide rectangles with photorealistic products. Show bare wall everywhere else.\n"
            "5. NO FRAME — The thin outline around the guide image is a diagram boundary marker ONLY. "
            "Do NOT render any dark frame, border, molding, trim, or outline around the wall edges in the output. "
            "The wall surface must extend cleanly to the edges of the frame with NO visible border — "
            "just smooth plaster/paint all the way to the image boundary."
        )
    elif img_order:
        ref_lines.append("Product reference photos — render each to exactly match its image:")
        for i, pid in enumerate(img_order, start=1):
            ref_lines.append(f"  IMAGE {i + 1} = product id={pid}")

    refs = "\n".join(ref_lines)

    # ── Product identity block ────────────────────────────────────────────────
    identity_lines: List[str] = []
    if img_order:
        identity_lines.append("PRODUCT IDENTITY FIDELITY:")
        identity_lines.append(
            "Reference photos may contain OTHER objects (chairs, accessories, staging props). "
            "IGNORE THEM. From each photo, extract and render ONLY the single named product."
        )
        for i, pid in enumerate(img_order, start=1):
            identity_lines.append(
                f"  IMAGE {i + 1} → render ONLY the \"{pid}\" — copy its exact design, shape, color, "
                f"material, frame, and structure. IGNORE all other objects in the photo."
            )
        identity_lines.append(
            "\nIDENTITY RULES:\n"
            "- Do NOT change the fabric color, pattern, or material of the named product.\n"
            "- Do NOT redesign the product shape, frame, or structure — copy faithfully.\n"
            "- The reference photo defines ONLY the appearance of the named product."
        )
    identity_block = "\n".join(identity_lines) if identity_lines else ""

    # ── Geometry block (raw JSON + percentage cross-checks) ───────────────────
    wall_w = float(wall.get("width") or 0)
    wall_h = float(wall.get("height") or 4.0)

    clean_products = []
    for p in payload.get("products", []):
        cp = {k: v for k, v in p.items() if k not in ("material", "image_url", "hex_color")}
        pos = cp.get("position", {})
        if pos and wall_w and wall_h:
            x = float(pos.get("x") or 0)
            y = float(pos.get("y") or 0)  # y = centre height from floor
            cp["position"] = {
                **pos,
                "x_pct": round(x / wall_w * 100, 1),
                "y_from_floor_pct": round(y / wall_h * 100, 1),
            }
        clean_products.append(cp)

    geometry = {
        "unit": payload.get("unit", "m"),
        "wall": wall,
        "openings": payload.get("openings", []),
        "products": clean_products,
    }
    geometry_block = (
        "Wall geometry — positions and dimensions are exact; render precisely.\n"
        "Coordinate system: origin (0,0) = bottom-left corner of wall (floor level, left edge). "
        "x increases rightward (0 = left edge, wall.width = right edge). "
        "y increases upward (0 = floor, wall.height = ceiling). "
        "position.x/y = centre of the product footprint.\n"
        + json.dumps(geometry, ensure_ascii=False, indent=2)
    )

    # ── Openings spec ─────────────────────────────────────────────────────────
    openings = payload.get("openings", [])
    openings_spec = ""
    if openings:
        o_parts = []
        for o in openings:
            otype = (o.get("type") or "opening").lower()
            ow = float(o.get("width") or 0)
            oh = float(o.get("height") or (2.2 if otype == "door" else 0.9))
            pos_left = float(o.get("position_from_left") or 0)
            if otype == "door":
                o_parts.append(
                    f"door at x={pos_left:.2f}m from left edge, {ow:.1f}m wide × {oh:.1f}m tall, at floor level"
                )
            else:
                sill = float(o.get("sill_height") or 1.2)
                o_parts.append(
                    f"window at x={pos_left:.2f}m from left edge, {ow:.1f}m wide × {oh:.1f}m tall, sill at {sill:.1f}m"
                )
        openings_spec = (
            f"DOORS & WINDOWS: Render these {len(openings)} opening(s) on this wall: "
            + "; ".join(o_parts) + ". "
            "Render each as a simple, clean architectural element — "
            "a plain door frame with flat panel and single handle, "
            "or a plain window with glass and simple frame. No extra decorative hardware."
        )

    # ── Hard constraints ──────────────────────────────────────────────────────
    n = len(payload.get("products", []))
    n_openings = len(openings)
    constraints = (
        f"HARD CONSTRAINTS — every rule is mandatory, violation = failure:\n"
        f"(1) Exactly {n} product item(s) — no extra accessories, rugs, lamps, or decor not listed.\n"
        f"(2) PRODUCT IDENTITY: copy ONLY the named product from each reference photo; "
        f"ignore all staging props visible in the same photo.\n"
        f"(3) No text, labels, dimension lines, or watermarks.\n"
        f"(4) Wall boundaries must match the guide exactly — no extra openings or structural features.\n"
        f"(5) Front-facing elevation view only — no top-down, no fisheye, no angled perspective.\n"
        f"(6) NO WALL FRAME — do NOT render any dark border, molding, trim, or outline around the wall "
        f"edges. The wall is a flat interior surface that extends to the image boundary; "
        f"there is no physical frame around it."
    )
    if n_openings:
        constraints += (
            f"\n(7) Render all {n_openings} opening(s) as simple, plain doors/windows — "
            "no extra hardware or decorative panels beyond what the guide shows."
        )

    return "\n\n".join(filter(None, [
        appearance, view_line, rotation_block, refs, identity_block,
        geometry_block, correction_notes, openings_spec, constraints,
    ])).strip() + "\n\nOutput: one photorealistic front-facing wall elevation render, no overlays, no on-image text."


_CORNER_CFG = {
    "sw": {
        "camera_desc": "SOUTH-WEST corner, looking diagonally toward the NORTH-EAST interior",
        "visible_walls": {
            "north": "the BACK wall — runs horizontally left-to-right at the far end of the room",
            "east":  "the RIGHT-SIDE wall — runs diagonally from the back-right corner toward the front-right",
        },
        "hidden_walls": ["south", "west"],
    },
    "se": {
        "camera_desc": "SOUTH-EAST corner, looking diagonally toward the NORTH-WEST interior",
        "visible_walls": {
            "north": "the BACK wall — runs horizontally left-to-right at the far end of the room",
            "west":  "the LEFT-SIDE wall — runs diagonally from the back-left corner toward the front-left",
        },
        "hidden_walls": ["south", "east"],
    },
    "ne": {
        "camera_desc": "NORTH-EAST corner, looking diagonally toward the SOUTH-WEST interior",
        "visible_walls": {
            "south": "the BACK wall — runs horizontally left-to-right at the far end of the room from this angle",
            "west":  "the LEFT-SIDE wall — runs diagonally from the back-left corner toward the front-left",
        },
        "hidden_walls": ["north", "east"],
    },
    "nw": {
        "camera_desc": "NORTH-WEST corner, looking diagonally toward the SOUTH-EAST interior",
        "visible_walls": {
            "south": "the BACK wall — runs horizontally left-to-right at the far end of the room from this angle",
            "east":  "the RIGHT-SIDE wall — runs diagonally from the back-right corner toward the front-right",
        },
        "hidden_walls": ["north", "west"],
    },
}


def build_composition_prompt(
    room_dimensions: Optional[Dict] = None,
    wall_labels: Optional[List[str]] = None,
    presets: Optional[Dict] = None,
    n_walls: int = 0,
    corner: str = "sw",
) -> str:
    """Prompt for one corner view of the final isometric room composition.

    corner:      "sw" (camera at SW, shows NORTH + EAST walls)
                 "ne" (camera at NE, shows SOUTH + WEST walls)
    wall_labels: compass labels for the walls provided in this corner (IMAGE 2, 3, …).
    """
    rd = room_dimensions or {}
    w = float(rd.get("width") or 0)
    d = float(rd.get("depth") or rd.get("length") or 0)
    wall_h = float(rd.get("wall_height") or 4.0)
    unit = rd.get("unit", "m")

    room_desc = (
        f"{w:.1f} × {d:.1f} × {wall_h:.1f} {unit} (width × depth × height)"
        if w and d else "see IMAGE 1 for proportions"
    )

    pr = presets or {}
    wall_color = pr.get("wall_color", "#F3EFE8")
    style_parts = [s for s in [pr.get("room_type"), pr.get("decor_style")] if s]
    style_line = ("Style: " + ", ".join(style_parts) + ".") if style_parts else ""

    cfg = _CORNER_CFG[corner]
    camera_desc = cfg["camera_desc"]
    visible_map = cfg["visible_walls"]        # compass → visual description
    hidden_walls = cfg["hidden_walls"]        # walls NOT visible from this corner

    provided_labels = [l.lower().strip() for l in (wall_labels or [])]

    # ── Per-wall sections for walls that have elevation images ─────────────────
    wall_sections: List[str] = []
    for i, label in enumerate(provided_labels):
        visual = visible_map.get(label, f"the {label.upper()} wall")
        wall_sections.append(
            f"IMAGE {i + 2} → {label.upper()} wall ({visual})\n"
            f"  Reproduce this elevation EXACTLY onto that wall surface (foreshortened to the corner angle):\n"
            f"  • Include EVERY item shown in the elevation — especially small wall-mounted ones\n"
            f"    such as framed art / canvases / posters, mirrors, shelves, clocks, and wall sconces.\n"
            f"    Do NOT omit, shrink away, or skip any item, no matter how small.\n"
            f"  • Keep each item's left-right position, height from floor, size, and exact appearance\n"
            f"  • Same openings (door/window) at the same positions\n"
            f"  • Same wall colour/finish\n"
            f"  • Do NOT move any item to a different wall, and do NOT reposition openings to 'fit' the view"
        )

    # ── Visible walls with NO elevation → plain surface ────────────────────────
    no_elevation_visible = [
        w for w in visible_map if w not in provided_labels
    ]
    plain_visible_text = ""
    if no_elevation_visible:
        plain_list = ", ".join(
            f"{w.upper()} ({visible_map[w]})" for w in no_elevation_visible
        )
        plain_visible_text = (
            f"## Visible walls with NO elevation provided\n"
            f"{plain_list}\n"
            f"Render as plain {wall_color} painted surfaces — NO products, NO openings. "
            f"Do NOT invent anything on these walls."
        )

    # ── Walls not visible from this corner ─────────────────────────────────────
    hidden_text = (
        f"## Walls NOT visible from this corner\n"
        f"{', '.join(w.upper() for w in hidden_walls)}\n"
        f"These walls are at the camera position. Do NOT render them."
    )

    wall_sections_block = "\n\n".join(wall_sections)

    return f"""## Task
Produce ONE photorealistic isometric room render from the {camera_desc}.

Use IMAGE 1 for the floor layout. Use IMAGE 2+ for the wall surfaces.
Render naturally from this corner angle — partial wall views at the edges are acceptable.
{style_line}

## Room dimensions
{room_desc}

## Camera position
Camera at the {camera_desc}.
Match the isometric angle, height, and zoom from IMAGE 1.

## Walls visible from this corner
{chr(10).join(f"- {w.upper()} wall: {desc}" for w, desc in visible_map.items())}

## IMAGE 1 — floor + base room reference
IMAGE 1 is the authoritative base. Preserve it faithfully:
- Positions, sizes, and orientations of all floor-standing furniture — keep them unchanged
- Floor material, room proportions, camera angle, and lighting
- Any openings (windows/doors) already shown in IMAGE 1 — keep them where they are; do NOT move them
The walls in IMAGE 1 are bare painted surfaces. Add ONLY the wall-mounted content
from the elevation images below onto the corresponding walls; change nothing else.

## Wall elevation images

{wall_sections_block if wall_sections_block else "(No wall elevations for this corner.)"}

{plain_visible_text}

{hidden_text}

## Rendering approach
1. Use IMAGE 1's isometric camera angle.
2. Reproduce the floor furniture from IMAGE 1 at their exact positions.
3. For each visible wall that has an elevation image: apply those wall products and openings to that wall surface. Products that appear at the edge of the view may be partially cropped — that is fine.
4. For visible walls without an elevation: render as plain {wall_color} painted surface.
5. Wall products stay on their own wall; floor products stay on the floor.
6. Soft ambient lighting consistent with IMAGE 1.

## Constraints
- Do not move floor furniture or openings from IMAGE 1's positions
- Every item visible in a wall's elevation MUST appear on that wall — never drop wall art, frames, mirrors, or sconces
- Do not add products or openings to walls without elevation images
- Do not invent any furniture, accessories, or architectural elements
- No text, labels, or watermarks

Output: a single photorealistic isometric room render, no overlays, no on-image text.
""".strip()


# Two opposite open-box (dollhouse) CORNER views. Each shows exactly the TWO far walls
# meeting at the back corner, with the two near walls removed so you can see in. The two
# views are opposite corners (180° apart) and together reveal all four walls with no wall
# shown twice:
#   - "front": camera at the SE corner looking toward the NW corner → shows WEST (left) +
#     NORTH (right); the near SOUTH and EAST walls are removed.
#   - "back":  camera at the NW corner looking toward the SE corner → shows EAST (left) +
#     SOUTH (right); the near NORTH and WEST walls are removed.
# A product whose wall is removed in a given view is simply NOT shown there — it appears in
# the other view. Products are never relocated onto a visible wall to "fit" them in.
_DOLLHOUSE_VIEWS = {
    "front": {
        "open_walls": ["south", "east"],
        "camera": "a slightly elevated three-quarter isometric view looking toward the back NORTH-WEST corner, with the two near walls (SOUTH and EAST) removed so you can see into the room (open dollhouse)",
        "visible": {
            "west":  "the LEFT-hand wall of the open box (the WEST wall) — it meets the other wall at the central back corner and recedes forward toward the open front-left",
            "north": "the RIGHT-hand wall of the open box (the NORTH wall) — it meets the other wall at the central back corner and recedes forward toward the open front-right",
        },
    },
    "back": {
        "open_walls": ["north", "west"],
        "camera": "a slightly elevated three-quarter isometric view from the OPPOSITE corner (rotated 180°), looking toward the back SOUTH-EAST corner, with the two near walls (NORTH and WEST) removed (open dollhouse)",
        "visible": {
            "east":  "the LEFT-hand wall of the open box (the EAST wall) — it meets the other wall at the central back corner and recedes forward toward the open front-left",
            "south": "the RIGHT-hand wall of the open box (the SOUTH wall) — it meets the other wall at the central back corner and recedes forward toward the open front-right",
        },
    },
}


def dollhouse_view_walls(view: str):
    """Return (open_walls, [visible compass walls in IMAGE 2+ / left-to-right order])."""
    cfg = _DOLLHOUSE_VIEWS[view]
    return list(cfg["open_walls"]), list(cfg["visible"].keys())


def _room_desc(room_dimensions: Optional[Dict]) -> str:
    rd = room_dimensions or {}
    w = float(rd.get("width") or 0)
    d = float(rd.get("depth") or rd.get("length") or 0)
    wall_h = float(rd.get("wall_height") or 4.0)
    unit = rd.get("unit", "m")
    return (
        f"{w:.1f} × {d:.1f} × {wall_h:.1f} {unit} (width × depth × height)"
        if w and d else "see IMAGE 1 for proportions"
    )


def _style_line(presets: Optional[Dict]) -> str:
    pr = presets or {}
    style_parts = [s for s in [pr.get("room_type"), pr.get("decor_style")] if s]
    return ("Style: " + ", ".join(style_parts) + ".") if style_parts else ""


def _describe_wall_openings(compass: str, ops: Optional[List[Dict]]) -> str:
    """Human-readable authoritative opening spec for one wall, e.g.
    'WEST wall: SOLID — no windows or doors.' or
    'NORTH wall: 1 window (centre-left).'"""
    C = compass.upper()
    if not ops:
        return f"- {C} wall: SOLID painted wall — NO windows and NO doors."
    parts = []
    for o in ops:
        otype = (o.get("type") or "opening").lower()
        pos = o.get("position_from_left")
        try:
            frac = float(pos)
            zone = "left" if frac < 0.34 else ("right" if frac > 0.66 else "centre")
        except (TypeError, ValueError):
            zone = "as in the floor plan"
        parts.append(f"a {otype} ({zone})")
    return f"- {C} wall: {len(ops)} opening(s) — " + ", ".join(parts) + "."


def build_dollhouse_shell_prompt(
    view: str,
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
    openings_by_wall: Optional[Dict[str, List[Dict]]] = None,
) -> str:
    """Step 1 of sequential composition: turn the floor render (IMAGE 1) into an EMPTY
    open-box dollhouse — correct camera + furniture, but completely BLANK walls.

    Wall content (products) is added afterward, one wall at a time. Architectural openings
    (doors/windows) are placed here, driven by the AUTHORITATIVE openings_by_wall spec so
    the model cannot invent windows on walls that should be solid.
    """
    cfg = _DOLLHOUSE_VIEWS[view]
    open_walls = cfg["open_walls"]
    open_walls_text = " and ".join(w.upper() for w in open_walls)
    # The camera stands over the corner where the two REMOVED (near) walls meet, and looks
    # across to the opposite back corner. This is what makes the two views genuinely opposite
    # (~180° apart): front stands at the SOUTH-EAST corner, back at the NORTH-WEST corner.
    standpoint = "-".join(w.upper() for w in open_walls)
    far_corner = "-".join(c.upper() for c in cfg["visible"].keys())
    camera = cfg["camera"]
    visible = cfg["visible"]
    pr = presets or {}
    wall_color = pr.get("wall_color", "#F3EFE8")
    ob = openings_by_wall or {}

    visible_lines = "\n".join(f"- {c.upper()} wall → {desc}" for c, desc in visible.items())

    # Authoritative per-visible-wall opening spec (overrides whatever IMAGE 1 seems to show).
    opening_spec = "\n".join(
        _describe_wall_openings(c, ob.get(c)) for c in visible.keys()
    )
    has_any_opening = any(ob.get(c) for c in visible.keys())
    openings_guidance = (
        "Render each listed opening as a realistic architectural element (a real door "
        "panel/frame, a real glazed window with frame), at the stated wall and rough "
        "position."
        if has_any_opening else
        "None of the visible walls have any openings — render all of them as solid, "
        "unbroken painted walls."
    )

    return f"""## Task
Re-render IMAGE 1 (a top-down floor plan of a room) as an EMPTY open-box (dollhouse) room
from {camera}. The {open_walls_text} walls are removed (open side toward the camera) — only the
two far walls listed below are drawn.
{_style_line(presets)}

## Room dimensions
{_room_desc(room_dimensions)}

## Camera viewpoint — FIXED (this determines how the furniture is seen)
The camera stands over the {standpoint} corner of the room (where the removed {open_walls_text}
walls meet) and looks diagonally across to the opposite {far_corner} back corner. You are
therefore seeing the room — and every piece of furniture — from its {standpoint} side.
- This is a true 3D three-quarter view: render the furniture FORESHORTENED and oriented exactly
  as it would look from the {standpoint} corner. Do NOT draw it flat, front-on, or from a default
  angle.
- The companion view of this same room is shot from the EXACT OPPOSITE ({far_corner}) corner, so
  the identical furniture must appear ROTATED ~180° between the two views. A sofa whose front
  faces the camera from this corner shows its BACK from the opposite corner; an item on the left
  here is on the right there. Make this viewpoint unmistakably the {standpoint} corner so the two
  views never look like the same angle.

## Keep from IMAGE 1 (EXACTLY)
- Every floor item: same identity, position, size, and real-world facing/placement in the room —
  do not move, add, or restyle anything. Only the camera angle changes how each piece is seen
  (it is viewed from the {standpoint} corner, foreshortened accordingly).
- The floor material, rug, and room proportions
- Soft, realistic, consistent lighting

## Wall positions in this view
{visible_lines}

## Openings — AUTHORITATIVE (this list overrides anything IMAGE 1 appears to show)
{opening_spec}
{openings_guidance}

## Walls — openings only, otherwise bare
- Place ONLY the openings listed above, on exactly those walls. A wall marked SOLID must have
  NO window and NO door — do not invent any. This is the most common mistake: do not add
  windows to fill empty walls.
- Add NO decor or products: no framed art, canvases, posters, mirrors, shelves, clocks, or
  objects on the walls. Those are added in a later step. Apart from the listed openings, the
  walls must be bare {wall_color} painted surfaces.

## Constraints
- Do NOT add, remove, move, or duplicate any door or window beyond the authoritative list.
- Do NOT add, move, duplicate, or restyle any floor furniture.
- Do NOT draw the {open_walls_text} walls (they are the open/near sides).
- No text, labels, watermarks, or overlays.

Output: a single photorealistic open-box dollhouse render with bare walls.
""".strip()


def build_geometric_furnish_prompt(
    view: str,
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
    openings_by_wall: Optional[Dict[str, List[Dict]]] = None,
) -> str:
    """Furniture-fill step of the DETERMINISTIC (geometric) backend.

    IMAGE 1 is a flat-shaded GEOMETRY GUIDE: the exact open-box room (correct camera,
    correct two walls, correct floor diamond, openings marked) computed by us. IMAGE 2 is
    the top-down/iso floor render that holds the furniture.

    The model's ONLY job is to photo-realistically furnish IMAGE 1's floor using the items
    in IMAGE 2 — it must NOT change the room's geometry, camera, wall count, or proportions
    (those are locked). Wall decor and the final wall textures are applied afterwards by a
    deterministic warp, so the walls here stay bare.
    """
    cfg = _DOLLHOUSE_VIEWS[view]
    open_walls_text = " and ".join(w.upper() for w in cfg["open_walls"])
    standpoint = "-".join(w.upper() for w in cfg["open_walls"])
    pr = presets or {}
    wall_color = pr.get("wall_color", "#F3EFE8")

    return f"""## Task
You are given IMAGE 1, a flat-shaded 3D template of an open-box ("dollhouse") room, and
IMAGE 2, a plan render of the same room's floor that shows the furniture. Produce ONE
photorealistic render that has the EXACT room structure of IMAGE 1, furnished with the items
from IMAGE 2.
{_style_line(presets)}

## Room dimensions
{_room_desc(room_dimensions)}

## Lock the structure to IMAGE 1 (do not reinvent it)
IMAGE 1 already defines the camera and the architecture. Match it exactly:
- The SAME camera angle and the SAME open three-quarter isometric viewpoint as IMAGE 1.
- EXACTLY TWO walls (the back-left and back-right planes of IMAGE 1) meeting at the central
  back corner. Do NOT add a third wall, a ceiling, or a near wall. The {open_walls_text} sides
  stay open toward the camera, exactly as in IMAGE 1.
- The floor occupies the SAME diamond footprint, and the walls have the SAME shape, height, and
  proportions as IMAGE 1. Keep every wall as a smooth, unbroken {wall_color} painted surface,
  except keep the door/window openings exactly where IMAGE 1 shows them.

## Furnish the floor from IMAGE 2
- Place every piece of furniture from IMAGE 2 onto the floor, keeping each item's identity,
  position within the room, footprint, and real-world facing.
- The camera looks at the room from its {standpoint} side (as in IMAGE 1), so render each piece
  FORESHORTENED and seen from that side — a true 3D three-quarter view, never flat or top-down.
- Furniture rests ON the floor with realistic contact shadows and consistent, soft lighting.

## Keep walls bare
- Put NOTHING on the walls: no framed art, canvases, posters, mirrors, shelves, or objects.
  Wall decor is added in a later step. Apart from the openings shown in IMAGE 1, the walls are
  bare {wall_color} surfaces.

## Constraints
- Do NOT change the room's shape, camera, wall count, or proportions from IMAGE 1.
- Do NOT add, remove, or move any door or window relative to IMAGE 1.
- No text, labels, watermarks, or overlays.

Output: a single photorealistic open-box dollhouse render — IMAGE 1's exact structure,
furnished from IMAGE 2, with bare walls.
""".strip()


def build_dollhouse_add_wall_prompt(
    view: str,
    compass: str,
    room_dimensions: Optional[Dict] = None,
    presets: Optional[Dict] = None,
    correction_notes: str = "",
) -> str:
    """Step 2+ of sequential composition: edit IMAGE 1 (the current dollhouse render) by
    placing the FULL contents of IMAGE 2 (one wall's flat elevation) onto ONE wall only.

    Only a single elevation is ever shown to the model here, so it cannot confuse walls.
    correction_notes: optional feedback from a failed validation attempt, injected near the
                      top so the model fixes the specific problems found.
    """
    cfg = _DOLLHOUSE_VIEWS[view]
    open_walls_text = " and ".join(w.upper() for w in cfg["open_walls"])
    visible = cfg["visible"]
    desc = visible.get(compass, f"the {compass.upper()} wall")
    C = compass.upper()

    correction_block = (
        f"\n{correction_notes.strip()}\n" if correction_notes and correction_notes.strip() else ""
    )

    return f"""## Task
Edit IMAGE 1 — an open-box (dollhouse) render of a room — by adding wall content to the
{C} wall, and the {C} wall ONLY. Everything else in IMAGE 1 must stay pixel-for-pixel identical.
{correction_block}
## Which wall
The {C} wall = {desc}.
IMAGE 2 is a flat, straight-on elevation (front view) of exactly this wall.

## What to do on the {C} wall
The {C} wall in IMAGE 1 already shows its correct door/window openings — KEEP those exactly.
Your job is to ADD the decor and wall-mounted products from IMAGE 2 onto this same wall:
- Reproduce every wall-mounted item from IMAGE 2 EXACTLY — same TYPE, count, shape, colour, and
  proportions. If IMAGE 2 shows a framed picture / art canvas, render a framed picture / art
  canvas — NOT a window. Never substitute one object for another.
- Keep each item's left-right position along the wall, its width, and its height above the floor
  the same as in IMAGE 2 (foreshortened naturally to the wall's angle), positioned relative to the
  openings already on the wall (e.g. art that sits between two windows stays between them).
- Read IMAGE 2 as if you are standing inside the room facing the {C} wall; keep its left-to-right
  order — do NOT mirror or flip it.

## Openings (doors/windows) — DO NOT TOUCH
- The doors and windows already on this wall (from IMAGE 1) are correct. Keep them unchanged.
- IMAGE 2 also shows those same openings — do NOT add a second copy of them. Add only the decor
  and products. The wall must end with the SAME number of doors/windows it had in IMAGE 1.

## Keep EVERYTHING ELSE identical (do not touch)
- All floor furniture, the floor, the rug, lighting, and the camera angle — unchanged.
- The OTHER walls — leave exactly as they are in IMAGE 1 (do not add or change anything on them).
- The {open_walls_text} walls stay open/removed.

## Hard rules — no false alterations
- Modify ONLY the {C} wall, and only by adding IMAGE 2's decor/products. Touch nothing else.
- Do NOT add, remove, move, or duplicate any window or door.
- Do NOT turn an art piece into a window (or vice-versa); copy IMAGE 2's object types exactly.
- Do NOT change the count, size, or position of items relative to IMAGE 2.
- No text, labels, watermarks, or overlays.

Output: the same dollhouse render, now with the {C} wall showing its openings plus IMAGE 2's decor.
""".strip()


def build_composition_refine_prompt(correction_notes: str = "") -> str:
    """Targeted self-correction edit: fix ONLY the listed problems on the current composite
    (IMAGE 1), changing nothing else. Used by the convergent QA loop (Nano Banana edits
    surgically without re-drawing the rest of the scene)."""
    notes = correction_notes.strip() if correction_notes else ""
    fixes = notes if notes else (
        "Correct any item that is on the wrong wall, missing, duplicated, shown with the wrong type "
        "(for example a window where there should be a framed artwork), or in the wrong position."
    )
    return f"""You are editing a photorealistic open-box "dollhouse" room render (the provided image) that is \
already almost correct. Make only the specific corrections below, exactly like a localized inpainting edit, and \
leave every other part of the image — the camera angle, the furniture, the floor, the lighting, and all the \
other walls — completely unchanged.

Corrections to apply:
{fixes}

Keep every object's true identity: a framed artwork or canvas stays a framed artwork and is never turned into \
a window, and you do not introduce, remove, or relocate any door or window beyond what these corrections \
require. Return the same render with only these corrections applied, clean and free of any text, labels, \
watermarks, or overlays.
""".strip()


# ── Single-shot dollhouse prompts ────────────────────────────────────────────
# Images always arrive in fixed compass order:
#   Image 1 = floor render, Image 2 = NORTH, Image 3 = SOUTH,
#   Image 4 = EAST, Image 5 = WEST.
# Two separate prompts map each compass wall to its on-screen position for
# that specific camera angle.

_ONESHOT_REMOVE_SOUTH = """Create a photorealistic architectural dollhouse visualization from the provided isometric room render and four wall images.

INPUT IMAGE MAPPING

Image 1 = Top-down isometric layout of the room — authoritative source for room geometry, furniture positions, footprints, spacing, and in-plan facing. It is NOT the final camera; the final three-quarter dollhouse camera is described below.

Image 2 = North Wall
Position: Far/back side of room

Image 3 = South Wall
Position: Near side — this wall is REMOVED (camera stands outside it)

Image 4 = East Wall
Position: Right edge of room

Image 5 = West Wall
Position: Left edge of room

ROOM RECONSTRUCTION RULES

North Wall = Image 2
South Wall = Image 3
East Wall  = Image 4
West Wall  = Image 5

Image 1 (isometric render) is the ground-truth source of:
- room dimensions
- wall positions
- furniture locations
- object orientations
- spacing between products

Wall images provide appearance, materials, colors, windows, doors, trims, and decorative details only.

Do not:
- swap wall locations
- mirror walls
- rotate walls
- alter room proportions
- move furniture
- add products
- remove products

DOLLHOUSE CUTAWAY VIEW

Camera Position:
Outside the South Wall looking toward the North Wall.

Remove the South Wall completely.

Keep the North, East, and West walls fully visible and accurately textured.

WALL PLACEMENT IN THIS VIEW (fixed by the camera — do not swap or mirror)

Because the camera stands outside the South wall facing North, each wall occupies one fixed on-screen position. Render each wall's appearance from its own image onto exactly the position below:
- NORTH wall (Image 2) = the BACK wall: runs left-to-right across the far side of the room.
- EAST wall  (Image 4) = the RIGHT-side wall: recedes from the front-right toward the back-right corner.
- WEST wall  (Image 5) = the LEFT-side wall: recedes from the front-left toward the back-left corner.
- SOUTH wall (Image 3) = the removed/open side nearest the camera: do NOT draw it, and do NOT place its contents on any other wall.

Never put one wall's contents on a different wall, and never swap the LEFT (West) and RIGHT (East) side walls.

WALL FIDELITY

Each wall image is the GROUND TRUTH for that wall. Render every wall exactly as its own image shows — do NOT generate, add, remove, change, resize, or relocate anything on a wall beyond what that wall's image already contains (this includes its art/canvas, doors, and windows).

Camera settings:
- 35–45 degree viewing angle
- slightly elevated perspective
- wide architectural lens
- entire room visible in one frame

FURNITURE & WALL-ART FIDELITY

This render's camera looks into the room from the open South side (the three-quarter dollhouse view described above), so each item is seen from that angle. Use Image 1 only for layout — it is top-down — and keep that layout exactly:

- Keep every furniture item in its EXACT location as shown in Image 1. Do not move, shift, slide, or reposition anything.
- Do NOT rotate, spin, or re-orient furniture to face the camera. Preserve each item's true real-world orientation; only the viewing angle changes.
- Do NOT reshape, rescale, restyle, or reconstruct any object.
- Reproduce every wall-mounted item (framed art, canvas, mirror, shelf, sconce) EXACTLY as shown in its wall image — same artwork, same type, same count, same colors, same design. Never repaint, alter, or swap a canvas/artwork, and never turn it into a window or any other object.

PLACEMENT & VISIBILITY REQUIREMENTS

Preserve exact furniture placement and scale from Image 1. No furniture may be relocated, resized, or duplicated.
Render each item only as it is genuinely seen from this angle: an item may be partially or fully occluded by OTHER FURNITURE in front of it — that is natural and correct. Do NOT move, shrink, duplicate, or re-arrange items to force every piece into view.
No furniture may be hidden behind a WALL — the near wall is open/removed in the dollhouse view, so nothing should be blocked by a wall.
Keep the whole room and its overall layout in frame; do not crop away large parts of the room.

RENDER STYLE

- ultra photorealistic
- luxury interior visualization
- realistic global illumination
- ray-traced shadows
- physically based materials (PBR)
- furniture catalog quality
- crisp details
- clean neutral background
- high-resolution architectural render"""


_ONESHOT_REMOVE_NORTH = """Create a photorealistic architectural dollhouse visualization from the provided isometric room render and four wall images.

INPUT IMAGE MAPPING

Image 1 = Top-down isometric layout of the room — authoritative source for room geometry, furniture positions, footprints, spacing, and in-plan facing. It is NOT the final camera; the final three-quarter dollhouse camera is described below.

Image 2 = South Wall
Position: Far/back side of room

Image 3 = North Wall
Position: Near side — this wall is REMOVED (camera stands outside it)

Image 4 = West Wall
Position: Right edge of room

Image 5 = East Wall
Position: Left edge of room

ROOM RECONSTRUCTION RULES

South Wall = Image 2
North Wall = Image 3
West Wall  = Image 4
East Wall  = Image 5

Image 1 (isometric render) is the ground-truth source of:
- room dimensions
- wall positions
- furniture locations
- object orientations
- spacing between products

Wall images provide appearance, materials, colors, windows, doors, trims, and decorative details only.

Do not:
- swap wall locations
- mirror walls
- rotate walls
- alter room proportions
- move furniture
- add products
- remove products

DOLLHOUSE CUTAWAY VIEW

Camera:
Build the three-quarter dollhouse view (see Camera settings below) from Image 1's layout. Keep the SAME orientation as Image 1 — do NOT orbit, rotate, flip, or mirror relative to it, and do NOT re-arrange its furniture: the wall at the BACK of Image 1 stays the back wall, the LEFT side stays the left wall, the RIGHT side stays the right wall, and the NEAR/front side is the open (removed) side.

Cutaway:
Remove the near/open wall at the FRONT of Image 1 (its elevation is Image 3) — do NOT draw it, and do NOT place its contents on any other wall.
Keep the far/back wall and both side walls fully visible and accurately textured.

WALL PLACEMENT IN THIS VIEW (fixed by Image 1's geometry — do not swap or mirror)

Each wall occupies one fixed on-screen position, matching Image 1's geometry exactly. Render each wall's appearance from its own image onto exactly the position below:
- BACK wall (far side, runs left-to-right across the back of the room) = Image 2 (South Wall).
- RIGHT-side wall (recedes from the front-right toward the back-right corner) = Image 4 (West Wall).
- LEFT-side wall (recedes from the front-left toward the back-left corner) = Image 5 (East Wall).
- NEAR/open wall (the removed front side) = Image 3 (North Wall): do NOT draw it, and do NOT place its contents on any other wall.

Never put one wall's contents on a different wall, and never swap the LEFT and RIGHT side walls.

WALL FIDELITY

Each wall image is the GROUND TRUTH for that wall. Render every wall exactly as its own image shows — do NOT generate, add, remove, change, resize, or relocate anything on a wall beyond what that wall's image already contains (this includes its art/canvas, doors, and windows).

Camera settings:
- 35–45 degree viewing angle
- slightly elevated perspective
- wide architectural lens
- entire room visible in one frame

FURNITURE & WALL-ART FIDELITY

This render's camera looks into the room from the open near side (the three-quarter dollhouse view described above). Use Image 1 only for layout — it is top-down — and keep that layout exactly:

- Keep every furniture item in its EXACT location as shown in Image 1. Do not move, shift, slide, or reposition anything.
- Do NOT rotate, spin, mirror, or re-orient furniture. Reproduce each item exactly as it appears in Image 1 — same position, same orientation, same visible side.
- Do NOT reshape, rescale, restyle, or reconstruct any object.
- Reproduce every wall-mounted item (framed art, canvas, mirror, shelf, sconce) EXACTLY as shown in its wall image — same artwork, same type, same count, same colors, same design. Never repaint, alter, or swap a canvas/artwork, and never turn it into a window or any other object.

PLACEMENT & VISIBILITY REQUIREMENTS

Preserve exact furniture placement and scale from Image 1. No furniture may be relocated, resized, or duplicated.
Render each item only as it is genuinely seen from this angle: an item may be partially or fully occluded by OTHER FURNITURE in front of it — that is natural and correct. Do NOT move, shrink, duplicate, or re-arrange items to force every piece into view.
No furniture may be hidden behind a WALL — the near wall is open/removed in the dollhouse view, so nothing should be blocked by a wall.
Keep the whole room and its overall layout in frame; do not crop away large parts of the room.

RENDER STYLE

- ultra photorealistic
- luxury interior visualization
- realistic global illumination
- ray-traced shadows
- physically based materials (PBR)
- furniture catalog quality
- crisp details
- clean neutral background
- high-resolution architectural render"""


def build_dollhouse_oneshot_prompt(
    removed_wall: str,
    blank_compass: Optional[List[str]] = None,
) -> str:
    """Return the prompt for the cutaway that removes `removed_wall` ("north" or "south").

    Images must arrive in fixed compass order [floor, N, S, E, W].
    blank_compass: list of compass walls that have no elevation and are visible in
                   this view — model is told to render them as plain painted surfaces.
    """
    base = _ONESHOT_REMOVE_SOUTH if removed_wall.lower() == "south" else _ONESHOT_REMOVE_NORTH

    if not blank_compass:
        return base

    blank_lines = "\n".join(
        f"- {c.upper()} wall: NO elevation provided — render as a plain painted surface. "
        "Do NOT add any products, art, windows, or doors to this wall."
        for c in blank_compass
    )
    return base + f"\n\nBLANK WALLS (no elevation image — plain surface only)\n{blank_lines}"
