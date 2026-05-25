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

    room_area = room_w * room_l

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

        # Wall gaps (from product edge to room boundary)
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

        footprint_pct = (eff_w * eff_d) / room_area * 100 if room_area else 0
        width_pct = eff_w / room_w * 100 if room_w else 0

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
                "- All four walls are SOLID with NO gaps — there are NO doors or windows in this room"
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
            f"All four walls must be completely SOLID with no gaps, no glass, no frames, no openings of any kind.\n"
        )
    constraints += (
        f"(3) PRODUCT IDENTITY: Copy ONLY the named product from each reference photo. Ignore staging props visible in the photo. "
        f"Match the named product's design, colors, and structure exactly — do NOT redesign or restyle.\n"
        f"(4) No text, labels, or watermarks.\n"
        f"(5) The ONLY objects in the render are: the {n} named products + the room itself (walls, floor). Nothing else."
    )
    if n_openings:
        constraints += f"\n(6) Render all {n_openings} opening(s) as simple, plain doors/windows in the correct walls — no extra hardware or decorative details."

    return "\n\n".join(filter(None, [appearance, view_line, rotation_block, refs, identity_block, geometry_block, spatial_spec, correction_notes, openings_spec, constraints])).strip() + (
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
        wall_h2 = float(wall.get("height") or 2.8)
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
    wall_h = float(wall.get("height") or 2.8)

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
            f"\n(6) Render all {n_openings} opening(s) as simple, plain doors/windows — "
            "no extra hardware or decorative panels beyond what the guide shows."
        )

    return "\n\n".join(filter(None, [
        appearance, view_line, rotation_block, refs, identity_block,
        geometry_block, correction_notes, openings_spec, constraints,
    ])).strip() + "\n\nOutput: one photorealistic front-facing wall elevation render, no overlays, no on-image text."


def build_composition_prompt(
    room_dimensions: Optional[Dict] = None,
    wall_labels: Optional[List[str]] = None,
    presets: Optional[Dict] = None,
    n_walls: int = 0,
) -> str:
    """Prompt for the final isometric room composition from floor + wall reference images."""
    rd = room_dimensions or {}
    w = float(rd.get("width") or 0)
    d = float(rd.get("height") or rd.get("depth") or 0)
    wall_h = float(rd.get("wall_height") or 2.8)
    unit = rd.get("unit", "m")

    if w and d:
        room_desc = f"{w:.1f} × {d:.1f} × {wall_h:.1f} {unit} (width × depth × height)"
    else:
        room_desc = "see reference images for proportions"

    wall_refs = "\n".join(
        f"- IMAGE {i + 2}: Front elevation of {wall_labels[i] if wall_labels and i < len(wall_labels) else f'wall {i+1}'}"
        for i in range(n_walls)
    )

    pr = presets or {}
    style_parts = []
    if pr.get("room_type"):
        style_parts.append(pr["room_type"])
    if pr.get("decor_style"):
        style_parts.append(pr["decor_style"])
    style_line = ("Style: " + ", ".join(style_parts) + ".") if style_parts else ""

    return f"""## Task
Create ONE photorealistic isometric interior room view that composites all provided reference surfaces into a single coherent scene.
Every surface must faithfully reflect its reference image — same materials, same furniture, same openings.
{style_line}

## Room dimensions
{room_desc}

## Reference images provided
- IMAGE 1: Top-down isometric view of the floor — ground truth for furniture layout, floor material, and product positions
{wall_refs}

## Composition rules
1. Render in a 3/4 isometric perspective (camera ~30–45° above, angled toward the front corner of the room so floor and two walls are visible)
2. The floor must match IMAGE 1 exactly: same furniture, same positions, same floor material/texture
3. Each wall must match its reference elevation: same openings (windows/doors at the same positions), same wall finish, same any furniture placed against it
4. Maintain correct proportions based on room dimensions: {room_desc}
5. Consistent lighting — soft ambient light from above, shadows matching the isometric camera angle
6. Walls and floor meet at correct 90° angles; no distortion

## Hard constraints
- No new furniture, accessories, or architectural elements beyond what appears in the reference images
- No text, labels, dimension lines, or watermarks on the output
- All provided surfaces (floor + {n_walls} wall(s)) must be visible and correctly oriented
- Photorealistic render quality — no cartoon or sketch style

Output: a single photorealistic isometric room render, no overlays, no on-image text.
""".strip()
