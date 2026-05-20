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
        x_m = item.get("x_m")
        y_m = item.get("y_m")

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
    return  (batch_items, room_dimensions, presets)


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

    n = len(payload.get("products", []))
    constraints = (
        f"HARD CONSTRAINTS: "
        f"(1) Render exactly the same {n} furniture item(s) visible in IMAGE 1 — same products, same positions, same layout. "
        f"(2) Match each product's material and finish to its reference photo (IMAGE 2+). "
        f"(3) No extra furniture, accessories, rugs, lamps, or decor not in IMAGE 1. "
        f"(4) No text, labels, dimension lines, or watermarks."
    )

    return "\n\n".join(filter(None, [appearance, view_line, refs, constraints])).strip() + (
        "\n\nOutput: one photorealistic render, no overlays, no on-image text."
    )


def build_floor_plan_prompt(
    payload: dict,
    image_order: Optional[List[str]] = None,
    has_guide: bool = False,
    view: str = "isometric",
    iso_base: bool = False,
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
    app: List[str] = ["Ultra-realistic architectural visualization"]

    lt = (lighting.get("type") or "natural_daylight").replace("_", " ")
    if "natural" in lt.lower() or "daylight" in lt.lower():
        app.append("warm natural daylight, soft ambient light from windows")
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
        "realistic material textures (fabric, wood, metal)",
        "no sticker or cutout look",
        "editorial photography quality",
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
            "IMAGE 1 is a colour-blob placement guide — each blob marks WHERE a product goes. "
            "Match each blob colour to its product reference photo:"
        )
        for i, pid in enumerate(img_order, start=1):
            hc = products_by_id.get(pid, {}).get("hex_color", "")
            if hc:
                ref_lines.append(f"  {hc} → IMAGE {i + 1}")
        ref_lines.append(
            "Replace every coloured blob with its photorealistic furniture piece. "
            "Show bare floor everywhere else."
        )
    elif img_order:
        ref_lines.append("Product reference photos — render each item to exactly match its image:")
        for i, pid in enumerate(img_order, start=1):
            ref_lines.append(f"  IMAGE {i + 1} = product id={pid}")

    refs = "\n".join(ref_lines)

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
        if rot == 0:
            rotation_lines.append(f"  {cat}: 0° — use reference photo orientation as-is.")
        elif rot == 90:
            long_axis = "width" if d > w else "depth"
            rotation_lines.append(
                f"  {cat}: 90° CW — its longer dimension runs LEFT↔RIGHT; front faces EAST."
            )
        elif rot == 180:
            rotation_lines.append(
                f"  {cat}: 180° — completely flipped from reference photo; front faces SOUTH."
            )
        elif rot == 270:
            rotation_lines.append(
                f"  {cat}: 270° CW — its longer dimension runs LEFT↔RIGHT; front faces WEST."
            )
        else:
            rotation_lines.append(f"  {cat}: {rot}° CW from reference photo; front faces {facing}.")

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

    # ── Hard constraints ───────────────────────────────────────────────────────
    n = len(payload.get("products", []))
    constraints = (
        f"HARD CONSTRAINTS: "
        f"(1) Exactly {n} furniture item(s) — no extra accessories, rugs, lamps, plants, or decor. "
        f"(2) No architectural elements beyond those in the geometry JSON. "
        f"(3) Render each product to exactly match its reference image — same shape, style, upholstery, finish. "
        f"(4) No text, labels, dimension lines, or watermarks in the output."
    )

    return "\n\n".join(filter(None, [appearance, view_line, rotation_block, refs, geometry_block, constraints])).strip() + (
        "\n\nOutput: one photorealistic render, no overlays, no on-image text."
    )
