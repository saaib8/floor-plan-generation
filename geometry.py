"""Deterministic dollhouse geometry + compositing (no 3D engine, no OpenCV).

The generative model keeps hallucinating room structure: it invents extra walls,
mirrors wall elevations, and renders both "opposite" views from the same angle.
None of that is creative work — it is fixed geometry that we can compute exactly
from the room dimensions and a chosen camera.

This module computes, for a given view, the on-screen quadrilateral of each wall
and the floor under a fixed isometric camera, renders a flat "guide" of the open
box, and warps each flat wall elevation onto its wall quad via a homography. Because
the camera, the wall quads, and the elevation->wall mapping are all computed:

  * an extra wall is impossible (we draw exactly two),
  * a mirrored wall is impossible (the elevation's left/right map to fixed ends),
  * the two views are guaranteed opposite (front camera = SOUTH-EAST corner, back
    camera = NORTH-WEST corner).

Coordinate system (metres):
    x = EAST (0..W), y = NORTH (0..D), z = UP (0..H)
Walls: NORTH = y=D, SOUTH = y=0, EAST = x=W, WEST = x=0.

Views (camera stands over the corner where the two REMOVED walls meet and looks
across to the opposite back corner):
    front -> camera at SOUTH-EAST, far walls = WEST (left) + NORTH (right)
    back  -> camera at NORTH-WEST, far walls = EAST (left) + SOUTH (right)
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

Point = Tuple[float, float]

# Isometric look: ground axes drawn at ±30° from horizontal; height rises vertically.
_ISO_COS = math.cos(math.radians(30.0))   # horizontal component per metre of ground
_ISO_SIN = math.sin(math.radians(30.0))   # vertical component per metre of ground
_HEIGHT_RATIO = 0.85                       # vertical px per metre of wall height / ground px


def _dims(room_dimensions: Optional[Dict]) -> Tuple[float, float, float]:
    rd = room_dimensions or {}
    w = float(rd.get("width") or 0) or 4.0
    d = float(rd.get("depth") or rd.get("length") or 0) or 4.0
    h = float(rd.get("wall_height") or rd.get("height") or 0) or 4.0
    return w, d, h


def _ground_uv(view: str, x: float, y: float, W: float, D: float) -> Tuple[float, float]:
    """Map a ground point to unscaled isometric (su, sv) where the back corner is at (0,0)
    and sv grows downward toward the near (camera) corner."""
    if view == "front":
        # along NORTH wall = x (0..W); along WEST wall = (D - y) (0..D)
        a, b = x, (D - y)
    else:  # back
        # along EAST wall = y (0..D); along SOUTH wall = (W - x) (0..W)
        a, b = y, (W - x)
    return (a - b), (a + b)


def _wall_ends(view: str, compass: str, W: float, D: float) -> Tuple[Point, Point]:
    """Ground coords of a wall's (LEFT_end, RIGHT_end) as read in its flat elevation
    (standing inside the room facing that wall). This fixed mapping is what prevents
    any left/right mirroring when the elevation is warped onto the wall."""
    # facing a wall, "left" and "right" ends in world coords:
    #   facing NORTH (+y): left = WEST end (x=0),  right = EAST end (x=W)
    #   facing SOUTH (-y): left = EAST end (x=W),  right = WEST end (x=0)
    #   facing EAST  (+x): left = NORTH end (y=D), right = SOUTH end (y=0)
    #   facing WEST  (-x): left = SOUTH end (y=0), right = NORTH end (y=D)
    ends = {
        "north": ((0.0, D), (W, D)),
        "south": ((W, 0.0), (0.0, 0.0)),
        "east":  ((W, D), (W, 0.0)),
        "west":  ((0.0, 0.0), (0.0, D)),
    }
    return ends[compass]


# Which two walls are the visible FAR walls per view, and their left/right screen role.
_VIEW_WALLS = {
    "front": {"left": "west", "right": "north"},
    "back":  {"left": "east", "right": "south"},
}


def compute_dollhouse_geometry(
    room_dimensions: Optional[Dict],
    view: str,
    img_w: int = 1024,
    img_h: int = 768,
    margin: float = 0.10,
    openings_by_wall: Optional[Dict[str, List[Dict]]] = None,
) -> Dict:
    """Compute the on-screen geometry of the open box for `view`.

    Returns a dict:
      {
        "view": str,
        "floor": [p_back, p_right, p_near, p_left],   # screen quad of the floor
        "walls": {compass: {"role": "left"/"right",
                            "quad": [tl, tr, br, bl],  # screen, elevation-aligned
                            "left_end": p, "right_end": p}},
        "openings": {compass: [[tl, tr, br, bl], ...]},  # screen quads
      }
    All points are (x, y) pixel tuples in the final image.
    """
    W, D, H = _dims(room_dimensions)

    # Collect all 8 box corners (floor z=0 and top z=H) to auto-fit the projection.
    corners_ground = [(0.0, 0.0), (W, 0.0), (0.0, D), (W, D)]
    raw = []
    for (x, y) in corners_ground:
        su, sv = _ground_uv(view, x, y, W, D)
        raw.append((su, sv, 0.0))
        raw.append((su, sv, H))

    sus = [su for su, _, _ in raw]
    svs = [sv for _, sv, _ in raw]
    # screen (pre-scale): X = su*COS ; Y = sv*SIN - z*HEIGHT_RATIO
    pre = [(su * _ISO_COS, sv * _ISO_SIN - z * _HEIGHT_RATIO) for su, sv, z in raw]
    xs = [p[0] for p in pre]
    ys = [p[1] for p in pre]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0

    avail_w = img_w * (1 - 2 * margin)
    avail_h = img_h * (1 - 2 * margin)
    scale = min(avail_w / span_x, avail_h / span_y)
    off_x = (img_w - span_x * scale) / 2 - min_x * scale
    off_y = (img_h - span_y * scale) / 2 - min_y * scale

    def project(x: float, y: float, z: float) -> Point:
        su, sv = _ground_uv(view, x, y, W, D)
        px = su * _ISO_COS * scale + off_x
        py = (sv * _ISO_SIN - z * _HEIGHT_RATIO) * scale + off_y
        return (px, py)

    # Floor quad (back corner, right, near corner, left) — order for a filled polygon.
    if view == "front":
        floor = [project(0, D, 0), project(W, D, 0), project(W, 0, 0), project(0, 0, 0)]
    else:
        floor = [project(W, 0, 0), project(W, D, 0), project(0, D, 0), project(0, 0, 0)]

    walls: Dict[str, Dict] = {}
    roles = _VIEW_WALLS[view]
    for role, compass in roles.items():
        (lx, ly), (rx, ry) = _wall_ends(view, compass, W, D)
        bl = project(lx, ly, 0.0)   # bottom-left (left end, floor)
        br = project(rx, ry, 0.0)   # bottom-right (right end, floor)
        tl = project(lx, ly, H)     # top-left
        tr = project(rx, ry, H)     # top-right
        walls[compass] = {
            "role": role,
            "quad": [tl, tr, br, bl],
            "left_end": (lx, ly),
            "right_end": (rx, ry),
        }

    # Opening quads (windows/doors) on each visible wall.
    openings_out: Dict[str, List[List[Point]]] = {}
    ob = openings_by_wall or {}
    for compass in roles.values():
        specs = ob.get(compass) or []
        if not specs:
            continue
        (lx, ly), (rx, ry) = _wall_ends(view, compass, W, D)
        wall_len = math.hypot(rx - lx, ry - ly) or 1.0
        quads: List[List[Point]] = []
        for op in specs:
            try:
                t = float(op.get("position_from_left", 0.5))
            except (TypeError, ValueError):
                t = 0.5
            try:
                ow = float(op.get("width_m") or 0) or 1.0
            except (TypeError, ValueError):
                ow = 1.0
            otype = (op.get("type") or "window").lower()
            is_door = "door" in otype
            oh = float(op.get("height_m") or (2.05 if is_door else 1.2))
            sill = 0.0 if is_door else float(op.get("sill_height") or 0.9)
            half = (ow / wall_len) / 2.0
            t0, t1 = max(0.0, t - half), min(1.0, t + half)

            def along(tt: float, z: float) -> Point:
                gx = lx + (rx - lx) * tt
                gy = ly + (ry - ly) * tt
                return project(gx, gy, z)

            z0, z1 = sill, min(H, sill + oh)
            quads.append([along(t0, z1), along(t1, z1), along(t1, z0), along(t0, z0)])
        if quads:
            openings_out[compass] = quads

    return {"view": view, "floor": floor, "walls": walls, "openings": openings_out, "project": project}


def _hex_to_rgb(hex_color: str, default=(243, 239, 232)) -> Tuple[int, int, int]:
    try:
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore
    except Exception:
        return default


def _shade(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)  # type: ignore


def render_guide(
    geometry: Dict,
    img_w: int = 1024,
    img_h: int = 768,
    wall_color: str = "#F3EFE8",
    floor_color: str = "#D9C7A8",
    bg_color: str = "#EFEAE2",
) -> Image.Image:
    """Render a flat-shaded open-box guide: floor + the two far walls + openings.
    This is a deterministic structural template (no furniture, no decor)."""
    img = Image.new("RGB", (img_w, img_h), _hex_to_rgb(bg_color))
    draw = ImageDraw.Draw(img)

    wall_rgb = _hex_to_rgb(wall_color)
    floor_rgb = _hex_to_rgb(floor_color)

    # Floor
    draw.polygon([tuple(p) for p in geometry["floor"]], fill=floor_rgb)

    # Walls — slightly different shading for left vs right so the corner reads clearly.
    for compass, w in geometry["walls"].items():
        factor = 1.0 if w["role"] == "right" else 0.92
        draw.polygon([tuple(p) for p in w["quad"]], fill=_shade(wall_rgb, factor))

    # Openings (windows = light glaze, doors = darker panel)
    for compass, quads in geometry.get("openings", {}).items():
        for q in quads:
            draw.polygon([tuple(p) for p in q], fill=(212, 226, 235))

    return img


def _perspective_coeffs(dst: List[Point], src: List[Point]) -> List[float]:
    """8 coefficients mapping DST (output) -> SRC (input) for PIL Image.PERSPECTIVE."""
    A = []
    B = []
    for (xd, yd), (xs, ys) in zip(dst, src):
        A.append([xd, yd, 1, 0, 0, 0, -xs * xd, -xs * yd])
        B.append(xs)
        A.append([0, 0, 0, xd, yd, 1, -ys * xd, -ys * yd])
        B.append(ys)
    res = np.linalg.solve(np.array(A, dtype=float), np.array(B, dtype=float))
    return res.tolist()


def warp_elevation_onto_quad(
    elevation: Image.Image,
    quad: List[Point],
    canvas_size: Tuple[int, int],
) -> Image.Image:
    """Warp a flat elevation image onto `quad` (order: tl, tr, br, bl) inside a transparent
    RGBA canvas of `canvas_size`. The quad corners are elevation-aligned, so no mirroring."""
    cw, ch = canvas_size
    src = elevation.convert("RGBA")
    sw, sh = src.size
    src_corners = [(0, 0), (sw, 0), (sw, sh), (0, sh)]  # tl, tr, br, bl
    coeffs = _perspective_coeffs([tuple(p) for p in quad], src_corners)
    warped = src.transform((cw, ch), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    # Clip strictly to the quad polygon (perspective sampling can bleed at edges).
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in quad], fill=255)
    out = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    out.paste(warped, (0, 0), mask)
    return out


def occluder_mask(
    furnished: Image.Image,
    guide: Image.Image,
    geometry: Dict,
    threshold: int = 42,
) -> Image.Image:
    """Build an L-mode mask of furniture that occludes the (bare) walls.

    `furnished` and `guide` must be the SAME size and share the SAME camera/geometry, so any
    large colour difference inside a wall quad is furniture standing in front of that wall.
    Returns 255 where furniture occludes a wall (decor must not be painted over it)."""
    cw, ch = guide.size
    f = np.asarray(furnished.convert("RGB").resize((cw, ch)), dtype=np.int16)
    g = np.asarray(guide.convert("RGB"), dtype=np.int16)
    diff = np.abs(f - g).sum(axis=2)
    furniture = (diff > threshold).astype("uint8") * 255

    walls_region = Image.new("L", (cw, ch), 0)
    d = ImageDraw.Draw(walls_region)
    for w in geometry["walls"].values():
        d.polygon([tuple(p) for p in w["quad"]], fill=255)
    wr = np.asarray(walls_region, dtype="uint8")

    mask = np.where(wr > 0, furniture, 0).astype("uint8")
    return Image.fromarray(mask, mode="L")


def composite_walls(
    base: Image.Image,
    geometry: Dict,
    elevations: Dict[str, bytes],
    occluder_mask: Optional[Image.Image] = None,
) -> Image.Image:
    """Warp every provided wall elevation onto its quad and composite over `base`.

    elevations:    {compass: png/jpeg bytes of that wall's flat elevation}
    occluder_mask: optional L-mode mask (255 = foreground furniture that must stay on top);
                   wall decor is suppressed where the mask is set so furniture is not painted over.
    """
    canvas = base.convert("RGBA")
    cw, ch = canvas.size
    for compass, w in geometry["walls"].items():
        elev_bytes = elevations.get(compass)
        if not elev_bytes:
            continue
        elev = Image.open(io.BytesIO(elev_bytes))
        layer = warp_elevation_onto_quad(elev, w["quad"], (cw, ch))
        if occluder_mask is not None:
            keep = Image.new("L", (cw, ch), 0)
            ImageDraw.Draw(keep).polygon([tuple(p) for p in w["quad"]], fill=255)
            # remove decor where furniture occludes
            inv = occluder_mask.point(lambda v: 255 - v)
            keep = Image.composite(keep, Image.new("L", (cw, ch), 0), inv)
            r, g, b, a = layer.split()
            a = Image.composite(a, Image.new("L", (cw, ch), 0), keep)
            layer = Image.merge("RGBA", (r, g, b, a))
        canvas = Image.alpha_composite(canvas, layer)
    return canvas
