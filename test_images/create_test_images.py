#!/usr/bin/env python3
"""
Generate simple test source images for the floor plan generation system.

Produces:
  room_base.png   - clean room floor plan outline (white walls, floor, door, window)
  room_guide.png  - same room with colored product placement blobs
  room_alt.png    - L-shaped room variant
"""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent


# ── helpers ──────────────────────────────────────────────────────────────────

def new_img(w=1024, h=1024, bg="#F8F6F2"):
    return Image.new("RGB", (w, h), bg), ImageDraw.Draw(Image.new("RGB", (w, h), bg))


def _cm_scaler(rx, ry, rw, rh, wall, room_w_cm, room_h_cm):
    """Return a function that converts (x,y,w,h) in cm to pixel box."""
    sx = (rw - wall * 2) / room_w_cm
    sy = (rh - wall * 2) / room_h_cm
    def scale(cx, cy, cw=None, ch=None):
        px = int(rx + wall + cx * sx)
        py = int(ry + wall + cy * sy)
        if cw is None:
            return px, py
        return px, py, int(px + cw * sx), int(py + ch * sy)
    return scale


# ── room_base.png ─────────────────────────────────────────────────────────────

def make_floor_base(path: Path, img_w=1024, img_h=1024):
    """
    Simple rectangular living room (500×700 cm) with:
    - north wall window at 170 cm from left, 160 cm wide
    - south wall door  at 205 cm from left, 90  cm wide
    """
    img = Image.new("RGB", (img_w, img_h), "#F8F6F2")
    d = ImageDraw.Draw(img)

    pad  = 80
    wall = 22
    rx, ry = pad, pad
    rw = img_w - pad * 2
    rh = img_h - pad * 2

    # Floor fill
    d.rectangle([rx + wall, ry + wall, rx + rw - wall, ry + rh - wall], fill="#EDE8E0")

    # Wall outline
    d.rectangle([rx, ry, rx + rw, ry + rh], outline="#2A2520", width=wall)

    cm = _cm_scaler(rx, ry, rw, rh, wall, 500, 700)

    # North wall window (170 cm from left, 160 cm wide)
    wx1, _ = cm(170, 0)
    wx2, _ = cm(170 + 160, 0)
    # Gap in north wall
    d.line([(wx1, ry), (wx2, ry)], fill="#F8F6F2", width=wall + 2)
    # Window lines
    d.line([(wx1, ry - 4), (wx2, ry - 4)], fill="#7AACBE", width=4)
    d.line([(wx1, ry + 4), (wx2, ry + 4)], fill="#7AACBE", width=3)
    d.line([(wx1, ry - 4), (wx1, ry + wall)], fill="#7AACBE", width=3)
    d.line([(wx2, ry - 4), (wx2, ry + wall)], fill="#7AACBE", width=3)

    # South wall door (205 cm from left, 90 cm wide)
    dx1, _ = cm(205, 0)
    dx2, _ = cm(205 + 90, 0)
    dy = ry + rh
    # Gap in south wall
    d.line([(dx1, dy), (dx2, dy)], fill="#F8F6F2", width=wall + 2)
    # Door leaf (thin line from hinge at dx2 to dx1)
    d.line([(dx2, dy - 1), (dx1, dy - 1)], fill="#9A8A78", width=2)
    # Swing arc
    door_w_px = dx2 - dx1
    d.arc(
        [dx1, dy - door_w_px, dx1 + door_w_px, dy],
        start=180, end=270, fill="#9A8A78", width=2,
    )

    # Grid (faint 1m squares)
    sx = (rw - wall * 2) / 5    # 5 m wide
    sy = (rh - wall * 2) / 7    # 7 m tall
    for i in range(1, 5):
        x = int(rx + wall + i * sx)
        d.line([(x, ry + wall), (x, ry + rh - wall)], fill="#DDD8D0", width=1)
    for j in range(1, 7):
        y = int(ry + wall + j * sy)
        d.line([(rx + wall, y), (rx + rw - wall, y)], fill="#DDD8D0", width=1)

    img.save(path, "PNG")
    print(f"  Saved {path.name}")


# ── room_guide.png ────────────────────────────────────────────────────────────

def make_floor_guide(path: Path, img_w=1024, img_h=1024):
    """
    Room guide overlay for the demo payload:
    - Chair left  (hex #E63946) at (165,540), 85×82 cm
    - Chair right (hex #F4A261) at (335,540), 85×82 cm
    - Coffee table(hex #2A9D8F) at (250,540), ø90 cm
    """
    img = Image.new("RGB", (img_w, img_h), "#FFFFFF")
    d = ImageDraw.Draw(img)

    pad  = 80
    wall = 22
    rx, ry = pad, pad
    rw = img_w - pad * 2
    rh = img_h - pad * 2

    # Room border
    d.rectangle([rx, ry, rx + rw, ry + rh], outline="#CCCCCC", width=3)

    cm = _cm_scaler(rx, ry, rw, rh, wall, 500, 700)

    # Chair left
    d.rectangle(cm(165 - 42, 540 - 41, 85, 82), fill="#E63946CC")

    # Chair right
    d.rectangle(cm(335 - 42, 540 - 41, 85, 82), fill="#F4A261CC")

    # Coffee table (circle)
    bx, by, bx2, by2 = cm(250 - 45, 540 - 45, 90, 90)
    d.ellipse([bx, by, bx2, by2], fill="#2A9D8FCC")

    img.save(path, "PNG")
    print(f"  Saved {path.name}")


# ── room_alt.png ──────────────────────────────────────────────────────────────

def make_room_alt(path: Path, img_w=1024, img_h=1024):
    """Simple blank L-shaped room outline for open-ended testing."""
    img = Image.new("RGB", (img_w, img_h), "#F8F6F2")
    d = ImageDraw.Draw(img)

    wall = 20
    # L-shape polygon corners (pixel coords)
    pts = [
        (120, 120), (780, 120), (780, 550),
        (500, 550), (500, 900), (120, 900),
    ]

    # Floor
    d.polygon(pts, fill="#EDE8E0", outline="#2A2520")

    # Redraw border thicker
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        d.line([a, b], fill="#2A2520", width=wall)

    # Simple door on bottom wall
    d.line([(200, 900), (310, 900)], fill="#F8F6F2", width=wall + 2)
    d.line([(200, 899), (310, 899)], fill="#9A8A78", width=2)
    d.arc([200, 900 - 110, 310, 900], start=180, end=270, fill="#9A8A78", width=2)

    img.save(path, "PNG")
    print(f"  Saved {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating test images…")
    make_floor_base(OUT / "room_base.png")
    make_floor_guide(OUT / "room_guide.png")
    make_room_alt(OUT / "room_alt.png")
    print("Done.")
