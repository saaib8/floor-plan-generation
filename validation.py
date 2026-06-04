"""Spatial validation of generated floor plan images using GPT-4o Vision.

Compares a generated isometric render against the 2D guide image to verify
that furniture positions, sizes, and rotations match the user's layout.
Produces structured scores and correction notes for retry prompts.
"""

import base64
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ── Configuration via environment variables ──────────────────────────────────

VALIDATION_ENABLED = os.getenv("VALIDATION_ENABLED", "true").lower() in ("true", "1", "yes")
VALIDATION_MODEL = os.getenv("VALIDATION_MODEL", "gpt-4o")
VALIDATION_MAX_ATTEMPTS = int(os.getenv("VALIDATION_MAX_ATTEMPTS", "3"))
VALIDATION_SCORE_THRESHOLD = float(os.getenv("VALIDATION_SCORE_THRESHOLD", "0.75"))
VALIDATION_CANDIDATES_PER_ATTEMPT = int(os.getenv("VALIDATION_CANDIDATES_PER_ATTEMPT", "1"))
VALIDATION_TIMEOUT = int(os.getenv("VALIDATION_TIMEOUT", "60"))

# ── Composition (dollhouse) validation — separate knobs so the user-facing
#    composite can be held to a higher bar without inflating wall/floor cost.
#    The QA loop does best-of-N SELECTION: each attempt generates
#    COMPOSITION_CANDIDATES_PER_ATTEMPT fresh candidates in parallel and keeps the
#    highest-scoring one, retrying up to COMPOSITION_MAX_ATTEMPTS times until one
#    passes. (No whole-image refine edits — those diverged in practice.) ──
COMPOSITION_VALIDATION_ENABLED = os.getenv(
    "COMPOSITION_VALIDATION_ENABLED", "true"
).lower() in ("true", "1", "yes")
COMPOSITION_MAX_ATTEMPTS = int(os.getenv("COMPOSITION_MAX_ATTEMPTS", "2"))
COMPOSITION_CANDIDATES_PER_ATTEMPT = int(os.getenv("COMPOSITION_CANDIDATES_PER_ATTEMPT", "2"))
COMPOSITION_SCORE_THRESHOLD = float(os.getenv("COMPOSITION_SCORE_THRESHOLD", "0.80"))


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ProductValidation:
    product_id: str
    position_correct: bool = True
    size_correct: bool = True
    rotation_correct: bool = True
    notes: str = ""


@dataclass
class ValidationResult:
    score: float = 1.0
    passed: bool = True
    product_checks: List[ProductValidation] = field(default_factory=list)
    missing_items: int = 0
    extra_items: int = 0
    room_shape_correct: bool = True
    openings_correct: bool = True   # used by wall validation; always True for floor
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── OpenAI API helper ────────────────────────────────────────────────────────

def _openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("FALLBACK_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY in environment.")
    return key


def _image_to_data_url(image_bytes: bytes) -> str:
    """Convert raw image bytes to a base64 data URL for the Vision API."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    # Detect format from magic bytes
    if image_bytes.startswith(b"\x89PNG"):
        mime = "image/png"
    elif image_bytes.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{b64}"


# ── Validation prompt ────────────────────────────────────────────────────────

def _build_validation_prompt(payload: dict) -> str:
    """Build the system/user prompt that asks GPT-4o to score spatial accuracy."""
    products = payload.get("products", [])
    room = payload.get("room", {})
    room_w = float(room.get("width") or 0)
    room_l = float(room.get("length") or 0)

    # Detect polygon room
    room_polygon = room.get("polygon")
    is_polygon = (
        room_polygon
        and isinstance(room_polygon, list)
        and len(room_polygon) >= 3
    )
    n_vertices = len(room_polygon) if is_polygon else 4

    product_specs = []
    for p in products:
        pid = p.get("id", "?")
        pos = p.get("position", {})
        dims = p.get("dimensions", {})
        rot = int(p.get("rotation_y") or 0) % 360
        cx = float(pos.get("x", 0))
        cy = float(pos.get("y", 0))
        x_pct = round(cx / room_w * 100, 1) if room_w else 0
        y_pct = round(cy / room_l * 100, 1) if room_l else 0
        pw = float(dims.get("width") or 0)
        pd = float(dims.get("depth") or 0)
        eff_w, eff_d = (pd, pw) if rot in (90, 270) else (pw, pd)
        gap_left = round(cx - eff_w / 2, 2)
        gap_right = round(room_w - cx - eff_w / 2, 2)
        gap_top = round(cy - eff_d / 2, 2)
        gap_bottom = round(room_l - cy - eff_d / 2, 2)
        product_specs.append(
            f"  - id={pid}: center at ({x_pct}% from left, {y_pct}% from top), "
            f"size {pw}x{pd}m, rotation {rot} deg. "
            f"Expected gaps: left={gap_left}m, right={gap_right}m, top={gap_top}m, bottom={gap_bottom}m"
        )

    products_block = "\n".join(product_specs)

    # Room description with polygon awareness
    if is_polygon:
        room_desc = (
            f"Room: {room_w:.1f}m wide x {room_l:.1f}m deep bounding box "
            f"(irregular polygon with {n_vertices} vertices — NOT rectangular)."
        )
        shape_check = (
            "- Does the room shape match the POLYGON outline from the guide? "
            "The room is NOT rectangular — verify it follows the polygon shape, not just proportions."
        )
        gap_note = (
            "- For this polygon-shaped room, check that furniture positions relative to the "
            "polygon walls match the guide, not just N/S/E/W gaps."
        )
    else:
        room_desc = f"Room: {room_w:.1f}m wide x {room_l:.1f}m deep."
        shape_check = "- Does the room shape match (walls, proportions)?"
        gap_note = (
            "- Are gaps between furniture and walls preserved? "
            "If a product has a 2m gap to a wall in the guide, the render must show approximately the same gap."
        )

    return f"""You are a strict spatial accuracy judge for interior design renders.

IMAGE 1 is a 2D floor plan guide showing exact furniture positions as numbered gray rectangles.
IMAGE 2 is a generated 3D isometric render that should match those positions.

{room_desc}

Expected product positions (percentages from room edges):
{products_block}

For each product, assess STRICTLY:
1. POSITION: Is the product center within 10% of its expected location? Pay special attention to gaps between furniture and walls/windows — if the guide shows a large gap, the render must show the same proportional gap.
2. SIZE: Is the product approximately the right relative size? (within 25% of expected)
3. ROTATION: Does the product face the correct direction?

Also check:
- Are there any MISSING products (in guide but not in render)?
- Are there any EXTRA products (in render but not in guide)? Count carefully — staging props from reference photos should NOT appear.
{shape_check}
{gap_note}

Respond with ONLY valid JSON in this exact format:
{{
  "products": [
    {{
      "product_id": "the_id",
      "position_correct": true/false,
      "size_correct": true/false,
      "rotation_correct": true/false,
      "notes": "brief description of any issues"
    }}
  ],
  "missing_items": 0,
  "extra_items": 0,
  "room_shape_correct": true/false,
  "overall_notes": "brief summary"
}}"""


# ── Core validation function ─────────────────────────────────────────────────

def validate_spatial_accuracy(
    generated_bytes: bytes,
    guide_bytes: bytes,
    payload: dict,
) -> ValidationResult:
    """Send guide + generated image to GPT-4o Vision for spatial comparison.

    On any failure (API error, unparseable response), returns a passing result
    with score=0.5 so generation is never blocked by validation errors.
    """
    try:
        prompt = _build_validation_prompt(payload)

        guide_url = _image_to_data_url(guide_bytes)
        gen_url = _image_to_data_url(generated_bytes)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": guide_url, "detail": "high"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": gen_url, "detail": "high"},
                    },
                ],
            }
        ]

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_openai_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": VALIDATION_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1500,
            },
            timeout=VALIDATION_TIMEOUT,
        )

        if not response.ok:
            logger.warning(
                "Validation API error %s: %s — returning pass-through result",
                response.status_code, response.text[:500],
            )
            return ValidationResult(score=0.5, passed=True, error=f"API error {response.status_code}")

        raw_text = response.json()["choices"][0]["message"]["content"]
        return _parse_validation_response(raw_text, payload)

    except Exception as exc:
        logger.warning("Validation failed (%s) — returning pass-through result", exc)
        return ValidationResult(score=0.5, passed=True, error=str(exc))


def _parse_validation_response(raw_text: str, payload: dict) -> ValidationResult:
    """Parse GPT-4o JSON response into a ValidationResult with computed score."""
    # Extract JSON from response (handle markdown code blocks)
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove ```json ... ``` wrapper
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse validation JSON — returning pass-through")
        return ValidationResult(score=0.5, passed=True, error="Unparseable JSON response")

    # Build product checks
    product_checks = []
    for p_data in data.get("products", []):
        product_checks.append(ProductValidation(
            product_id=p_data.get("product_id", "?"),
            position_correct=p_data.get("position_correct", True),
            size_correct=p_data.get("size_correct", True),
            rotation_correct=p_data.get("rotation_correct", True),
            notes=p_data.get("notes", ""),
        ))

    missing = int(data.get("missing_items", 0))
    extra = int(data.get("extra_items", 0))
    room_ok = data.get("room_shape_correct", True)

    # Compute score: start at 1.0, deduct per failure
    score = 1.0
    for pc in product_checks:
        if not pc.position_correct:
            score -= 0.15
        if not pc.size_correct:
            score -= 0.10
        if not pc.rotation_correct:
            score -= 0.05
    score -= missing * 0.20
    score -= extra * 0.05
    if not room_ok:
        score -= 0.15
    score = max(0.0, min(1.0, score))

    passed = score >= VALIDATION_SCORE_THRESHOLD

    return ValidationResult(
        score=round(score, 3),
        passed=passed,
        product_checks=product_checks,
        missing_items=missing,
        extra_items=extra,
        room_shape_correct=room_ok,
    )


# ── Correction notes for retry prompts ────────────────────────────────────────

def build_correction_notes(validation: ValidationResult) -> str:
    """Convert validation failures into prompt correction text for retries."""
    if validation.passed:
        return ""

    lines = ["CORRECTIONS REQUIRED (previous attempt had spatial errors):"]

    for pc in validation.product_checks:
        issues = []
        if not pc.position_correct:
            issues.append("WRONG POSITION")
        if not pc.size_correct:
            issues.append("WRONG SIZE")
        if not pc.rotation_correct:
            issues.append("WRONG ROTATION")
        if issues:
            detail = f" — {pc.notes}" if pc.notes else ""
            lines.append(
                f"  CORRECTION: {pc.product_id}: {', '.join(issues)}.{detail}"
            )

    if validation.missing_items > 0:
        lines.append(
            f"  MISSING: {validation.missing_items} product(s) from the guide are not visible in the render. "
            f"ALL products must appear."
        )
    if validation.extra_items > 0:
        lines.append(
            f"  EXTRA: {validation.extra_items} unexpected item(s) appeared in the render. "
            f"Remove all items not in the guide."
        )
    if not validation.room_shape_correct:
        lines.append(
            "  ROOM SHAPE: The room proportions or wall layout don't match the guide. "
            "Maintain exact room dimensions and wall positions."
        )

    lines.append(
        "\nFix ALL issues above. Match every product's position, size, and rotation "
        "to the guide image EXACTLY. Do not add or remove any items."
    )

    return "\n".join(lines)


# ── Candidate ranking ────────────────────────────────────────────────────────

def pick_best_candidate(
    candidates: List[Tuple[bytes, ValidationResult]],
) -> Tuple[bytes, ValidationResult]:
    """Rank candidates by overall score, return the highest-scoring one."""
    if not candidates:
        raise ValueError("No candidates to pick from")
    return max(candidates, key=lambda c: c[1].score)


# ── Wall elevation validation ─────────────────────────────────────────────────

def _build_wall_validation_prompt(payload: dict) -> str:
    """Build the GPT-4o prompt to score spatial accuracy of a wall elevation render."""
    products = payload.get("products", [])
    wall = payload.get("wall", {})
    wall_w = float(wall.get("width") or 0)
    wall_h = float(wall.get("height") or 4.0)
    openings = payload.get("openings", [])

    product_specs = []
    for p in products:
        pid = p.get("id", "?")
        pos = p.get("position", {})
        dims = p.get("dimensions", {})
        cx = float(pos.get("x", 0))       # centre from left edge
        cy = float(pos.get("y", 0))        # centre height from floor
        dw = float(dims.get("width") or 0)
        dh = float(dims.get("height") or dims.get("depth") or 0)
        x_pct = round(cx / wall_w * 100, 1) if wall_w else 0
        y_floor_pct = round(cy / wall_h * 100, 1) if wall_h else 0
        w_pct = round(dw / wall_w * 100, 1) if wall_w else 0
        h_pct = round(dh / wall_h * 100, 1) if wall_h else 0
        gap_left = round(cx - dw / 2, 2)
        gap_right = round(wall_w - cx - dw / 2, 2)
        product_specs.append(
            f"  - id={pid}: centre at ({x_pct}% from left, {y_floor_pct}% from floor), "
            f"size {dw:.1f}×{dh:.1f}m ({w_pct}% of wall width × {h_pct}% of wall height). "
            f"Gaps from wall edges: left={gap_left:.2f}m, right={gap_right:.2f}m"
        )

    products_block = "\n".join(product_specs)

    openings_block = ""
    if openings:
        specs = []
        for o in openings:
            otype = (o.get("type") or "opening").lower()
            pos_left = float(o.get("position_from_left") or 0)
            ow = float(o.get("width") or 0)
            if otype == "door":
                specs.append(f"  - door at x={pos_left:.2f}m from left, {ow:.1f}m wide, at floor level")
            else:
                sill = float(o.get("sill_height") or 1.2)
                oh = float(o.get("height") or 0.9)
                specs.append(
                    f"  - window at x={pos_left:.2f}m from left, {ow:.1f}m wide × {oh:.1f}m tall, "
                    f"sill at {sill:.1f}m from floor"
                )
        openings_block = "\nExpected openings (must be visible in the render):\n" + "\n".join(specs)

    openings_check = (
        f"\n- Are all {len(openings)} opening(s) (doors/windows) clearly visible at the correct "
        f"positions on the wall?" if openings else ""
    )

    return f"""You are a strict spatial accuracy judge for interior design wall elevation renders.

IMAGE 1 is a 2D wall elevation guide showing exact product positions as numbered gray rectangles.
IMAGE 2 is a generated front-facing wall elevation render that should match IMAGE 1.

Wall dimensions: {wall_w:.1f}m wide × {wall_h:.1f}m tall.

Expected product positions:
{products_block}{openings_block}

For each product, assess STRICTLY:
1. POSITION: Is the product centre within 10% of its expected left-right position on the wall \
AND at the correct height from the floor? Pay attention to gaps between products and wall edges.
2. SIZE: Is the product approximately the right relative size compared to the wall? (within 25% of expected)

Also check:
- Are there any MISSING products (visible in guide but absent from render)?
- Are there any EXTRA products or props (present in render but not in guide)?{openings_check}

Respond with ONLY valid JSON in this exact format:
{{
  "products": [
    {{
      "product_id": "the_id",
      "position_correct": true/false,
      "size_correct": true/false,
      "rotation_correct": true,
      "notes": "brief description of any issues"
    }}
  ],
  "missing_items": 0,
  "extra_items": 0,
  "openings_correct": true/false,
  "room_shape_correct": true/false,
  "overall_notes": "brief summary"
}}"""


def _parse_wall_validation_response(raw_text: str, payload: dict) -> ValidationResult:
    """Parse GPT-4o JSON response for a wall validation into a ValidationResult."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse wall validation JSON — returning pass-through")
        return ValidationResult(score=0.5, passed=True, error="Unparseable JSON response")

    product_checks = []
    for p_data in data.get("products", []):
        product_checks.append(ProductValidation(
            product_id=p_data.get("product_id", "?"),
            position_correct=p_data.get("position_correct", True),
            size_correct=p_data.get("size_correct", True),
            rotation_correct=True,   # not applicable for wall elevation
            notes=p_data.get("notes", ""),
        ))

    missing = int(data.get("missing_items", 0))
    extra = int(data.get("extra_items", 0))
    openings_ok = bool(data.get("openings_correct", True))
    wall_ok = bool(data.get("room_shape_correct", True))

    score = 1.0
    for pc in product_checks:
        if not pc.position_correct:
            score -= 0.15
        if not pc.size_correct:
            score -= 0.10
    score -= missing * 0.20
    score -= extra * 0.05
    if not openings_ok:
        score -= 0.10
    if not wall_ok:
        score -= 0.10
    score = max(0.0, min(1.0, score))

    passed = score >= VALIDATION_SCORE_THRESHOLD

    return ValidationResult(
        score=round(score, 3),
        passed=passed,
        product_checks=product_checks,
        missing_items=missing,
        extra_items=extra,
        room_shape_correct=wall_ok,
        openings_correct=openings_ok,
    )


def validate_wall_accuracy(
    generated_bytes: bytes,
    guide_bytes: bytes,
    payload: dict,
) -> ValidationResult:
    """Send wall guide + generated elevation to GPT-4o Vision for spatial comparison.

    Mirrors validate_spatial_accuracy but uses wall-specific prompt and scoring.
    On any failure returns score=0.5/passed=True so generation is never blocked.
    """
    try:
        prompt = _build_wall_validation_prompt(payload)
        guide_url = _image_to_data_url(guide_bytes)
        gen_url = _image_to_data_url(generated_bytes)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": guide_url, "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": gen_url, "detail": "high"}},
                ],
            }
        ]

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_openai_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": VALIDATION_MODEL,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1500,
            },
            timeout=VALIDATION_TIMEOUT,
        )

        if not response.ok:
            logger.warning(
                "Wall validation API error %s: %s — returning pass-through result",
                response.status_code, response.text[:500],
            )
            return ValidationResult(score=0.5, passed=True, error=f"API error {response.status_code}")

        raw_text = response.json()["choices"][0]["message"]["content"]
        return _parse_wall_validation_response(raw_text, payload)

    except Exception as exc:
        logger.warning("Wall validation failed (%s) — returning pass-through result", exc)
        return ValidationResult(score=0.5, passed=True, error=str(exc))


# ── Composition (dollhouse) validation ───────────────────────────────────────
#
# The composite is the only image users see, so it gets its own image-to-image
# check: the generated dollhouse view is compared against the SAME inputs it was
# assembled from — the floor render and each visible wall's elevation. The model
# verifies that every wall's content survived faithfully (same object TYPES, no
# art↔window swaps), openings are preserved, nothing migrated between walls, and
# the floor furniture is intact.

def _describe_wall_openings_for_qa(compass: str, ops: Optional[List[Dict]]) -> str:
    C = compass.upper()
    if not ops:
        return f"  - {C} wall: MUST have ZERO windows and ZERO doors (solid wall)."
    counts: Dict[str, int] = {}
    for o in ops:
        t = (o.get("type") or "opening").lower()
        counts[t] = counts.get(t, 0) + 1
    spec = ", ".join(f"{n} {t}{'s' if n > 1 else ''}" for t, n in counts.items())
    return f"  - {C} wall: MUST have exactly {spec} — no more, no fewer, none relocated."


def _build_composition_validation_prompt(
    designed_walls: List[str],
    visible_walls: List[str],
    openings_by_wall: Optional[Dict[str, List[Dict]]],
    view: str,
) -> str:
    """Prompt for scoring a dollhouse composite against its source images + ground truth.

    The image order sent to the model is:
      IMAGE 1 = the GENERATED composite (to be judged)
      IMAGE 2 = the floor render (furniture ground truth)
      IMAGE 3.. = each DESIGNED wall elevation, in `designed_walls` order
    `openings_by_wall` is the AUTHORITATIVE per-wall opening spec for ALL visible walls.
    """
    ob = openings_by_wall or {}

    wall_lines = []
    for i, compass in enumerate(designed_walls):
        wall_lines.append(
            f"  - IMAGE {i + 3} = the {compass.upper()} wall elevation. In the composite, the "
            f"{compass.upper()} wall MUST show exactly these wall-mounted items (same TYPES — a "
            f"framed picture/art stays a picture, never a window; never swapped), the same count, "
            f"the same heights, and the SAME LEFT-TO-RIGHT ORDER as the elevation — NOT mirrored "
            f"or flipped."
        )
    walls_block = "\n".join(wall_lines) if wall_lines else "  (No designed walls with elevations in this view.)"

    openings_block = "\n".join(
        _describe_wall_openings_for_qa(c, ob.get(c)) for c in visible_walls
    ) or "  (No opening data provided.)"

    wall_ids = ", ".join(w.upper() for w in visible_walls) or "(none)"

    return f"""You are a strict QA inspector for an interior-design composite render.

You are given several images:
- IMAGE 1 = a generated open-box "dollhouse" render of a room (the {view.upper()} view) — THIS is what you judge.
- IMAGE 2 = the floor render: the ground truth for floor furniture (identity, count, positions).
{walls_block}

## AUTHORITATIVE openings (the single source of truth — trust this over any image)
For EACH visible wall, the composite must show exactly these openings and nothing else:
{openings_block}

Carefully COUNT the windows and doors on each wall in IMAGE 1 and compare to the list above.
If a wall is marked "ZERO windows and ZERO doors" but IMAGE 1 shows ANY window or door on it,
that is a FAILURE (set that wall's openings_correct=false and count each invented opening in
extra_items). Invented windows on solid walls are the single most important defect to catch.

Judge whether IMAGE 1 faithfully ASSEMBLES the sources. Check, per visible wall ({wall_ids}):
1. CONTENT TYPE: every wall-mounted item from that wall's elevation appears on that SAME wall in
   IMAGE 1, as the SAME type of object. A framed picture rendered as a window (or vice-versa) is a
   FAILURE. A dropped/missing wall item is a FAILURE.
2. POSITION/SIZE: each item is at roughly the right spot along the wall and the right relative size.
3. OPENINGS: windows/doors EXACTLY match the authoritative list above — no extra, missing, or relocated.
4. LEFT-RIGHT ORDER (MIRRORING): the items across the wall must appear in the SAME left-to-right
   order as in that wall's elevation — NOT reversed. Use any door or window on the wall as an
   anchor: if the elevation shows an item to the RIGHT of the door, it must STILL be to the right
   of the door in IMAGE 1 (and likewise for left). If the wall's contents are mirrored / flipped
   (order reversed, or an item that was on the right of an opening is now on its left), that wall
   is a FAILURE — set its "left_right_correct" to false. This is a common, important defect.

Also check globally:
- MIGRATION: no item from one wall appears on a different wall, the floor, or the ceiling.
- FLOOR: the furniture from IMAGE 2 is present, with the same count and roughly the same layout.
- INVENTED/EXTRA: nothing was invented that is absent from all the sources / the authoritative list.

Respond with ONLY valid JSON in this exact format:
{{
  "walls": [
    {{"wall": "north", "content_matches": true, "openings_correct": true, "left_right_correct": true, "notes": "..."}}
  ],
  "migrated_items": 0,
  "missing_items": 0,
  "extra_items": 0,
  "floor_furniture_correct": true,
  "overall_notes": "brief summary of the most important problems, if any"
}}"""


def _parse_composition_validation_response(raw_text: str) -> ValidationResult:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse composition validation JSON — returning pass-through")
        return ValidationResult(score=0.5, passed=True, error="Unparseable JSON response")

    product_checks: List[ProductValidation] = []
    walls_ok = True
    openings_ok = True
    mirrored_walls = 0
    for w in data.get("walls", []):
        content_ok = bool(w.get("content_matches", True))
        opening_ok = bool(w.get("openings_correct", True))
        lr_ok = bool(w.get("left_right_correct", True))
        if not lr_ok:
            mirrored_walls += 1
        walls_ok = walls_ok and content_ok and lr_ok
        openings_ok = openings_ok and opening_ok
        product_checks.append(ProductValidation(
            product_id=str(w.get("wall", "?")),
            # A mirrored wall is a position failure (items are on the wrong side).
            position_correct=content_ok and lr_ok,
            size_correct=content_ok,
            rotation_correct=lr_ok,
            notes=w.get("notes", ""),
        ))

    migrated = int(data.get("migrated_items", 0))
    missing = int(data.get("missing_items", 0))
    extra = int(data.get("extra_items", 0))
    floor_ok = bool(data.get("floor_furniture_correct", True))

    # Score: migrations and missing items are the worst failures (they're exactly
    # the dislocation/drop problems we are trying to eliminate). A mirrored/flipped
    # wall is penalised heavily too, so best-of-N selection rejects flipped candidates.
    score = 1.0
    for pc in product_checks:
        if not pc.size_correct:           # content/type mismatch
            score -= 0.20
    score -= mirrored_walls * 0.30
    score -= migrated * 0.25
    score -= missing * 0.20
    score -= extra * 0.10
    if not openings_ok:
        score -= 0.15
    if not floor_ok:
        score -= 0.15
    score = max(0.0, min(1.0, score))

    return ValidationResult(
        score=round(score, 3),
        passed=score >= COMPOSITION_SCORE_THRESHOLD,
        product_checks=product_checks,
        missing_items=missing,
        # fold migrations into extra_items so existing logging stays meaningful
        extra_items=extra + migrated,
        room_shape_correct=floor_ok,
        openings_correct=openings_ok,
    )


def validate_composition_accuracy(
    composite_bytes: bytes,
    floor_bytes: bytes,
    wall_items: List[Tuple[str, bytes]],
    view: str,
    visible_walls: Optional[List[str]] = None,
    openings_by_wall: Optional[Dict[str, List[Dict]]] = None,
) -> ValidationResult:
    """Compare a dollhouse composite against the floor render + each wall elevation,
    using the authoritative per-wall openings as ground truth.

    wall_items:       (compass, elevation_bytes) for DESIGNED walls in this view (have images).
    visible_walls:    ALL walls visible in this view (designed or not) — for opening checks.
    openings_by_wall: authoritative {compass: [opening,...]} ground truth.
    On any failure returns score=0.5/passed=True so composition is never blocked.
    """
    try:
        designed_walls = [c for c, _ in wall_items]
        all_visible = visible_walls or designed_walls
        prompt = _build_composition_validation_prompt(
            designed_walls, all_visible, openings_by_wall, view
        )

        content: List[dict] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(composite_bytes), "detail": "high"}},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(floor_bytes), "detail": "high"}},
        ]
        for _compass, elev_bytes in wall_items:
            content.append(
                {"type": "image_url", "image_url": {"url": _image_to_data_url(elev_bytes), "detail": "high"}}
            )

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_openai_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": VALIDATION_MODEL,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "max_tokens": 1500,
            },
            timeout=VALIDATION_TIMEOUT,
        )

        if not response.ok:
            logger.warning(
                "Composition validation API error %s: %s — returning pass-through result",
                response.status_code, response.text[:500],
            )
            return ValidationResult(score=0.5, passed=True, error=f"API error {response.status_code}")

        raw_text = response.json()["choices"][0]["message"]["content"]
        return _parse_composition_validation_response(raw_text)

    except Exception as exc:
        logger.warning("Composition validation failed (%s) — returning pass-through result", exc)
        return ValidationResult(score=0.5, passed=True, error=str(exc))


def build_composition_correction_notes(validation: ValidationResult) -> str:
    """Turn composition validation failures into corrective text for a retry.

    Returns a per-wall + global instruction block to append to the add-wall prompts.
    """
    if validation.passed:
        return ""

    lines = ["## CORRECTIONS FROM THE PREVIOUS ATTEMPT (fix these exactly)"]

    for pc in validation.product_checks:
        if not pc.position_correct:
            detail = f" — {pc.notes}" if pc.notes else ""
            lines.append(
                f"- The {pc.product_id.upper()} wall did not match its elevation.{detail} "
                f"Reproduce that wall's items with the correct object TYPE, count, position, and size."
            )

    if validation.missing_items > 0:
        lines.append(
            f"- {validation.missing_items} item(s) from the source images are MISSING. "
            f"Every wall item and every piece of floor furniture must appear."
        )
    if validation.extra_items > 0:
        lines.append(
            f"- {validation.extra_items} item(s) were INVENTED or MIGRATED to the wrong wall. "
            f"Remove anything not present in that wall's own elevation; keep each item on its own wall."
        )
    if not validation.openings_correct:
        lines.append(
            "- Door/window openings are wrong (extra, missing, or relocated). "
            "Keep exactly the openings shown for each wall — do not add or move any."
        )
    if not validation.room_shape_correct:
        lines.append(
            "- The floor furniture changed. Keep all furniture exactly as in the floor render."
        )

    return "\n".join(lines)


# ── Cross-view consistency validation ────────────────────────────────────────
# Verifies that the front and back composite views depict the SAME room.

CROSS_VIEW_VALIDATION_ENABLED = os.getenv(
    "CROSS_VIEW_VALIDATION_ENABLED", "true"
).lower() in ("true", "1", "yes")
CROSS_VIEW_SCORE_THRESHOLD = float(os.getenv("CROSS_VIEW_SCORE_THRESHOLD", "0.70"))


def _build_cross_view_prompt(
    expected_furniture_count: int,
    shared_walls: List[str],
) -> str:
    """Prompt for GPT-4o to compare front and back views of the same room."""
    shared = ", ".join(w.upper() for w in shared_walls) or "none"
    return f"""You are a QA inspector comparing TWO renders of the EXACT SAME room from opposite camera angles.

IMAGE 1 = Front view (camera at South-East corner, looking toward North-West).
  Visible walls: North (back), East (right side), West (left side). South wall removed.

IMAGE 2 = Back/opposite view (camera at North-West corner, looking toward South-East).
  Visible walls: South (back), East (left side), West (right side). North wall removed.

IMAGE 3 = Floor plan (ground truth for furniture count and positions).

These MUST depict the EXACT same room. Only the camera position changes.

Expected furniture count from the floor plan: {expected_furniture_count} item(s).

SHARED WALLS visible in BOTH views: {shared}.
These walls MUST show the SAME content (same windows, doors, decorations) in both views.

## CHECKS

1. FURNITURE COUNT
   Count every distinct piece of furniture on the floor in IMAGE 1 and IMAGE 2.
   Both counts should equal {expected_furniture_count} (from the floor plan).

2. SHARED WALLS ({shared})
   For each shared wall, check: same windows/doors, same wall-mounted items, same type of objects.

3. FURNITURE IDENTITY
   Same types of furniture in both views (if one has a sofa, the other must too).
   No object-type substitutions (side table turned into a chair, etc.).

4. FLOOR CONSISTENCY
   Same flooring material and pattern in both views.

5. EXTRA / MISSING OBJECTS
   Neither view should contain furniture or wall decor absent from the other.

Respond with ONLY valid JSON:
{{
  "furniture_count_front": <int>,
  "furniture_count_back": <int>,
  "counts_match": <bool>,
  "shared_walls": [
    {{"wall": "east", "consistent": <bool>, "notes": "..."}},
    {{"wall": "west", "consistent": <bool>, "notes": "..."}}
  ],
  "furniture_types_match": <bool>,
  "floor_consistent": <bool>,
  "extra_objects_front": <int>,
  "extra_objects_back": <int>,
  "overall_consistent": <bool>,
  "notes": "<brief summary of problems, if any>"
}}"""


def _parse_cross_view_response(raw_text: str) -> ValidationResult:
    """Parse the GPT-4o cross-view comparison into a ValidationResult."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse cross-view validation JSON — returning pass-through")
        return ValidationResult(score=0.5, passed=True, error="Unparseable JSON response")

    counts_match = bool(data.get("counts_match", True))
    furniture_match = bool(data.get("furniture_types_match", True))
    floor_ok = bool(data.get("floor_consistent", True))
    overall = bool(data.get("overall_consistent", True))
    extra_front = int(data.get("extra_objects_front", 0))
    extra_back = int(data.get("extra_objects_back", 0))

    shared_walls_ok = True
    shared_wall_checks: List[ProductValidation] = []
    for sw in data.get("shared_walls", []):
        ok = bool(sw.get("consistent", True))
        shared_walls_ok = shared_walls_ok and ok
        shared_wall_checks.append(ProductValidation(
            product_id=str(sw.get("wall", "?")),
            position_correct=ok,
            size_correct=ok,
            notes=sw.get("notes", ""),
        ))

    # Scoring: furniture count mismatch is the most critical failure.
    score = 1.0
    if not counts_match:
        score -= 0.30
    if not shared_walls_ok:
        for sw in data.get("shared_walls", []):
            if not bool(sw.get("consistent", True)):
                score -= 0.15
    if not furniture_match:
        score -= 0.15
    if not floor_ok:
        score -= 0.05
    score -= extra_front * 0.08
    score -= extra_back * 0.08
    score = max(0.0, min(1.0, score))

    return ValidationResult(
        score=round(score, 3),
        passed=score >= CROSS_VIEW_SCORE_THRESHOLD,
        product_checks=shared_wall_checks,
        missing_items=abs(int(data.get("furniture_count_front", 0)) - int(data.get("furniture_count_back", 0))),
        extra_items=extra_front + extra_back,
        room_shape_correct=floor_ok,
        openings_correct=shared_walls_ok,
    )


def validate_cross_view_consistency(
    front_bytes: bytes,
    back_bytes: bytes,
    floor_bytes: bytes,
    expected_furniture_count: int = 0,
    shared_walls: Optional[List[str]] = None,
) -> ValidationResult:
    """Compare front and back composite views to verify they depict the same room.

    Returns a ValidationResult with score and pass/fail. On any API error,
    returns score=0.5/passed=True so the pipeline is never blocked.
    """
    if not CROSS_VIEW_VALIDATION_ENABLED:
        return ValidationResult(score=1.0, passed=True)

    try:
        prompt = _build_cross_view_prompt(
            expected_furniture_count=expected_furniture_count,
            shared_walls=shared_walls or ["east", "west"],
        )

        content: List[dict] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(front_bytes), "detail": "high"}},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(back_bytes), "detail": "high"}},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(floor_bytes), "detail": "high"}},
        ]

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_openai_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": VALIDATION_MODEL,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.1,
                "max_tokens": 1500,
            },
            timeout=VALIDATION_TIMEOUT,
        )

        if not response.ok:
            logger.warning(
                "Cross-view validation API error %s: %s — returning pass-through",
                response.status_code, response.text[:500],
            )
            return ValidationResult(score=0.5, passed=True, error=f"API error {response.status_code}")

        raw_text = response.json()["choices"][0]["message"]["content"]
        result = _parse_cross_view_response(raw_text)
        logger.info(
            "Cross-view validation: score=%.3f passed=%s missing=%d extra=%d",
            result.score, result.passed, result.missing_items, result.extra_items,
        )
        return result

    except Exception as exc:
        logger.warning("Cross-view validation failed (%s) — returning pass-through", exc)
        return ValidationResult(score=0.5, passed=True, error=str(exc))
