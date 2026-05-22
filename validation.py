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

    return f"""You are a strict spatial accuracy judge for interior design renders.

IMAGE 1 is a 2D floor plan guide showing exact furniture positions as numbered gray rectangles.
IMAGE 2 is a generated 3D isometric render that should match those positions.

Room: {room_w:.1f}m wide x {room_l:.1f}m deep.

Expected product positions (percentages from room edges):
{products_block}

For each product, assess STRICTLY:
1. POSITION: Is the product center within 10% of its expected location? Pay special attention to gaps between furniture and walls/windows — if the guide shows a large gap, the render must show the same proportional gap.
2. SIZE: Is the product approximately the right relative size? (within 25% of expected)
3. ROTATION: Does the product face the correct direction?

Also check:
- Are there any MISSING products (in guide but not in render)?
- Are there any EXTRA products (in render but not in guide)? Count carefully — staging props from reference photos should NOT appear.
- Does the room shape match (walls, proportions)?
- Are gaps between furniture and walls preserved? If a product has a 2m gap to a wall in the guide, the render must show approximately the same gap.

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
