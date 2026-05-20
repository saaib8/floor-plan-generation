---
name: generation-flow
description: Current gpt-image-2 generation architecture — prompt separation, image inputs, and floor vs wall vs Floor Plan JSON flows
metadata:
  type: project
---

## Core principle (established 2026-05-19)

**STRUCTURED DATA = geometry. ENGLISH PROMPT = appearance.**

Never put dimensions, coordinates, positions, or layout in the English prompt. Never put materials, lighting, or mood in the JSON. The model reads JSON for geometry and English for how it should look.

**Why:** Mixing geometry into English prose causes gpt-image-2 to "imagine" plausible-looking rooms instead of following exact specs. A guide image (2D floor plan PNG) was tried and also caused noise — the model edited the 2D flat image instead of generating a 3D isometric render.

---

## Floor Plan JSON flow (`POST /api/floor-plan`)

**Input:** Structured JSON payload with `unit`, `room`, `openings`, `products` (each with `id`, `dimensions`, `position`, `rotation_y`, `image_url`), `lighting`, `camera_views`.

**Images sent to gpt-image-2:**
- IMAGE 1 = blank white canvas (neutral, not a guide)
- IMAGE 2, 3, … = product reference photos (downloaded from `image_url`)

**Prompt structure (`build_floor_plan_prompt`, `has_guide=False`):**
1. Short English paragraph — appearance only: lighting, flooring material, wall color, product fabric/finish, render quality
2. Camera/view line
3. Product image mapping: `IMAGE 2 = product id=X`, `IMAGE 3 = product id=Y`
4. Raw JSON geometry block — all of `room`, `openings`, `products` (positions, dimensions, rotations), `camera_views`; material/category/image_url stripped
5. Hard constraints: exact product count, no extra decor, match reference images, no text overlays

**Key function:** `generate_floor_plan(payload, size)` in `generation.py`

---

## Surface Editor floor flow (`POST /api/generate`, `type: "floor"`)

Same architecture as Floor Plan JSON. The canvas-drawn guide blobs are NOT sent.

**Images sent:**
- IMAGE 1 = blank white canvas
- IMAGE 2, 3, … = product reference photos

**Prompt:** identical structure via `build_floor_plan_prompt(payload, image_order, has_guide=False)`.  
Payload built by `_batch_to_fp_payload()` from Surface Editor batch items (hex_color, dims, x_m, y_m, rotation).

---

## Surface Editor wall flow (`POST /api/generate`, `type: "wall"`)

Still uses guide blobs — the wall flow needs them to mask exact pixel regions on the base image.

**Images sent:**
- IMAGE 1 = current base (wall render being built up in batches)
- IMAGE 2 = guide blobs (colored rectangles showing where products go on the wall)
- IMAGE 3, 4, … = product reference photos

**Prompt:** `build_wall_placement_prompt()` — front-elevation focused, includes scale/position text.

---

## What was tried and rejected

- **Guide image (2D PIL floor plan) as IMAGE 1**: drew room walls + product footprints to scale as PNG. Rejected — model treated it as source to edit rather than transforming to 3D isometric. Produced flat 2D output with colored rectangles.
- **Verbose prose geometry in prompt**: percentages like "18% of room width × 31% of room length ≈ 184×322 px". Rejected — unreliable, model ignored or approximated.
- **Product name in prompt**: explicitly prohibited by user — do not send product names/categories in any prompt.
