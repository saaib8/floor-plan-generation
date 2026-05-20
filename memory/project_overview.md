---
name: project-overview
description: Core architecture and data flow of the Walls & Floor Generation app (Zory)
metadata:
  type: project
---

Interior design visualization tool — user draws a room, places furniture, AI generates a photorealistic render.

**Why:** Zory product for visualizing walls & floor layouts with real furniture.

**How to apply:** Understand the full pipeline before suggesting changes — each layer is tightly coupled.

## Stack
- Backend: FastAPI + ThreadPoolExecutor (async generation jobs stored in-memory dict)
- AI: OpenAI `gpt-image-2` via `/v1/images/edits` multipart API
- Frontend: Vanilla JS, two `<canvas>` elements, no framework

## Data flow
1. User draws polygon room on **Canvas 1** (click corners, double-click to close)
2. User clicks a wall segment or the floor area → **Canvas 2** shows that surface
3. User drags products from sidebar onto Canvas 2 — each gets a unique hex color from `PRODUCT_COLORS[]`
4. "Generate" exports a **highlight image** (1024×1024 PNG): white surface + solid-color rectangles per product
5. Frontend POSTs to `/api/generate` with: highlight_image (base64), products list (hex_color, image_url, dims, rotation), room_dimensions, type (`floor`|`wall`)
6. Backend saves the highlight PNG, spawns a thread running `generate_product_placement()`
7. Generation pipeline:
   - Downloads base + guide images (both are the same highlight image)
   - Builds color masks (NumPy pixel matching per hex, connected components, min_area filter)
   - Builds prompt via `prompt.py` (floor → top-down isometric; wall → front elevation)
   - Calls OpenAI edit API with: [current_base, guide_image, ...product_images]
   - Iterates in batches of 20 products
8. Frontend polls `/api/generate/{gen_id}` every 2s; shows result image on success

## Key files
- `main.py` — FastAPI routes + async job runner
- `generation.py` — image download, color masking, OpenAI API call
- `prompt.py` — prompt builders for floor and wall modes
- `models.py` — Pydantic request/response models
- `products/catalog.json` — 10 furniture items with SVG icons and meter dimensions
- `static/app.js` — all frontend logic (~820 lines, single file)

## Notable design choices
- Guide image = highlight image (same file); color masks tell AI where to place each product
- `lock_all_except_color_regions()` exists in generation.py but is NOT currently called in the pipeline
- Generation state is in-memory (`_generations` dict) — lost on server restart
- Product colors assigned from a fixed 20-color palette (`PRODUCT_COLORS` in app.js)
