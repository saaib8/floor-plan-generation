import asyncio
import base64
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing the local modules below: several of them read env vars at IMPORT
# time into module-level constants (e.g. COMPOSITION_BACKEND, COMPOSITION_VALIDATION_ENABLED in
# generation/validation). If .env is loaded after those imports, the constants silently fall back
# to their defaults and .env overrides are ignored.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI, HTTPException, Query, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from db import get_categories as db_get_categories  # noqa: E402
from db import get_product_two_d_icon, get_products as db_get_products  # noqa: E402
from placement_categories import normalize_surface  # noqa: E402
from generation import generate_floor_plan, generate_product_placement, generate_room_composition  # noqa: E402
from models import (  # noqa: E402
    ComposeRequest, FloorPlanRequest, GenerationRequest,
    GenerationStartResponse, GenerationStatusResponse,
)
from s3_client import s3  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Walls & Floor Generation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PRODUCTS_DIR = BASE_DIR / "products"
OUTPUT_DIR = BASE_DIR / "outputs"
TEST_IMAGES_DIR = BASE_DIR / "test_images"
OUTPUT_DIR.mkdir(exist_ok=True)
TEST_IMAGES_DIR.mkdir(exist_ok=True)

app.mount("/products", StaticFiles(directory=str(PRODUCTS_DIR)), name="products")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/test_images", StaticFiles(directory=str(TEST_IMAGES_DIR)), name="test_images")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# In-memory generation state  {gen_id: {...}}
_generations: dict = {}
_executor = ThreadPoolExecutor(max_workers=4)
_s3_icon_key_cache: dict = {"expires_at": 0, "keys": set()}


def _icon_bucket() -> str:
    return (
        os.getenv("TWO_D_ICON_BUCKET")
        or os.getenv("AWS_TEMPORARY_BUCKET_NAME")
        or os.getenv("AWS_PUBLIC_BUCKET_NAME")
        or "zory-temporary-uploads-backup"
    )


def _icon_prefixes() -> list[str]:
    raw = os.getenv("TWO_D_ICON_PREFIXES") or os.getenv("TWO_D_ICON_PREFIX") or "2D_icons"
    return [prefix.strip("/ ") for prefix in raw.split(",") if prefix.strip("/ ")]


def _existing_s3_icon_keys() -> set[str]:
    now = time.time()
    if _s3_icon_key_cache["expires_at"] > now:
        return _s3_icon_key_cache["keys"]

    bucket = _icon_bucket()
    keys: set[str] = set()
    paginator = s3().get_paginator("list_objects_v2")

    for prefix in _icon_prefixes():
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/")
        for page in page_iterator:
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if key and not key.endswith("/"):
                    keys.add(key)

    _s3_icon_key_cache["keys"] = keys
    _s3_icon_key_cache["expires_at"] = now + int(os.getenv("TWO_D_ICON_S3_CACHE_SECONDS", "300"))
    return keys


def _to_float(value):
    return float(value) if value is not None else None


def _to_meters(value, unit: str | None):
    number = _to_float(value)
    if number is None:
        return None
    normalized = (unit or "").strip().lower()
    if normalized in {"cm", "centimeter", "centimeters"}:
        return number / 100
    if normalized in {"mm", "millimeter", "millimeters"}:
        return number / 1000
    # Most catalog data is in cm; guard against huge metre values when unit is missing.
    if number > 20:
        return number / 100
    return number


def _format_dims(length, width, unit):
    length_f = _to_float(length)
    width_f = _to_float(width)
    if length_f is None or width_f is None:
        return ""
    suffix = unit or ""
    return f"{length_f:g} x {width_f:g} {suffix}".strip()


def _category_value(category: str | None) -> str:
    return (category or "").strip() or "uncategorized"


def _row_to_payload(row: dict) -> dict:
    length_m = _to_meters(row.get("length"), row.get("dimension_unit"))
    width_m = _to_meters(row.get("width"), row.get("dimension_unit"))
    dimensions = None
    if length_m and width_m:
        # length = product's long dimension (along the wall, canvas X-axis)
        # width  = product's short dimension (depth from the wall, canvas Y-axis)
        dimensions = {
            "width": length_m,
            "depth": width_m,
        }

    product_id = row["id"]
    return {
        "id": product_id,
        "name": row.get("name_english") or f"Product {product_id}",
        "category": _category_value(row.get("category")),
        "store_id": row.get("store_id"),
        "store_name": row.get("store_name") or "",
        "icon": f"/api/products/{product_id}/icon",
        "two_d_icon": row.get("two_d_icon") or "",
        "image_url": row.get("image_url") or "",
        "dims": _format_dims(row.get("length"), row.get("width"), row.get("dimension_unit")),
        "dimensions": dimensions,
    }


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/api/products")
def get_products(
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
    store_id: int | None = Query(default=None),
    surface: str | None = Query(default=None, description="'floor' or 'wall'"),
    limit: int = Query(default=200, ge=1, le=500),
):
    try:
        if surface and not normalize_surface(surface):
            raise HTTPException(status_code=400, detail="surface must be 'floor' or 'wall'")
        existing_icon_keys = _existing_s3_icon_keys()
        rows = db_get_products(
            allowed_keys=existing_icon_keys,
            category=category,
            search=search,
            store_id=store_id,
            limit=limit,
            surface=surface,
        )
        return [_row_to_payload(row) for row in rows]
    except Exception as exc:
        logger.error("Failed to load DB products: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load DB products: {exc}")


@app.get("/api/categories")
def get_categories(
    surface: str | None = Query(default=None, description="'floor' or 'wall'"),
):
    try:
        if surface and not normalize_surface(surface):
            raise HTTPException(status_code=400, detail="surface must be 'floor' or 'wall'")
        existing_icon_keys = _existing_s3_icon_keys()
        rows = db_get_categories(allowed_keys=existing_icon_keys, surface=surface)

        counts: dict[str, int] = {}
        for row in rows:
            cat = _category_value(row.get("category"))
            counts[cat] = counts.get(cat, 0) + int(row.get("count", 0))

        return [
            {"category": cat, "count": count}
            for cat, count in sorted(counts.items(), key=lambda item: item[0].lower())
        ]
    except Exception as exc:
        logger.error("Failed to load DB categories: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load DB categories: {exc}")


@app.get("/api/products/{product_id}/icon")
def get_product_icon(product_id: int):
    key = get_product_two_d_icon(product_id)
    if not key:
        raise HTTPException(status_code=404, detail="Product icon not found")

    if key.startswith("http://") or key.startswith("https://"):
        return RedirectResponse(key)
    if key not in _existing_s3_icon_keys():
        raise HTTPException(status_code=404, detail="Product icon key does not exist in S3")

    try:
        obj = s3().get_object(Bucket=_icon_bucket(), Key=key)
        body = obj["Body"].read()
        content_type = obj.get("ContentType") or "image/svg+xml"
        if key.lower().endswith(".svg"):
            content_type = "image/svg+xml"
    except Exception as exc:
        logger.error("Failed to load product icon %s from S3 key %s: %s", product_id, key, exc, exc_info=True)
        raise HTTPException(status_code=404, detail=f"Failed to load product icon from S3: {exc}")

    return Response(
        content=body,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/generate", response_model=GenerationStartResponse)
async def start_generation(request: GenerationRequest):
    if not request.products:
        raise HTTPException(status_code=400, detail="products list is empty")
    if len(request.products) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 products allowed")
    if request.type not in ("floor", "wall"):
        raise HTTPException(status_code=400, detail="type must be 'floor' or 'wall'")

    gen_id = str(uuid.uuid4())

    try:
        raw_b64 = request.highlight_image.split(",")[-1]
        img_data = base64.b64decode(raw_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid highlight_image base64")

    highlight_path = OUTPUT_DIR / f"{gen_id}_highlight.png"
    with open(highlight_path, "wb") as f:
        f.write(img_data)

    _generations[gen_id] = {
        "gen_id": gen_id,
        "status": "processing",
        "result_image": None,
        "reason": None,
    }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_generation,
        gen_id,
        str(highlight_path),
        [p.model_dump() for p in request.products],
        request.presets,
        request.room_dimensions,
        request.type,
        request.openings or [],
    )

    return GenerationStartResponse(gen_id=gen_id, status="processing")


@app.get("/api/generate/{gen_id}", response_model=GenerationStatusResponse)
def get_generation(gen_id: str):
    gen = _generations.get(gen_id)
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    return GenerationStatusResponse(**gen)


@app.post("/api/floor-plan", response_model=GenerationStartResponse)
async def start_floor_plan_generation(request: FloorPlanRequest):
    gen_id = str(uuid.uuid4())
    _generations[gen_id] = {
        "gen_id": gen_id,
        "status": "processing",
        "result_image": None,
        "reason": None,
    }
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_floor_plan_generation,
        gen_id,
        request.model_dump(),
    )
    return GenerationStartResponse(gen_id=gen_id, status="processing")


def _save_view_results(gen_id: str, results: dict) -> dict:
    """Save each view image and return {view_name: url_path}."""
    paths = {}
    for view_name, img_bytes in results.items():
        output_path = OUTPUT_DIR / f"{gen_id}_{view_name}.png"
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        paths[view_name] = f"/outputs/{gen_id}_{view_name}.png"
    return paths


def _build_validation_metrics(attempt_metrics: list) -> dict:
    if not attempt_metrics:
        return {"enabled": False}
    final = attempt_metrics[-1]
    return {
        "enabled": True,
        "total_attempts": len(attempt_metrics),
        "final_score": final.get("best_score", 0),
        "passed": final.get("passed", False),
        "attempts": attempt_metrics,
    }


def _run_floor_plan_generation(gen_id: str, payload: dict):
    size = payload.pop("size", "1024x1024")
    try:
        results, attempt_metrics = generate_floor_plan(payload, size=size)
        paths = _save_view_results(gen_id, results)
        _generations[gen_id]["status"] = "success"
        _generations[gen_id]["result_image"] = paths.get("isometric") or next(iter(paths.values()))
        _generations[gen_id]["result_images"] = paths
        if attempt_metrics:
            _generations[gen_id]["validation_metrics"] = _build_validation_metrics(attempt_metrics)
        logger.info("Floor plan generation %s completed: %s", gen_id, list(paths.keys()))
    except Exception as e:
        logger.error("Floor plan generation %s failed: %s", gen_id, e, exc_info=True)
        _generations[gen_id]["status"] = "failed"
        _generations[gen_id]["reason"] = str(e)


def _run_generation(gen_id, highlight_path, products, presets, room_dimensions, generation_type, openings=None):
    try:
        view_dict, attempt_metrics = generate_product_placement(
            products=products,
            room_dimensions=room_dimensions,
            presets=presets,
            generation_type=generation_type,
            openings=openings or [],
        )
        paths = _save_view_results(gen_id, view_dict)
        primary = paths.get("isometric") or paths.get("elevation") or next(iter(paths.values()))
        _generations[gen_id]["result_image"] = primary
        _generations[gen_id]["result_images"] = paths
        if attempt_metrics:
            _generations[gen_id]["validation_metrics"] = _build_validation_metrics(attempt_metrics)
        _generations[gen_id]["status"] = "success"
        logger.info("Generation %s completed successfully", gen_id)
    except Exception as e:
        logger.error("Generation %s failed: %s", gen_id, e, exc_info=True)
        _generations[gen_id]["status"] = "failed"
        _generations[gen_id]["reason"] = str(e)


@app.post("/api/compose", response_model=GenerationStartResponse)
async def start_composition(request: ComposeRequest):
    if not request.wall_image_urls:
        raise HTTPException(status_code=400, detail="wall_image_urls is empty")

    gen_id = str(uuid.uuid4())
    _generations[gen_id] = {
        "gen_id": gen_id,
        "status": "processing",
        "result_image": None,
        "reason": None,
    }
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _run_composition,
        gen_id,
        request.model_dump(),
    )
    return GenerationStartResponse(gen_id=gen_id, status="processing")


def _run_composition(gen_id: str, payload: dict):
    try:
        results = generate_room_composition(
            floor_image_url=payload["floor_image_url"],
            wall_image_urls=[dict(wi) for wi in payload["wall_image_urls"]],
            room_dimensions=payload.get("room_dimensions"),
            presets=payload.get("presets"),
            openings=payload.get("openings") or [],
            base_dir=str(BASE_DIR),
        )
        paths: dict = {}
        for corner_name, img_bytes in results.items():
            out_path = OUTPUT_DIR / f"{gen_id}_composition_{corner_name}.png"
            out_path.write_bytes(img_bytes)
            paths[f"composition_{corner_name}"] = f"/outputs/{gen_id}_composition_{corner_name}.png"

        # SW corner is the primary (standard hero view)
        primary = paths.get("composition_front") or next(iter(paths.values()))
        _generations[gen_id]["status"] = "success"
        _generations[gen_id]["result_image"] = primary
        _generations[gen_id]["result_images"] = paths
        logger.info("Composition %s completed: %s", gen_id, list(paths.keys()))
    except Exception as e:
        logger.error("Composition %s failed: %s", gen_id, e, exc_info=True)
        _generations[gen_id]["status"] = "failed"
        _generations[gen_id]["reason"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
