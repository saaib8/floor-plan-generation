import asyncio
import base64
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from generation import generate_floor_plan, generate_product_placement, generate_room_composition
from models import (
    ComposeRequest, FloorPlanRequest, GenerationRequest,
    GenerationStartResponse, GenerationStatusResponse,
)

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parents[1]

# Reuse Mesaky backend environment for DB/S3, while letting this app's local
# .env override OpenAI/Gemini/runtime values.
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=True)

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mesaky_backend.settings")

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
_django_initialized = False
_s3_icon_key_cache: dict = {"expires_at": 0, "keys": set()}


def _ensure_django():
    """Initialize Django lazily so this app reuses Mesaky DB/settings."""
    global _django_initialized
    if _django_initialized:
        return

    import django

    django.setup()
    _django_initialized = True


def _product_model():
    _ensure_django()
    from core.models import Product

    return Product


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

    _ensure_django()
    from core.utils.s3_helper import s3

    bucket = _icon_bucket()
    keys: set[str] = set()
    paginator = s3.get_paginator("list_objects_v2")

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


def _product_to_payload(product) -> dict:
    length_m = _to_meters(product.length, product.dimension_unit)
    width_m = _to_meters(product.width, product.dimension_unit)
    dimensions = None
    if length_m and width_m:
        dimensions = {
            "width": width_m,
            "depth": length_m,
        }

    product_id = product.id
    return {
        "id": product_id,
        "name": product.name_english or f"Product {product_id}",
        "category": _category_value(product.category),
        "store_id": product.store_id,
        "store_name": getattr(product.store, "name_english", "") or "",
        "icon": f"/api/products/{product_id}/icon",
        "two_d_icon": product.two_d_icon or "",
        "image_url": product.image_url or "",
        "dims": _format_dims(product.length, product.width, product.dimension_unit),
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
    limit: int = Query(default=200, ge=1, le=500),
):
    try:
        Product = _product_model()
        existing_icon_keys = _existing_s3_icon_keys()
        if not existing_icon_keys:
            return []

        qs = (
            Product.objects.filter(
                is_active=True,
                two_d_icon__isnull=False,
                two_d_icon__in=existing_icon_keys,
            )
            .exclude(two_d_icon="")
            .select_related("store")
            .order_by("name_english")
        )

        if category:
            if category == "uncategorized":
                from django.db.models import Q

                qs = qs.filter(Q(category__isnull=True) | Q(category=""))
            else:
                qs = qs.filter(category__iexact=category)
        if search:
            qs = qs.filter(name_english__icontains=search)
        if store_id:
            qs = qs.filter(store_id=store_id)

        return [_product_to_payload(product) for product in qs[:limit]]
    except Exception as exc:
        logger.error("Failed to load DB products: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load DB products: {exc}")


@app.get("/api/categories")
def get_categories():
    try:
        from django.db.models import Count

        Product = _product_model()
        existing_icon_keys = _existing_s3_icon_keys()
        if not existing_icon_keys:
            return []

        rows = (
            Product.objects.filter(
                is_active=True,
                two_d_icon__isnull=False,
                two_d_icon__in=existing_icon_keys,
            )
            .exclude(two_d_icon="")
            .values("category")
            .annotate(count=Count("id"))
        )
        counts = {}
        for row in rows:
            category = _category_value(row["category"])
            counts[category] = counts.get(category, 0) + row["count"]

        return [
            {"category": category, "count": count}
            for category, count in sorted(counts.items(), key=lambda item: item[0].lower())
        ]
    except Exception as exc:
        logger.error("Failed to load DB categories: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load DB categories: {exc}")


@app.get("/api/products/{product_id}/icon")
def get_product_icon(product_id: int):
    Product = _product_model()
    product = (
        Product.objects.filter(
            id=product_id,
            two_d_icon__isnull=False,
        )
        .exclude(two_d_icon="")
        .only("two_d_icon")
        .first()
    )
    if not product:
        raise HTTPException(status_code=404, detail="Product icon not found")

    key = product.two_d_icon
    if key.startswith("http://") or key.startswith("https://"):
        return RedirectResponse(key)
    if key not in _existing_s3_icon_keys():
        raise HTTPException(status_code=404, detail="Product icon key does not exist in S3")

    try:
        from core.utils.s3_helper import s3

        obj = s3.get_object(Bucket=_icon_bucket(), Key=key)
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

    # Decode and save the highlight image from base64
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
    """Build a validation_metrics dict from the attempt metrics list."""
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
        result_bytes = generate_room_composition(
            floor_image_url=payload["floor_image_url"],
            wall_image_urls=[dict(wi) for wi in payload["wall_image_urls"]],
            room_dimensions=payload.get("room_dimensions"),
            presets=payload.get("presets"),
            base_dir=str(BASE_DIR),
        )
        output_path = OUTPUT_DIR / f"{gen_id}_composition.png"
        output_path.write_bytes(result_bytes)
        url = f"/outputs/{gen_id}_composition.png"
        _generations[gen_id]["status"] = "success"
        _generations[gen_id]["result_image"] = url
        _generations[gen_id]["result_images"] = {"isometric": url}
        logger.info("Composition %s completed", gen_id)
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
