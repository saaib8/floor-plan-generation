import asyncio
import base64
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from generation import generate_floor_plan, generate_product_placement, generate_room_composition
from models import (
    ComposeRequest, FloorPlanRequest, GenerationRequest,
    GenerationStartResponse, GenerationStatusResponse,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Walls & Floor Generation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
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


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/api/products")
def get_products():
    catalog_path = PRODUCTS_DIR / "catalog.json"
    if not catalog_path.exists():
        return []
    with open(catalog_path) as f:
        return json.load(f)


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
