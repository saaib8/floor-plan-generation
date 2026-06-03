from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ProductItem(BaseModel):
    product_id: Optional[int] = None
    category: Optional[str] = None
    product_name: Optional[str] = None  # human-readable name e.g. "Work Desk", "Double Bed"
    image_url: str
    hex_color: str
    dims: Optional[str] = None
    dimensions: Optional[Any] = None   # dict {width, depth} in metres, or legacy string
    rotation: Optional[int] = 0
    x_m: Optional[float] = None   # center X position in metres within the surface
    y_m: Optional[float] = None   # center Y position in metres within the surface


class GenerationRequest(BaseModel):
    highlight_image: str          # base64 data URI  e.g. "data:image/png;base64,..."
    products: List[ProductItem]
    presets: Optional[Dict[str, Any]] = None
    room_dimensions: Optional[Dict[str, Any]] = None
    type: str = "floor"           # "floor" | "wall"
    openings: Optional[List[Dict[str, Any]]] = []


class GenerationStartResponse(BaseModel):
    gen_id: str
    status: str


class GenerationStatusResponse(BaseModel):
    gen_id: str
    status: str                          # "processing" | "success" | "failed"
    result_image: Optional[str] = None  # primary (isometric) image path
    result_images: Optional[Dict[str, str]] = None  # views: {isometric} or {elevation}
    reason: Optional[str] = None
    validation_metrics: Optional[Dict[str, Any]] = None  # spatial validation stats when enabled


class WallImageUrl(BaseModel):
    url: str                           # server-relative path, e.g. "/outputs/xxx_elevation.png"
    wall_id: Optional[int] = None
    label: Optional[str] = None
    compass: Optional[str] = None      # "north"/"south"/"east"/"west" — used for placement
    width_m: Optional[float] = None
    height_m: Optional[float] = None


class ComposeRequest(BaseModel):
    floor_image_url: str               # server-relative path, e.g. "/outputs/xxx_isometric.png"
    wall_image_urls: List[WallImageUrl]
    room_dimensions: Optional[Dict[str, Any]] = None
    presets: Optional[Dict[str, Any]] = None
    # Authoritative architectural openings per wall (ground truth for which walls have
    # windows/doors). Each item: {compass, type, position_from_left(0..1), width_m, sill_height}.
    openings: Optional[List[Dict[str, Any]]] = []


class FloorPlanRequest(BaseModel):
    scene_id: Optional[str] = None
    unit: str = "cm"
    room: Dict[str, Any]
    openings: Optional[List[Dict[str, Any]]] = []
    products: Optional[List[Dict[str, Any]]] = []
    lighting: Optional[Dict[str, Any]] = None
    camera_views: Optional[List[Dict[str, Any]]] = []
    size: str = "1024x1024"       # "1024x1024" | "1536x1024" | "1024x1536"
