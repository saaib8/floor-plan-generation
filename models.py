from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ProductItem(BaseModel):
    product_id: Optional[int] = None
    category: Optional[str] = None
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
    result_images: Optional[Dict[str, str]] = None  # all views: {isometric, front, corner}
    reason: Optional[str] = None
    validation_metrics: Optional[Dict[str, Any]] = None  # spatial validation stats when enabled


class FloorPlanRequest(BaseModel):
    scene_id: Optional[str] = None
    unit: str = "cm"
    room: Dict[str, Any]
    openings: Optional[List[Dict[str, Any]]] = []
    products: Optional[List[Dict[str, Any]]] = []
    lighting: Optional[Dict[str, Any]] = None
    camera_views: Optional[List[Dict[str, Any]]] = []
    size: str = "1024x1024"       # "1024x1024" | "1536x1024" | "1024x1536"
