from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ProductItem(BaseModel):
    product_id: Optional[int] = None
    image_url: str
    hex_color: str
    dims: Optional[str] = None
    dimensions: Optional[str] = None
    rotation: Optional[int] = 0


class GenerationRequest(BaseModel):
    highlight_image: str          # base64 data URI  e.g. "data:image/png;base64,..."
    products: List[ProductItem]
    presets: Optional[Dict[str, Any]] = None
    room_dimensions: Optional[Dict[str, Any]] = None
    type: str = "floor"           # "floor" | "wall"


class GenerationStartResponse(BaseModel):
    gen_id: str
    status: str


class GenerationStatusResponse(BaseModel):
    gen_id: str
    status: str                   # "processing" | "success" | "failed"
    result_image: Optional[str] = None
    reason: Optional[str] = None
