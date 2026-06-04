from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


# ── Scene Graph ──────────────────────────────────────────────────────────────
# Canonical room representation used by both composite views. Every view is
# derived from the SAME scene — only the camera position changes.

_COMPASS_SWAP = {"north": "south", "south": "north", "east": "west", "west": "east"}


@dataclass
class RoomScene:
    """Canonical room data. Single source of truth for all composite views."""
    room_width_m: float
    room_depth_m: float
    wall_height_m: float
    floor_image: bytes
    walls: Dict[str, bytes]                   # compass → elevation image bytes
    openings: Dict[str, List[Dict]]           # compass → opening specs
    floor_products: List[Dict]                # [{product_name, x_m, y_m, rotation, dimensions}]
    wall_color: str = "#F3EFE8"

    def camera_view(self, removed_wall: str) -> "CameraViewData":
        """Derive a camera-specific view from this canonical scene.

        For the front view (remove south), data passes through unchanged.
        For the back view (remove north), all coordinates and compass labels
        are transformed by 180° so the model can use its front-view priors.
        """
        is_front = removed_wall.lower() == "south"

        if is_front:
            return CameraViewData(
                removed_wall="south",
                prompt_removed_wall="south",
                visible_walls=["north", "east", "west"],
                wall_images={
                    "north": self.walls.get("north"),
                    "south": None,   # removed
                    "east": self.walls.get("east"),
                    "west": self.walls.get("west"),
                },
                floor_products=list(self.floor_products),
                openings_by_wall=dict(self.openings),
                rotate_floor=False,
            )

        # Back view: 180° transform — swap compass labels so the model
        # can use the standard front-view prompt (remove "south").
        swapped_walls = {
            "north": self.walls.get("south"),    # South → "North" slot (back wall)
            "south": None,                        # "South" = removed
            "east":  self.walls.get("west"),      # West → "East" slot (right side)
            "west":  self.walls.get("east"),      # East → "West" slot (left side)
        }

        swapped_openings: Dict[str, List[Dict]] = {}
        for compass, ops in self.openings.items():
            swapped_openings[_COMPASS_SWAP.get(compass, compass)] = ops

        flipped_products = []
        for fp in self.floor_products:
            item = dict(fp)
            item["x_m"] = self.room_width_m - float(item.get("x_m") or 0)
            item["y_m"] = self.room_depth_m - float(item.get("y_m") or 0)
            item["rotation"] = (int(item.get("rotation") or 0) + 180) % 360
            flipped_products.append(item)

        return CameraViewData(
            removed_wall="north",
            prompt_removed_wall="south",   # trick: use front-view prompt
            visible_walls=["south", "east", "west"],
            wall_images=swapped_walls,
            floor_products=flipped_products,
            openings_by_wall=swapped_openings,
            rotate_floor=True,
        )


@dataclass
class CameraViewData:
    """Transformed room data for a specific camera position."""
    removed_wall: str                         # real compass of removed wall
    prompt_removed_wall: str                  # compass used in the prompt template
    visible_walls: List[str]                  # real compass labels of visible walls
    wall_images: Dict[str, Optional[bytes]]   # prompt compass → image bytes (None=blank)
    floor_products: List[Dict]                # transformed coordinates
    openings_by_wall: Dict[str, List[Dict]]   # prompt compass → opening specs
    rotate_floor: bool                        # whether the floor image needs 180° rotation


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
    # Floor-placed furniture with explicit positions, rotations, dimensions.
    # Each item: {product_name, x_m, y_m, rotation, dimensions: {width, depth}}.
    floor_products: Optional[List[Dict[str, Any]]] = None


class FloorPlanRequest(BaseModel):
    scene_id: Optional[str] = None
    unit: str = "cm"
    room: Dict[str, Any]
    openings: Optional[List[Dict[str, Any]]] = []
    products: Optional[List[Dict[str, Any]]] = []
    lighting: Optional[Dict[str, Any]] = None
    camera_views: Optional[List[Dict[str, Any]]] = []
    size: str = "1024x1024"       # "1024x1024" | "1536x1024" | "1024x1536"
