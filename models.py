from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


# ── Scene Graph ──────────────────────────────────────────────────────────────
# Canonical room representation used by both composite views. Every view is
# derived from the SAME scene — only the camera position changes.


# ── SceneObject — unified registry entry for any room object ─────────────────

@dataclass
class SceneObject:
    """A single object in the room: floor furniture, wall art, or opening."""
    id: str              # "F1", "W_N1", "OPEN_E_WIN1"
    name: str            # "Work Desk", "Abstract Canvas", "Window"
    obj_type: str        # "floor_furniture" | "wall_art" | "window" | "door"
    binding: str         # "floor" | "north" | "south" | "east" | "west"
    # Floor items
    x_m: Optional[float] = None
    y_m: Optional[float] = None
    rotation: int = 0
    width_m: float = 0.0
    depth_m: float = 0.0
    # Wall items
    x_along_wall_m: Optional[float] = None
    y_from_floor_m: Optional[float] = None
    height_m: float = 0.0
    # Metadata
    category: str = ""
    product_id: Optional[int] = None


# ── ViewManifest — what a specific camera view must show ─────────────────────

@dataclass
class ViewManifest:
    """Authoritative manifest for a single camera view (front or back)."""
    view_name: str                                    # "front" | "back"
    removed_wall: str
    visible_walls: List[str] = field(default_factory=list)
    floor_objects: List[SceneObject] = field(default_factory=list)
    wall_objects_by_wall: Dict[str, List[SceneObject]] = field(default_factory=dict)
    openings_by_wall: Dict[str, List[SceneObject]] = field(default_factory=dict)
    must_appear: List[str] = field(default_factory=list)
    must_not_appear: List[str] = field(default_factory=list)
    total_floor_count: int = 0
    total_wall_art_count: int = 0
    total_openings_count: int = 0
    wall_summaries: Dict[str, str] = field(default_factory=dict)
    room_width_m: float = 0.0
    room_depth_m: float = 0.0
    wall_height_m: float = 0.0


# ── RoomScene — canonical room data ─────────────────────────────────────────

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
    wall_products: Optional[List[Dict]] = None
    _object_registry: Optional[List[SceneObject]] = field(default=None, repr=False)

    def build_object_registry(self) -> List[SceneObject]:
        """Build a unified object registry with stable IDs.

        Assigns IDs:
        - F1, F2, ... for floor furniture
        - W_N1, W_N2, ... for wall art per compass
        - OPEN_E_WIN1, OPEN_N_DOOR1, ... for openings
        Caches the result so repeated calls return the same list.
        """
        if self._object_registry is not None:
            return self._object_registry

        registry: List[SceneObject] = []

        # Floor products → F1, F2, ...
        for idx, fp in enumerate(self.floor_products):
            dims = fp.get("dimensions") or {}
            registry.append(SceneObject(
                id=f"F{idx + 1}",
                name=fp.get("product_name") or "item",
                obj_type="floor_furniture",
                binding="floor",
                x_m=float(fp.get("x_m") or 0),
                y_m=float(fp.get("y_m") or 0),
                rotation=int(fp.get("rotation") or 0) % 360,
                width_m=float(dims.get("width") or 0),
                depth_m=float(dims.get("depth") or dims.get("height") or 0),
                category=fp.get("category") or "",
            ))

        # Wall products → W_N1, W_N2, ... per compass
        wall_counters: Dict[str, int] = {}
        for wp in (self.wall_products or []):
            compass = (wp.get("wall_compass") or "").lower()
            if compass not in ("north", "south", "east", "west"):
                continue
            key = compass[0].upper()
            wall_counters[key] = wall_counters.get(key, 0) + 1
            registry.append(SceneObject(
                id=f"W_{key}{wall_counters[key]}",
                name=wp.get("product_name") or "wall item",
                obj_type="wall_art",
                binding=compass,
                x_along_wall_m=float(wp.get("x_m") or 0),
                y_from_floor_m=float(wp.get("y_from_floor_m") or 0),
                width_m=float(wp.get("width_m") or 0),
                height_m=float(wp.get("height_m") or 0),
                category=wp.get("category") or "",
            ))

        # Openings → OPEN_E_WIN1, OPEN_N_DOOR1, ...
        opening_counters: Dict[str, int] = {}
        for compass, ops in self.openings.items():
            c_key = compass[0].upper()
            for op in ops:
                otype = (op.get("type") or "opening").lower()
                type_key = "WIN" if "win" in otype else "DOOR" if "door" in otype else "OPEN"
                counter_key = f"{c_key}_{type_key}"
                opening_counters[counter_key] = opening_counters.get(counter_key, 0) + 1
                registry.append(SceneObject(
                    id=f"OPEN_{c_key}_{type_key}{opening_counters[counter_key]}",
                    name=otype.capitalize(),
                    obj_type=otype if otype in ("window", "door") else "opening",
                    binding=compass,
                    x_along_wall_m=float(op.get("position_from_left") or 0),
                    width_m=float(op.get("width_m") or op.get("width") or 0),
                    height_m=float(op.get("height") or (0.9 if "win" in otype else 2.1)),
                    y_from_floor_m=float(op.get("sill_height") or (1.2 if "win" in otype else 0)),
                ))

        self._object_registry = registry
        return registry

    def view_manifest(self, removed_wall: str) -> ViewManifest:
        """Build a ViewManifest for a specific camera view.

        removed_wall: "south" for front view, "north" for back view.
        """
        registry = self.build_object_registry()
        removed = removed_wall.lower()
        is_front = removed == "south"
        view_name = "front" if is_front else "back"

        visible_walls = [c for c in ("north", "south", "east", "west") if c != removed]

        floor_objects: List[SceneObject] = []
        wall_objects_by_wall: Dict[str, List[SceneObject]] = {c: [] for c in visible_walls}
        openings_by_wall_so: Dict[str, List[SceneObject]] = {c: [] for c in visible_walls}
        must_appear: List[str] = []
        must_not_appear: List[str] = []

        for obj in registry:
            if obj.obj_type == "floor_furniture":
                # All floor objects are visible in every view
                floor_objects.append(obj)
                must_appear.append(obj.id)
            elif obj.obj_type == "wall_art":
                if obj.binding == removed:
                    must_not_appear.append(obj.id)
                elif obj.binding in wall_objects_by_wall:
                    wall_objects_by_wall[obj.binding].append(obj)
                    must_appear.append(obj.id)
            elif obj.obj_type in ("window", "door", "opening"):
                if obj.binding == removed:
                    must_not_appear.append(obj.id)
                elif obj.binding in openings_by_wall_so:
                    openings_by_wall_so[obj.binding].append(obj)
                    must_appear.append(obj.id)

        # Build per-wall summary strings
        wall_summaries: Dict[str, str] = {}
        for c in visible_walls:
            parts = []
            wall_arts = wall_objects_by_wall.get(c, [])
            wall_opens = openings_by_wall_so.get(c, [])
            if wall_arts:
                parts.append(f"{len(wall_arts)} wall item(s)")
            if wall_opens:
                parts.append(f"{len(wall_opens)} opening(s)")
            if not parts:
                parts.append("empty (bare wall)")
            wall_summaries[c] = f"{c.upper()} wall: {', '.join(parts)}"

        total_wall_art = sum(len(v) for v in wall_objects_by_wall.values())
        total_openings = sum(len(v) for v in openings_by_wall_so.values())

        return ViewManifest(
            view_name=view_name,
            removed_wall=removed,
            visible_walls=visible_walls,
            floor_objects=floor_objects,
            wall_objects_by_wall=wall_objects_by_wall,
            openings_by_wall=openings_by_wall_so,
            must_appear=must_appear,
            must_not_appear=must_not_appear,
            total_floor_count=len(floor_objects),
            total_wall_art_count=total_wall_art,
            total_openings_count=total_openings,
            wall_summaries=wall_summaries,
            room_width_m=self.room_width_m,
            room_depth_m=self.room_depth_m,
            wall_height_m=self.wall_height_m,
        )

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
    # Wall-mounted products with positions relative to their wall.
    # Each item: {product_name, wall_compass, x_m, y_from_floor_m, width_m, height_m, category}.
    wall_products: Optional[List[Dict[str, Any]]] = None


class FloorPlanRequest(BaseModel):
    scene_id: Optional[str] = None
    unit: str = "cm"
    room: Dict[str, Any]
    openings: Optional[List[Dict[str, Any]]] = []
    products: Optional[List[Dict[str, Any]]] = []
    lighting: Optional[Dict[str, Any]] = None
    camera_views: Optional[List[Dict[str, Any]]] = []
    size: str = "1024x1024"       # "1024x1024" | "1536x1024" | "1024x1536"
