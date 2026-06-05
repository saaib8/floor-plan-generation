"""
Which product categories can be placed on floor vs wall surfaces.

Category slugs match core_product.category in the Mesaky database.
"""

from typing import Literal, Optional

SurfaceType = Literal["floor", "wall"]

# Wall-mounted or wall-elevation items (art, shelves, clocks, wall lights, etc.)
WALL_CATEGORIES: frozenset[str] = frozenset({
    "art-canvas",
    "decorative-hanger",
    "shelve",
    "wall-clock",
    "wall-lighting",
})

# Floor-standing furniture and floor accessories
FLOOR_CATEGORIES: frozenset[str] = frozenset({
    "2-seater-sofa",
    "3-seater-sofa",
    "bed",
    "bedspread",
    "candle",
    "carpet",
    "center-table",
    "chair",
    "chaise-lounge",
    "coffee-maker",
    "comforter",
    "console",
    "cooking-appliance",
    "cooking-pot",
    "cup",
    "dining-table",
    "dressing-table",
    "floor-stand",
    "flower",
    "flower-pot-and-plant",
    "food-processor",
    "l-shape-sofa",
    "lampshade",
    "laundry-basket",
    "mattress-pad",
    "mattresses",
    "office-chair",
    "office-table",
    "pillow",
    "plate",
    "service-table",
    "serving-utensil-and-tray",
    "side-table",
    "sofa",
    "statue-and-antique",
    "storage-box",
    "tv-table",
    "vase",
    "wardrobe",
})


def normalize_surface(surface: Optional[str]) -> Optional[SurfaceType]:
    if not surface:
        return None
    value = surface.strip().lower()
    if value in ("floor", "wall"):
        return value  # type: ignore[return-value]
    return None


def categories_for_surface(surface: Optional[str]) -> Optional[frozenset[str]]:
    """Return allowed category slugs for a surface, or None if surface is not set."""
    normalized = normalize_surface(surface)
    if normalized == "wall":
        return WALL_CATEGORIES
    if normalized == "floor":
        return FLOOR_CATEGORIES
    return None


def category_allowed_on_surface(category: Optional[str], surface: Optional[str]) -> bool:
    allowed = categories_for_surface(surface)
    if allowed is None:
        return True
    slug = (category or "").strip().lower()
    if not slug or slug == "uncategorized":
        return False
    return slug in allowed
