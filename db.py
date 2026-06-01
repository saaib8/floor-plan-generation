"""
Standalone database access layer for the floor-plan-generation app.

Replaces the previous Django ORM + core.models.Product dependency with
direct psycopg2 queries against the same PostgreSQL database.

Connection is configured via environment variables:
  DATABASE_URL  — full DSN, e.g. postgresql://user:pass@host:5432/dbname
  or the individual vars DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.
"""

import logging
import os
from contextlib import contextmanager
from typing import Optional

import psycopg2
import psycopg2.extras

from placement_categories import categories_for_surface

logger = logging.getLogger(__name__)


def _dsn() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "")
    user = os.getenv("DB_USER", "")
    password = os.getenv("DB_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


@contextmanager
def _get_conn():
    conn = psycopg2.connect(_dsn())
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def _surface_category_clause(surface: Optional[str]) -> tuple[str, list]:
    """SQL fragment + params restricting products to floor or wall categories."""
    allowed = categories_for_surface(surface)
    if not allowed:
        return "", []
    return " AND lower(p.category) = ANY(%s)", [sorted(allowed)]


def get_products(
    allowed_keys: set[str],
    category: Optional[str] = None,
    search: Optional[str] = None,
    store_id: Optional[int] = None,
    limit: int = 200,
    surface: Optional[str] = None,
) -> list[dict]:
    """Return a list of product dicts matching the filters.

    Mirrors the Django queryset in main.py:
        Product.objects.filter(is_active=True, two_d_icon__in=allowed_keys)
    """
    if not allowed_keys:
        return []

    sql = """
        SELECT
            p.id,
            p.name_english,
            p.category,
            p.store_id,
            p.two_d_icon,
            p.image_url,
            p.length,
            p.width,
            p.dimension_unit,
            s.name_english AS store_name
        FROM core_product p
        LEFT JOIN core_store s ON s.id = p.store_id
        WHERE p.is_active = true
          AND p.two_d_icon IS NOT NULL
          AND p.two_d_icon != ''
          AND p.two_d_icon = ANY(%s)
    """
    params: list = [list(allowed_keys)]

    surface_sql, surface_params = _surface_category_clause(surface)
    sql += surface_sql
    params.extend(surface_params)

    if category == "uncategorized":
        sql += " AND (p.category IS NULL OR p.category = '')"
    elif category:
        sql += " AND lower(p.category) = lower(%s)"
        params.append(category)

    if search:
        sql += " AND p.name_english ILIKE %s"
        params.append(f"%{search}%")

    if store_id:
        sql += " AND p.store_id = %s"
        params.append(store_id)

    sql += " ORDER BY p.name_english LIMIT %s"
    params.append(limit)

    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def get_categories(allowed_keys: set[str], surface: Optional[str] = None) -> list[dict]:
    """Return [{category, count}] for products that have an icon in S3."""
    if not allowed_keys:
        return []

    sql = """
        SELECT p.category, COUNT(p.id) AS count
        FROM core_product p
        WHERE p.is_active = true
          AND p.two_d_icon IS NOT NULL
          AND p.two_d_icon != ''
          AND p.two_d_icon = ANY(%s)
    """
    params: list = [list(allowed_keys)]
    surface_sql, surface_params = _surface_category_clause(surface)
    sql += surface_sql
    params.extend(surface_params)
    sql += " GROUP BY p.category"

    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]


def get_product_two_d_icon(product_id: int) -> Optional[str]:
    """Return the two_d_icon value for a single product, or None."""
    sql = """
        SELECT two_d_icon
        FROM core_product
        WHERE id = %s
          AND two_d_icon IS NOT NULL
          AND two_d_icon != ''
        LIMIT 1
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [product_id])
            row = cur.fetchone()
            return row[0] if row else None
