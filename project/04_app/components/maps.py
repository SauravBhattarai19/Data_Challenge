"""
Map components for the Streamlit app.
- National: Plotly county choropleth for IGS scores (local GeoJSON with CDN fallback)
- Delta: Folium tract-level map with numeric or categorical layers + optional FQHC markers
"""

from __future__ import annotations

import json
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import COUNTIES_GEOJSON

_COUNTIES_CDN = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

TYPOLOGY_COLORS = {
    "Stuck": "#dc2626",
    "Turnaround": "#059669",
    "Declining": "#f97316",
    "Resilient": "#2563eb",
    "Unknown": "#9ca3af",
}


def _load_us_counties_geojson() -> dict:
    """Prefer local counties.geojson for offline demos; fall back to Plotly CDN."""
    if COUNTIES_GEOJSON.exists():
        with open(COUNTIES_GEOJSON, encoding="utf-8") as f:
            return json.load(f)
    with urlopen(_COUNTIES_CDN) as response:
        return json.load(response)


def make_national_choropleth(
    county_df: pd.DataFrame,
    color_col: str = "igs_score",
    color_label: str = "Mean IGS Score",
    title: str = "Inclusive Growth Score by County",
) -> go.Figure:
    """
    Plotly county choropleth using US county FIPS GeoJSON.
    county_df must have 'county_fips5' column.
    """
    counties_geo = _load_us_counties_geojson()

    fig = go.Figure(
        go.Choropleth(
            geojson=counties_geo,
            locations=county_df["county_fips5"],
            z=county_df[color_col],
            colorscale=[
                [0.0, "#dc2626"],
                [0.3, "#f97316"],
                [0.45, "#eab308"],
                [0.55, "#84cc16"],
                [0.7, "#22c55e"],
                [1.0, "#059669"],
            ],
            zmin=county_df[color_col].quantile(0.02),
            zmax=county_df[color_col].quantile(0.98),
            colorbar=dict(
                title=dict(text=color_label, font=dict(size=12)),
                thickness=15,
                len=0.6,
            ),
            hovertemplate=(
                "<b>County FIPS: %{location}</b><br>"
                f"{color_label}: %{{z:.1f}}<br>"
                "<extra></extra>"
            ),
            marker_line_width=0.3,
            marker_line_color="#e5e7eb",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#111827")),
        geo=dict(
            scope="usa",
            bgcolor="#fafbfc",
            lakecolor="#dbeafe",
            landcolor="#f3f4f6",
            showlakes=True,
        ),
        paper_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
    )
    return fig


def add_fqhc_markers(m, fqhc_df: pd.DataFrame) -> None:
    """Add FQHC site markers (expects lat, lon; optional site_name, address)."""
    import folium

    if fqhc_df is None or len(fqhc_df) == 0:
        return
    df = fqhc_df.dropna(subset=["lat", "lon"])
    for _, row in df.iterrows():
        try:
            lat, lon = float(row["lat"]), float(row["lon"])
        except (TypeError, ValueError):
            continue
        name = str(row.get("site_name", "") or "")
        addr = str(row.get("address", "") or "")
        popup_html = f"<b>{name}</b><br>{addr}" if name or addr else "FQHC site"
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#1d4ed8",
            weight=2,
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(m)


def make_delta_folium_map(
    delta_df: pd.DataFrame,
    geojson_path=None,
    color_col: str = "igs_score",
    color_label: str = "IGS Score",
    fqhc_df: pd.DataFrame | None = None,
    show_fqhc: bool = True,
) -> object:
    """
    Folium choropleth for Delta census tracts.
    Supports numeric columns (linear colormap) or categorical ``typology``.
    HPSA column ``pc_hpsa_score_max`` is clipped to 0–26 for coloring.
    """
    import folium
    from branca.colormap import LinearColormap

    center_lat = 33.5
    center_lon = -90.7
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if geojson_path is None:
        if show_fqhc and fqhc_df is not None:
            add_fqhc_markers(m, fqhc_df)
        return m

    import geopandas as gpd

    gdf = gpd.read_file(str(geojson_path))
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    delta_df = delta_df.copy()
    delta_df["GEOID"] = delta_df["GEOID"].astype(str)

    merge_cols = ["GEOID", color_col]
    for extra in ("igs_score", "typology", "pc_hpsa_score_max"):
        if extra in delta_df.columns and extra not in merge_cols:
            merge_cols.append(extra)

    merged = gdf.merge(
        delta_df[merge_cols].drop_duplicates("GEOID"),
        on="GEOID",
        how="inner",
    )

    if len(merged) == 0 or color_col not in merged.columns:
        if show_fqhc and fqhc_df is not None:
            add_fqhc_markers(m, fqhc_df)
        return m

    is_categorical = color_col == "typology"

    if is_categorical:
        merged["_typ_cat"] = merged[color_col].fillna("Unknown").astype(str)
        merged["_typ_cat"] = merged["_typ_cat"].replace("", "Unknown")

        def _style_typology(feature):
            cat = feature["properties"].get("_typ_cat", "Unknown")
            fill = TYPOLOGY_COLORS.get(cat, TYPOLOGY_COLORS["Unknown"])
            return {
                "fillColor": fill,
                "color": "#6b7280",
                "weight": 1,
                "fillOpacity": 0.72,
            }

        tooltip_fields = ["GEOID", "_typ_cat"]
        tooltip_aliases = ["Tract:", "Typology:"]
        if "igs_score" in merged.columns:
            tooltip_fields.append("igs_score")
            tooltip_aliases.append("IGS Score:")

        folium.GeoJson(
            merged.to_json(),
            style_function=_style_typology,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                style="font-size:12px;",
            ),
        ).add_to(m)
    else:
        if color_col == "pc_hpsa_score_max":
            merged["_map_z"] = merged[color_col].clip(0, 26).fillna(0.0)
            vmin, vmax = 0.0, 26.0
        else:
            merged["_map_z"] = merged[color_col]
            valid = merged["_map_z"].dropna()
            if len(valid) == 0:
                if show_fqhc and fqhc_df is not None:
                    add_fqhc_markers(m, fqhc_df)
                return m
            vmin = float(valid.min())
            vmax = float(valid.max())
            if vmin == vmax:
                vmax = vmin + 1e-6
            merged["_map_z"] = merged["_map_z"].fillna(vmin)

        colormap = LinearColormap(
            colors=["#dc2626", "#f97316", "#eab308", "#22c55e", "#059669"],
            vmin=vmin,
            vmax=vmax,
            caption=color_label,
        )

        def _style_numeric(feature):
            z = feature["properties"].get("_map_z", vmin)
            try:
                zf = float(z)
            except (TypeError, ValueError):
                zf = vmin
            return {
                "fillColor": colormap(zf),
                "color": "#6b7280",
                "weight": 1,
                "fillOpacity": 0.7,
            }

        tooltip_fields = ["GEOID", color_col]
        tooltip_aliases = ["Tract:", f"{color_label}:"]
        if color_col != "igs_score" and "igs_score" in merged.columns:
            tooltip_fields.append("igs_score")
            tooltip_aliases.append("IGS Score:")
        if color_col != "typology" and "typology" in merged.columns:
            tooltip_fields.append("typology")
            tooltip_aliases.append("Typology:")

        folium.GeoJson(
            merged.to_json(),
            style_function=_style_numeric,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                style="font-size:12px;",
            ),
        ).add_to(m)
        colormap.add_to(m)

    if show_fqhc and fqhc_df is not None:
        add_fqhc_markers(m, fqhc_df)

    return m
