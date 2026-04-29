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
    Reads GeoJSON directly (no geopandas) and injects data values into feature properties.
    Supports numeric columns (linear colormap) or categorical typology.
    """
    import folium
    from branca.colormap import LinearColormap

    center_lat, center_lon = 33.5, -90.7
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

    # Load GeoJSON as plain dict — no geopandas needed
    with open(geojson_path, encoding="utf-8") as f:
        geojson_data = json.load(f)

    # Build GEOID → row lookup from delta_df
    needed = [color_col]
    for extra in ("igs_score", "typology", "pc_hpsa_score_max"):
        if extra in delta_df.columns and extra not in needed:
            needed.append(extra)

    delta_lkp = (
        delta_df[["GEOID"] + needed]
        .copy()
        .assign(GEOID=lambda d: d["GEOID"].astype(str))
        .drop_duplicates("GEOID")
        .set_index("GEOID")
        .to_dict(orient="index")
    )

    # Inject values into GeoJSON feature properties
    matched = 0
    for feature in geojson_data["features"]:
        geoid = str(feature["properties"].get("GEOID", ""))
        row = delta_lkp.get(geoid, {})
        feature["properties"].update(row)
        if row:
            matched += 1

    if matched == 0 or color_col not in next(iter(delta_lkp.values()), {}):
        if show_fqhc and fqhc_df is not None:
            add_fqhc_markers(m, fqhc_df)
        return m

    is_categorical = color_col == "typology"

    if is_categorical:
        def _style_typology(feature):
            cat = str(feature["properties"].get(color_col) or "Unknown")
            fill = TYPOLOGY_COLORS.get(cat, TYPOLOGY_COLORS["Unknown"])
            return {"fillColor": fill, "color": "#6b7280", "weight": 1, "fillOpacity": 0.72}

        tt_fields  = ["GEOID", color_col]
        tt_aliases = ["Tract:", "Typology:"]
        if "igs_score" in needed and color_col != "igs_score":
            tt_fields.append("igs_score"); tt_aliases.append("IGS Score:")

        folium.GeoJson(
            geojson_data,
            style_function=_style_typology,
            tooltip=folium.GeoJsonTooltip(fields=tt_fields, aliases=tt_aliases,
                                          style="font-size:12px;"),
        ).add_to(m)

    else:
        if color_col == "pc_hpsa_score_max":
            vmin, vmax = 0.0, 26.0
        else:
            raw_vals = [
                feature["properties"].get(color_col)
                for feature in geojson_data["features"]
            ]
            vals = [float(v) for v in raw_vals if v is not None]
            if not vals:
                if show_fqhc and fqhc_df is not None:
                    add_fqhc_markers(m, fqhc_df)
                return m
            vmin, vmax = min(vals), max(vals)
            if vmin == vmax:
                vmax = vmin + 1e-6

        colormap = LinearColormap(
            colors=["#dc2626", "#f97316", "#eab308", "#22c55e", "#059669"],
            vmin=vmin, vmax=vmax, caption=color_label,
        )

        def _style_numeric(feature):
            z = feature["properties"].get(color_col)
            try:
                zf = float(z) if z is not None else vmin
                zf = max(vmin, min(vmax, zf))
            except (TypeError, ValueError):
                zf = vmin
            return {"fillColor": colormap(zf), "color": "#6b7280", "weight": 1,
                    "fillOpacity": 0.7}

        tt_fields  = ["GEOID", color_col]
        tt_aliases = ["Tract:", f"{color_label}:"]
        if color_col != "igs_score" and "igs_score" in needed:
            tt_fields.append("igs_score"); tt_aliases.append("IGS Score:")

        folium.GeoJson(
            geojson_data,
            style_function=_style_numeric,
            tooltip=folium.GeoJsonTooltip(fields=tt_fields, aliases=tt_aliases,
                                          style="font-size:12px;"),
        ).add_to(m)
        colormap.add_to(m)

    if show_fqhc and fqhc_df is not None:
        add_fqhc_markers(m, fqhc_df)

    return m
