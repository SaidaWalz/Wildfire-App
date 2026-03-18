# streamlit_wildfire.py
# Wildfire Risk Predictor — Québec
# Flow: Click map → H3 r5 snap → fetch ERA5-Land (last 12 months) → XGBoost inference → show result
#
# Requirements:
#   pip install streamlit streamlit-folium folium h3 cdsapi xarray numpy pandas xgboost joblib
#
# Secrets (.streamlit/secrets.toml):
#   CDS_URL = "https://cds.climate.copernicus.eu/api"
#   CDS_KEY = "your-cds-api-key"
#
# Model file:
#   saved_model/xgb_model.pkl   (XGBoost trained on ERA5-Land features, seq_len=12)

import math
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import folium
import streamlit as st
from folium import Element
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wildfire Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="stSidebar"]          { background: #f7f7f7; border-right:1px solid #e0e0e0; }
h1,h2,h3,h4                        { color: #1a1a1a !important; }
.risk-card {
    border-radius: 12px; padding: 18px 22px; margin: 10px 0;
    border: 1px solid #e0e0e0;
}
.risk-high   { background:#fff0f0; border-color:#cc3333; }
.risk-medium { background:#fff8f0; border-color:#cc7700; }
.risk-low    { background:#f0fff0; border-color:#33aa33; }
.risk-none   { background:#f7f7f7; border-color:#aaaaaa; }
.risk-label-high   { color:#cc2222; font-size:2rem; font-weight:800; }
.risk-label-medium { color:#cc7700; font-size:2rem; font-weight:800; }
.risk-label-low    { color:#228822; font-size:2rem; font-weight:800; }
.risk-label-none   { color:#666666; font-size:2rem; font-weight:800; }
.feat-table { font-size:0.85rem; color:#333; }
.stButton>button { background:#f0f0f0; color:#1a1a1a; border:1px solid #cccccc; border-radius:8px; font-weight:700; }
.stButton>button:hover { background:#e0e0e0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
H3_RES        = 5
SEQ_LEN       = 12
N_FEATURES    = 7
THRESHOLD     = 0.79        # best F1 threshold from XGBoost evaluation
MODEL_PATH    = Path("saved_model") / "xgb_model.pkl"
SCALER_PATH   = Path("saved_model") / "scaler.pkl"   # optional

FEATURE_NAMES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "surface_solar_radiation_downwards",
    "total_evaporation",
    "wind_total",
    "total_precipitation",
    "leaf_area_index_high_vegetation",
]
ERA5_VARIABLES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "surface_solar_radiation_downwards",
    "total_evaporation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "leaf_area_index_high_vegetation",
]

DEFAULT_CENTER  = (52.0, -71.0)
DEFAULT_ZOOM    = 5

# ─────────────────────────────────────────────────────────
# CACHED MODEL LOADING
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import joblib
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}\n"
                                f"Place your xgb_model.pkl at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def load_scaler():
    if not SCALER_PATH.exists():
        return None
    import joblib
    return joblib.load(SCALER_PATH)

# ─────────────────────────────────────────────────────────
# ERA5-LAND FETCH  (cdsapi)
# ─────────────────────────────────────────────────────────
def get_cds_client():
    url = st.secrets.get("CDS_URL", os.getenv("CDS_URL", ""))
    key = st.secrets.get("CDS_KEY", os.getenv("CDS_KEY", ""))
    if not url or not key:
        st.error("CDS credentials missing.\n\n"
                 "Add to `.streamlit/secrets.toml`:\n"
                 '```\nCDS_URL = "https://cds.climate.copernicus.eu/api"\n'
                 'CDS_KEY  = "your-key"\n```')
        st.stop()
    import cdsapi
    return cdsapi.Client(url=url, key=key, quiet=True)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_era5_sequence(lat: float, lon: float, end_date_str: str) -> pd.DataFrame:
    import cdsapi
    import xarray as xr
    import zipfile

    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    latest = datetime.utcnow().replace(day=1) - timedelta(days=90)
    if end > latest:
        end = latest

    months_list = []
    d = end.replace(day=1)
    for _ in range(SEQ_LEN):
        months_list.append(d)
        if d.month == 1:
            d = d.replace(year=d.year - 1, month=12)
        else:
            d = d.replace(month=d.month - 1)
    months_list = sorted(months_list)

    years  = sorted({str(d.year)           for d in months_list})
    months = sorted({str(d.month).zfill(2) for d in months_list})
    area   = [round(lat + 0.5, 2), round(lon - 0.5, 2),
              round(lat - 0.5, 2), round(lon + 0.5, 2)]

    client = get_cds_client()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    client.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable":     ERA5_VARIABLES,
            "year":         years,
            "month":        months,
            "time":         "00:00",
            "area":         area,
            "format":       "netcdf",
        },
        tmp_path,
    )

    if zipfile.is_zipfile(tmp_path):
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(tmp_path, "r") as z:
            z.extractall(extract_dir)
        candidates = list(Path(extract_dir).glob("*.nc")) + list(Path(extract_dir).glob("*.netcdf"))
        if not candidates:
            raise RuntimeError("ZIP from CDS contained no .nc files")
        tmp_path = str(candidates[0])

    ds    = xr.open_dataset(tmp_path, engine="netcdf4")
    ds_pt = ds.sel(latitude=lat, longitude=lon, method="nearest")
    time_coord = "valid_time" if "valid_time" in ds_pt.coords else "time"

    records = []
    for d in months_list:
        month_str = d.strftime("%Y-%m")

        def _val(var):
            try:
                times = ds_pt[time_coord].values
                mask  = [(str(t)[:7] == month_str) for t in times]
                idx   = next(i for i, m in enumerate(mask) if m)
                return float(ds_pt[var].isel({time_coord: idx}).values)
            except Exception:
                return np.nan

        u = _val("u10")
        v = _val("v10")
        records.append({
            "date":                                  month_str,
            "2m_temperature":                        _val("t2m"),
            "volumetric_soil_water_layer_1":         _val("swvl1"),
            "surface_solar_radiation_downwards":     _val("ssrd"),
            "total_evaporation":                     _val("e"),
            "wind_total":                            math.sqrt(u**2 + v**2) if not (np.isnan(u) or np.isnan(v)) else np.nan,
            "total_precipitation":                   _val("tp"),
            "leaf_area_index_high_vegetation":       _val("lai_hv"),
        })

    ds.close()
    os.unlink(tmp_path)
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────
# INFERENCE — XGBoost
# ─────────────────────────────────────────────────────────
def run_xgboost(model, scaler, df: pd.DataFrame) -> float:
    """df shape (12, 7) → flatten to (1, 84) → scalar fire probability."""
    x = df[FEATURE_NAMES].values.astype(np.float32)   # (12, 7)

    if scaler is not None:
        flat = x.reshape(-1, N_FEATURES)
        flat = scaler.transform(flat)
        x    = flat.reshape(SEQ_LEN, N_FEATURES)

    # XGBoost expects 2D: flatten sequence → (1, 12*7) = (1, 84)
    x_flat = x.reshape(1, -1)

    prob = model.predict_proba(x_flat)[0, 1]
    return float(prob)

# ─────────────────────────────────────────────────────────
# RISK HELPERS
# ─────────────────────────────────────────────────────────
def risk_info(p: float):
    if   p >= 0.75: return "High",     "🔥🔥🔥", "risk-high",   "risk-label-high"
    elif p >= 0.5:  return "Moderate", "🔥🔥",   "risk-medium", "risk-label-medium"
    elif p >= 0.25: return "Low",      "🔥",      "risk-low",    "risk-label-low"
    else:           return "Minimal",  "—",       "risk-none",   "risk-label-none"

def hex_color(p: float) -> str:
    if   p >= 0.75: return "#cc2222"
    elif p >= 0.5:  return "#dd8800"
    elif p >= 0.25: return "#eecc00"
    else:           return "#33aa33"

# ─────────────────────────────────────────────────────────
# H3 MAP HELPER
# ─────────────────────────────────────────────────────────
def h3_polygon_coords(cell: str):
    boundary = h3.cell_to_boundary(cell)
    return [(lat, lon) for lat, lon in boundary]

# ─────────────────────────────────────────────────────────
# LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────────────────
st.title("Wildfire Prediction")
st.caption("Click anywhere on the map → snaps to H3 r5 cell → fetches ERA5-Land (last 12 months) → XGBoost prediction")

model_obj  = None
scaler_obj = None
model_err  = None

with st.spinner("Loading XGBoost model…"):
    try:
        model_obj  = load_model()
        scaler_obj = load_scaler()
    except Exception as e:
        model_err  = str(e)

if model_err:
    st.warning(f"Model not loaded: {model_err}\n\nPlace your model at `{MODEL_PATH}`")

# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
if "center"   not in st.session_state: st.session_state["center"]   = DEFAULT_CENTER
if "h3_cell"  not in st.session_state: st.session_state["h3_cell"]  = None
if "cell_lat" not in st.session_state: st.session_state["cell_lat"] = None
if "cell_lon" not in st.session_state: st.session_state["cell_lon"] = None
if "result"   not in st.session_state: st.session_state["result"]   = None
if "era5_df"  not in st.session_state: st.session_state["era5_df"]  = None

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Selected Cell")
    if st.session_state["h3_cell"]:
        st.code(st.session_state["h3_cell"], language=None)
        st.write(f"Center: `{st.session_state['cell_lat']:.5f}, {st.session_state['cell_lon']:.5f}`")
    else:
        st.info("Click the map to select a cell.")

    st.divider()
    st.header("Date Range")
    end_date = st.date_input(
        "End date (last day of sequence)",
        value=datetime.utcnow().date() - timedelta(days=5),
        max_value=datetime.utcnow().date() - timedelta(days=5),
    )
    st.caption(f"Fetches 12 months ending: {end_date.strftime('%Y-%m')}")

    st.divider()
    run_btn   = st.button("▶ Run Prediction", type="primary", use_container_width=True,
                          disabled=(st.session_state["h3_cell"] is None or model_obj is None))
    clear_btn = st.button("Clear", use_container_width=True)

    st.divider()
    st.header("Info")
    st.markdown(f"""
- **Model:** XGBoost
- **Input:** `(12 × 7)` flattened → `(84,)`
- **H3 resolution:** {H3_RES}
- **ERA5-Land variables:** {len(ERA5_VARIABLES)} raw → 7 features
- **wind_total** = √(u² + v²)
- **Threshold:** {THRESHOLD}
""")

if clear_btn:
    for k in ["result", "era5_df", "h3_cell", "cell_lat", "cell_lon"]:
        st.session_state[k] = None
    st.rerun()

# ─────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────
st.subheader("Click to select location")
map_center = list(st.session_state["center"])
m = folium.Map(
    location=map_center,
    zoom_start=DEFAULT_ZOOM,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
)

if st.session_state["h3_cell"]:
    cell    = st.session_state["h3_cell"]
    p_val   = st.session_state["result"]
    f_color = hex_color(p_val) if p_val is not None else "#4a8a4a"
    coords  = h3_polygon_coords(cell)
    folium.Polygon(
        locations=coords,
        color=f_color,
        fill=True,
        fill_color=f_color,
        fill_opacity=0.35,
        weight=2,
        popup=f"H3: {cell}<br>Center: ({st.session_state['cell_lat']:.5f}, {st.session_state['cell_lon']:.5f})"
              + (f"<br>p = {p_val:.3f}" if p_val is not None else ""),
    ).add_to(m)
    folium.CircleMarker(
        location=(st.session_state["cell_lat"], st.session_state["cell_lon"]),
        radius=5, color="#ffffff", fill=True, fill_opacity=1.0,
        popup="H3 cell centre",
    ).add_to(m)

legend = """
<div style="position:fixed;bottom:28px;right:28px;z-index:9999;
            background:#ffffff;border:1px solid #cccccc;border-radius:10px;
            padding:12px 16px;font-size:12px;color:#333333;">
  <b style="color:#1a1a1a">Fire Risk</b><br>
  <span style="color:#cc2222">■</span> High ≥ 0.75<br>
  <span style="color:#dd8800">■</span> Moderate ≥ 0.50<br>
  <span style="color:#eecc00">■</span> Low ≥ 0.25<br>
  <span style="color:#33aa33">■</span> Minimal &lt; 0.25
</div>
"""
m.get_root().html.add_child(Element(legend))
map_data = st_folium(m, width="100%", height=500, key="main_map")

if map_data and map_data.get("last_clicked"):
    clat = float(map_data["last_clicked"]["lat"])
    clon = float(map_data["last_clicked"]["lng"])
    new_cell = h3.latlng_to_cell(clat, clon, H3_RES)
    if new_cell != st.session_state["h3_cell"]:
        cell_center                     = h3.cell_to_latlng(new_cell)
        st.session_state["h3_cell"]     = new_cell
        st.session_state["cell_lat"]    = round(cell_center[0], 6)
        st.session_state["cell_lon"]    = round(cell_center[1], 6)
        st.session_state["center"]      = (cell_center[0], cell_center[1])
        st.session_state["result"]      = None
        st.session_state["era5_df"]     = None
        st.rerun()

# ─────────────────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────────────────
if run_btn and st.session_state["h3_cell"] and model_obj:
    cell_lat = st.session_state["cell_lat"]
    cell_lon = st.session_state["cell_lon"]
    end_str  = end_date.strftime("%Y-%m-%d")

    with st.spinner(f"Fetching ERA5-Land monthly data for ({cell_lat:.4f}, {cell_lon:.4f}) — last 12 months…"):
        try:
            era5_df = fetch_era5_sequence(cell_lat, cell_lon, end_str)
            st.session_state["era5_df"] = era5_df
        except Exception as e:
            st.error(f"ERA5 fetch failed: {e}")
            st.stop()

    with st.spinner("Running XGBoost inference…"):
        try:
            p = run_xgboost(model_obj, scaler_obj, st.session_state["era5_df"])
            st.session_state["result"] = p
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    st.rerun()

# ─────────────────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────────────────
if st.session_state["result"] is not None:
    p_val                           = st.session_state["result"]
    label, emoji, card_cls, lbl_cls = risk_info(p_val)

    st.divider()
    st.subheader("Prediction Result")

    col_risk, col_prob = st.columns([2, 1])
    with col_risk:
        st.markdown(
            f'<div class="risk-card {card_cls}">'
            f'<span class="{lbl_cls}">{emoji} {label} Risk</span><br><br>'
            f'<span style="color:#333333;font-size:0.95rem;">'
            f'H3 cell: <code>{st.session_state["h3_cell"]}</code><br>'
            f'Centre: {st.session_state["cell_lat"]:.5f}, {st.session_state["cell_lon"]:.5f}'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    with col_prob:
        st.metric("Fire Probability", f"{p_val:.4f}")
        st.metric("Threshold", f"{THRESHOLD}")
        above = "🔴 Above threshold" if p_val >= THRESHOLD else "🟢 Below threshold"
        st.write(above)

    if st.session_state["era5_df"] is not None:
        st.divider()
        st.subheader("ERA5-Land Input Sequence (12 months)")
        df_show = st.session_state["era5_df"].copy()

        fmt = {
            "2m_temperature":                    "{:.2f} K",
            "volumetric_soil_water_layer_1":     "{:.4f} m³/m³",
            "surface_solar_radiation_downwards": "{:.0f} J/m²",
            "total_evaporation":                 "{:.6f} m",
            "wind_total":                        "{:.2f} m/s",
            "total_precipitation":               "{:.6f} m",
            "leaf_area_index_high_vegetation":   "{:.3f}",
        }
        styled = df_show.set_index("date").style
        for col, f in fmt.items():
            if col in df_show.columns:
                styled = styled.format({col: f})
        st.dataframe(styled, use_container_width=True)

        st.subheader("Feature Trends")
        cols = st.columns(3)
        chart_features = [
            ("2m_temperature",                    "Temperature (K)"),
            ("total_precipitation",               "Precipitation (m)"),
            ("wind_total",                        "Wind Speed (m/s)"),
            ("volumetric_soil_water_layer_1",     "Soil Water (m³/m³)"),
            ("surface_solar_radiation_downwards", "Solar Rad (J/m²)"),
            ("leaf_area_index_high_vegetation",   "LAI High Veg"),
        ]
        for i, (feat, title) in enumerate(chart_features):
            with cols[i % 3]:
                st.caption(title)
                if feat in df_show.columns:
                    st.line_chart(df_show.set_index("date")[feat], height=120, use_container_width=True)

        st.download_button(
            "⬇️ Download ERA5 sequence CSV",
            data=df_show.to_csv(index=False).encode("utf-8"),
            file_name=f"era5_{st.session_state['h3_cell']}_{end_date}.csv",
            mime="text/csv",
        )

elif st.session_state["h3_cell"] is None:
    st.info("Click anywhere on the map to select an H3 r5 cell.")
else:
    st.info("Cell selected — press **▶ Run Prediction** in the sidebar.")

st.caption("Wildfire Prediction · ERA5-Land + XGBoost · H3 r5")
