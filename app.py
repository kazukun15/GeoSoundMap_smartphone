import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import math
from skimage import measure
import branca.colormap as cm

# Initial setup
if "map_center" not in st.session_state:
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 14
if "speakers" not in st.session_state:
    st.session_state.speakers = [[34.25741795269067, 133.20450105700033, [0.0, 90.0]]]
if "measurements" not in st.session_state:
    st.session_state.measurements = []
if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None
if "contours" not in st.session_state:
    st.session_state.contours = {"60dB": [], "80dB": []}
if "L0" not in st.session_state:
    st.session_state.L0 = 80
if "r_max" not in st.session_state:
    st.session_state.r_max = 500

# Direction conversion
DIRECTION_MAPPING = {"N": 0, "E": 90, "S": 180, "W": 270, "NE": 45, "SE": 135, "SW": 225, "NW": 315}

def parse_direction_to_degrees(direction_str):
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    return float(direction_str)

# Heatmap and contour calculation
def calculate_heatmap_and_contours(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for spk in speakers:
        lat, lon, dirs = spk
        spk_coords = np.array([lat, lon])
        distances = np.sqrt(np.sum((grid_coords - spk_coords)**2, axis=1)) * 111320
        distances[distances < 1] = 1

        bearings = np.degrees(np.arctan2(grid_coords[:, 1] - lon, grid_coords[:, 0] - lat)) % 360
        power = np.zeros_like(distances)

        for direction in dirs:
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            directivity_factor = np.clip(1 - angle_diff / 180, 0, 1)
            power += directivity_factor * 10**((L0 - 20 * np.log10(distances)) / 10)

        power[distances > r_max] = 0
        power_sum += power.reshape(Nx, Ny)

    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    sound_grid = np.clip(sound_grid, L0 - 40, L0)
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] for i in range(Nx) for j in range(Ny) if not np.isnan(sound_grid[i, j])]

    contours = {"60dB": [], "80dB": []}
    levels = {"60dB": 60, "80dB": 80}
    cgrid = np.where(np.isnan(sound_grid), -9999, sound_grid)
    for key, level in levels.items():
        raw_contours = measure.find_contours(cgrid, level=level)
        for contour in raw_contours:
            lat_lon_contour = [(grid_lat[int(y), int(x)], grid_lon[int(y), int(x)]) for y, x in contour]
            contours[key].append(lat_lon_contour)

    return heat_data, contours

# CSV import/export
def import_csv(file):
    try:
        data = pd.read_csv(file)
        speakers = []
        measurements = []

        for _, row in data.iterrows():
            if not pd.isna(row["Speaker_Lat"]):
                dirs = [parse_direction_to_degrees(row[f"Direction{i}"]) for i in range(1, 4) if not pd.isna(row[f"Direction{i}"])]
                speakers.append([row["Speaker_Lat"], row["Speaker_Lon"], dirs])

            if not pd.isna(row["Measurement_Lat"]):
                measurements.append([row["Measurement_Lat"], row["Measurement_Lon"], row["Measured_dB"]])

        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return None, None

def export_csv(speakers, measurements):
    data = []
    for spk in speakers:
        data.append({"Speaker_Lat": spk[0], "Speaker_Lon": spk[1], "Direction1": spk[2][0] if len(spk[2]) > 0 else None,
                     "Direction2": spk[2][1] if len(spk[2]) > 1 else None, "Direction3": spk[2][2] if len(spk[2]) > 2 else None,
                     "Measurement_Lat": None, "Measurement_Lon": None, "Measured_dB": None})
    for meas in measurements:
        data.append({"Speaker_Lat": None, "Speaker_Lon": None, "Direction1": None, "Direction2": None, "Direction3": None,
                     "Measurement_Lat": meas[0], "Measurement_Lon": meas[1], "Measured_dB": meas[2]})

    return pd.DataFrame(data)

# Map display
st.title("音圧ヒートマップ表示 - 防災スピーカー")
if st.session_state.heatmap_data is None and st.session_state.speakers:
    lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
    lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
    zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
    grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, zoom_factor), np.linspace(lon_min, lon_max, zoom_factor))
    st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
        st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
    )

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
for spk in st.session_state.speakers:
    lat, lon, dirs = spk
    popup_text = f"スピーカー: ({lat:.6f}, {lon:.6f})<br>初期音圧レベル: {st.session_state.L0} dB<br>最大伝播距離: {st.session_state.r_max} m"
    folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300), icon=folium.Icon(color="blue")).add_to(m)
if st.session_state.heatmap_data:
    HeatMap(st.session_state.heatmap_data, radius=15, blur=20, min_opacity=0.4).add_to(m)
st_folium(m, width=700, height=500)

# Operation panel
st.subheader("操作パネル")
uploaded_file = st.file_uploader("CSVをインポート", type=["csv"])
if uploaded_file is not None:
    speakers, measurements = import_csv(uploaded_file)
    if speakers is not None and measurements is not None:
        st.session_state.speakers = speakers
        st.session_state.measurements = measurements
        st.success("CSVファイルをインポートしました！")

if st.button("CSVをエクスポート"):
    df = export_csv(st.session_state.speakers, st.session_state.measurements)
    st.download_button("ダウンロード", df.to_csv(index=False), "data.csv", "text/csv")
