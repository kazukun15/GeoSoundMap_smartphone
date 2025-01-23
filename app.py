import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math
from skimage import measure
import branca.colormap as cm
import pandas as pd

# 初期設定
if "map_center" not in st.session_state:
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 14

if "speakers" not in st.session_state:
    st.session_state.speakers = []

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

if "update_flag" not in st.session_state:
    st.session_state.update_flag = False

# 方角の変換
DIRECTION_MAPPING = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction_to_degrees(direction_str):
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    return float(direction_str)

def calculate_heatmap_and_contours(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for spk in speakers:
        lat, lon, dirs = spk
        spk_coords = np.array([lat, lon])
        distances = np.sqrt(np.sum((grid_coords - spk_coords) ** 2, axis=1)) * 111320
        distances[distances < 1] = 1
        bearings = np.degrees(np.arctan2(grid_coords[:, 1] - lon, grid_coords[:, 0] - lat)) % 360
        power = np.zeros_like(distances)

        for direction in dirs:
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            directivity_factor = np.clip(1 - angle_diff / 180, 0, 1)
            power += directivity_factor * 10 ** ((L0 - 20 * np.log10(distances)) / 10)

        power[distances > r_max] = 0
        power_sum += power.reshape(Nx, Ny)

    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
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

st.title("音圧ヒートマップ表示")

lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, zoom_factor), np.linspace(lon_min, lon_max, zoom_factor))

if st.session_state.update_flag and st.session_state.speakers:
    st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
        st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
    )
    st.session_state.update_flag = False  # 更新フラグをリセット

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

for spk in st.session_state.speakers:
    lat, lon, dirs = spk
    popup_text = f"スピーカー: ({lat:.6f}, {lon:.6f})<br>初期音圧レベル: {st.session_state.L0} dB<br>最大伝播距離: {st.session_state.r_max} m"
    folium.Marker(location=[lat, lon], popup=popup_text, icon=folium.Icon(color="blue")).add_to(m)

for meas in st.session_state.measurements:
    lat, lon, db = meas
    popup_text = f"計測値: {db:.2f} dB<br>位置: ({lat:.6f}, {lon:.6f})"
    folium.Marker(location=[lat, lon], popup=popup_text, icon=folium.Icon(color="green")).add_to(m)

if st.session_state.heatmap_data:
    HeatMap(st.session_state.heatmap_data, radius=15, blur=20, min_opacity=0.4).add_to(m)

for contour in st.session_state.contours["60dB"]:
    folium.PolyLine(locations=contour, color="blue", weight=2, tooltip="60dB").add_to(m)

for contour in st.session_state.contours["80dB"]:
    folium.PolyLine(locations=contour, color="red", weight=2, tooltip="80dB").add_to(m)

st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])

if st_data:
    if "center" in st_data:
        st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    if "zoom" in st_data:
        st.session_state.map_zoom = st_data["zoom"]

st.subheader("操作パネル")
with st.form(key="controls"):
    csv_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                if row["タイプ"] == "スピーカー":
                    st.session_state.speakers.append([row["緯度"], row["経度"], [row["方向1"], row["方向2"], row["方向3"]]])
                elif row["タイプ"] == "計測値":
                    st.session_state.measurements.append([row["緯度"], row["経度"], row["デシベル"]])
            st.success("CSVデータを読み込みました")
        except Exception as e:
            st.error(f"CSVの読み込みに失敗しました: {e}")

    if st.form_submit_button("更新"):
        st.session_state.update_flag = True
        st.success("ヒートマップを更新します")
