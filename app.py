import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import math
from skimage import measure
import branca.colormap as cm
import io

# ───────────── 初期設定 ───────────── #
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
if "L0" not in st.session_state:
    st.session_state.L0 = 80
if "r_max" not in st.session_state:
    st.session_state.r_max = 500

# 方角変換
DIRECTION_MAPPING = {"N": 0, "E": 90, "S": 180, "W": 270, "NE": 45, "SE": 135, "SW": 225, "NW": 315}

def parse_direction(direction_str):
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    try:
        return float(direction_str)
    except ValueError:
        st.error(f"方向 '{direction_str}' を変換できません。0度に設定します。")
        return 0.0

# ───────────── CSV機能 ───────────── #
def load_csv(file):
    try:
        df = pd.read_csv(file)
        speakers, measurements = [], []
        for _, row in df.iterrows():
            if not pd.isna(row.get("スピーカー緯度")):
                lat, lon = row["スピーカー緯度"], row["スピーカー経度"]
                directions = [parse_direction(row.get(f"方向{i}", "")) for i in range(1, 4) if not pd.isna(row.get(f"方向{i}"))]
                speakers.append([lat, lon, directions])
            if not pd.isna(row.get("計測位置緯度")):
                lat, lon, db = row["計測位置緯度"], row["計測位置経度"], row.get("計測デシベル", 0)
                measurements.append([lat, lon, float(db)])
        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return [], []

def export_csv(data, columns):
    df = pd.DataFrame(data, columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ───────────── ヒートマップ計算 ───────────── #
def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
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
            power += np.clip(1 - angle_diff / 180, 0, 1) * 10 ** ((L0 - 20 * np.log10(distances)) / 10)

        power[distances > r_max] = 0
        power_sum += power.reshape(Nx, Ny)

    sound_grid = np.full_like(power_sum, np.nan)
    sound_grid[power_sum > 0] = 10 * np.log10(power_sum[power_sum > 0])
    sound_grid = np.clip(sound_grid, L0 - 40, L0)
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] for i in range(Nx) for j in range(Ny) if not np.isnan(sound_grid[i, j])]
    return heat_data

# ───────────── アプリ表示 ───────────── #
st.title("防災スピーカー音圧可視化マップ")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
if uploaded_file:
    speakers, measurements = load_csv(uploaded_file)
    if speakers:
        st.session_state.speakers.extend(speakers)
    if measurements:
        st.session_state.measurements.extend(measurements)
    st.success("CSVファイルを読み込みました。")

lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))

if st.session_state.heatmap_data is None:
    st.session_state.heatmap_data = calculate_heatmap(st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon)

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
for spk in st.session_state.speakers:
    lat, lon, dirs = spk
    popup_text = f"<b>スピーカー</b>: ({lat:.6f}, {lon:.6f})<br><b>方向</b>: {dirs}"
    folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300)).add_to(m)

if st.session_state.heatmap_data:
    HeatMap(st.session_state.heatmap_data).add_to(m)

st_data = st_folium(m, width=700, height=500)

st.subheader("操作パネル")
new_speaker = st.text_input("スピーカー (緯度,経度,方向1,方向2...)", placeholder="例: 34.2579,133.2072,N,E")
if st.button("スピーカー追加"):
    try:
        parts = new_speaker.split(",")
        lat, lon = float(parts[0]), float(parts[1])
        directions = [parse_direction(dir_str) for dir_str in parts[2:]]
        st.session_state.speakers.append([lat, lon, directions])
        st.session_state.heatmap_data = None
        st.success(f"スピーカーを追加しました: {lat}, {lon}, {directions}")
    except ValueError:
        st.error("形式が正しくありません。(緯度,経度,方向...) の形式で入力してください。")

if st.button("スピーカーリセット"):
    st.session_state.speakers = []
    st.session_state.heatmap_data = None
    st.success("スピーカーをリセットしました")

st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)

csv_data_speakers = export_csv(st.session_state.speakers, ["スピーカー緯度", "スピーカー経度", "方向1", "方向2", "方向3"])
st.download_button("スピーカーCSVダウンロード", csv_data_speakers, "speakers.csv", "text/csv")

