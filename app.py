import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math
from skimage import measure
import branca.colormap as cm

# 初期設定
if "map_center" not in st.session_state:
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 14  # 初期ズームレベル

if "speakers" not in st.session_state:
    st.session_state.speakers = [[34.25741795269067, 133.20450105700033, [0.0, 90.0]]]

if "measurements" not in st.session_state:
    st.session_state.measurements = []  # 計測値リスト

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None

if "contours" not in st.session_state:
    st.session_state.contours = {"60dB": [], "80dB": []}

if "L0" not in st.session_state:
    st.session_state.L0 = 80  # 初期音圧レベル

if "r_max" not in st.session_state:
    st.session_state.r_max = 500  # 初期伝播距離

# 方角の変換
DIRECTION_MAPPING = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction_to_degrees(direction_str):
    """方角文字列を角度に変換"""
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    return float(direction_str)  # 数値の場合そのまま返す

# 音圧ヒートマップと等高線の計算
def calculate_heatmap_and_contours(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))

    # ベクトル計算でスピーカーごとの寄与を効率的に加算
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for spk in speakers:
        lat, lon, dirs = spk
        spk_coords = np.array([lat, lon])
        distances = np.sqrt(np.sum((grid_coords - spk_coords) ** 2, axis=1)) * 111320  # 距離を計算 (メートル換算)
        distances[distances < 1] = 1  # 最小距離を1mに設定

        # スピーカーの指向性を計算
        bearings = np.degrees(np.arctan2(grid_coords[:, 1] - lon, grid_coords[:, 0] - lat)) % 360
        power = np.zeros_like(distances)

        for direction in dirs:
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            directivity_factor = np.clip(1 - angle_diff / 180, 0, 1)  # 指向性の減衰を適用
            power += directivity_factor * 10 ** ((L0 - 20 * np.log10(distances)) / 10)

        # 距離制限を適用
        power[distances > r_max] = 0
        power_sum += power.reshape(Nx, Ny)

    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    sound_grid = np.clip(sound_grid, L0 - 40, L0)  # 範囲外の値をクリップ
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] for i in range(Nx) for j in range(Ny) if not np.isnan(sound_grid[i, j])]

    # 等高線を計算 (60dB, 80dB)
    contours = {"60dB": [], "80dB": []}
    levels = {"60dB": 60, "80dB": 80}
    cgrid = np.where(np.isnan(sound_grid), -9999, sound_grid)
    for key, level in levels.items():
        raw_contours = measure.find_contours(cgrid, level=level)
        for contour in raw_contours:
            lat_lon_contour = [(grid_lat[int(y), int(x)], grid_lon[int(y), int(x)]) for y, x in contour]
            contours[key].append(lat_lon_contour)

    return heat_data, contours

# 地図の表示
st.title("音圧ヒートマップ表示 - 防災スピーカーの非可聴域検出")
lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01

# ズームレベルに応じた分割数を調整
zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, zoom_factor), np.linspace(lon_min, lon_max, zoom_factor))

if st.session_state.heatmap_data is None and st.session_state.speakers:
    st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
        st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
    )

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

# スピーカーのマーカー
for spk in st.session_state.speakers:
    lat, lon, dirs = spk
    popup_content = f"""
    <div style="font-size:14px; line-height:1.5;">
        <b>スピーカー位置:</b> ({lat:.6f}, {lon:.6f})<br>
        <b>方向:</b> {dirs}<br>
        <b>初期音圧レベル:</b> {st.session_state.L0} dB<br>
        <b>最大伝播距離:</b> {st.session_state.r_max} m
    </div>
    """
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_content, max_width=300),
        icon=folium.Icon(color="blue")
    ).add_to(m)

# 計測値のマーカー
for meas in st.session_state.measurements:
    lat, lon, db = meas
    theoretical_value = calculate_heatmap_and_contours(
        [[lat, lon, []]], st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
    )[0][0][-1]
    popup_content = f"""
    <div style="font-size:14px; line-height:1.5;">
        <b>計測位置:</b> ({lat:.6f}, {lon:.6f})<br>
        <b>計測値:</b> {db} dB<br>
        <b>理論値:</b> {theoretical_value:.2f} dB
    </div>
    """
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_content, max_width=300),
        icon=folium.Icon(color="green")
    ).add_to(m)

# 地図を表示
st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])
