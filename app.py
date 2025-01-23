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

    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for spk in speakers:
        lat, lon, dirs = spk
        spk_coords = np.array([lat, lon])
        distances = np.sqrt(np.sum((grid_coords - spk_coords) ** 2, axis=1)) * 111320
        distances[distances < 1] = 1  # 最小距離を1mに設定

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

# 地図の表示
st.title("音圧ヒートマップ表示 - 防災スピーカーの非可聴域検出")
lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01

zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, zoom_factor), np.linspace(lon_min, lon_max, zoom_factor))

if st.session_state.heatmap_data is None and st.session_state.speakers:
    st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
        st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
    )

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

for spk in st.session_state.speakers:
    lat, lon, dirs = spk
    popup_text = f"スピーカー: ({lat:.6f}, {lon:.6f})\n方向: {dirs}"
    folium.Marker(location=[lat, lon], popup=popup_text, icon=folium.Icon(color="blue")).add_to(m)

for meas in st.session_state.measurements:
    lat, lon, db = meas
    folium.Marker(location=[lat, lon], popup=f"計測値: {db} dB", icon=folium.Icon(color="green")).add_to(m)

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

# 操作パネル
st.subheader("操作パネル")
with st.form(key="controls"):
    st.write("スピーカーの設定")
    col1, col2 = st.columns(2)

    with col1:
        new_speaker = st.text_input("新しいスピーカー (緯度,経度,方向1,方向2...)", placeholder="例: 34.2579,133.2072,N,E")
        if st.form_submit_button("スピーカーを追加"):
            try:
                parts = new_speaker.split(",")
                lat, lon = float(parts[0]), float(parts[1])
                directions = [parse_direction_to_degrees(d) for d in parts[2:]]
                st.session_state.speakers.append([lat, lon, directions])
                st.session_state.heatmap_data = None
                st.success(f"スピーカーを追加しました: ({lat}, {lon}), 方向: {directions}")
            except ValueError:
                st.error("入力形式が正しくありません")

    with col2:
        if st.form_submit_button("スピーカーをリセット"):
            st.session_state.speakers = []
            st.session_state.heatmap_data = None
            st.session_state.contours = {"60dB": [], "80dB": []}
            st.success("スピーカーをリセットしました")

    st.write("音圧設定")
    st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
    st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)

    st.write("計測値の設定")
    new_measurement = st.text_input("計測値 (緯度,経度,デシベル)", placeholder="例: 34.2579,133.2072,75")
    if st.form_submit_button("計測値を追加"):
        try:
            lat, lon, db = map(float, new_measurement.split(","))
            st.session_state.measurements.append([lat, lon, db])
            st.success(f"計測値を追加しました: ({lat}, {lon}), {db} dB")
        except ValueError:
            st.error("入力形式が正しくありません")

    if st.form_submit_button("更新"):
        if st.session_state.speakers:
            st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
                st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
            )
            st.success("ヒートマップと等高線を更新しました")
        else:
            st.error("スピーカーが存在しません。")

# 計測値一覧表示
st.subheader("計測値の一覧")
if st.session_state.measurements:
    table_data = []
    for lat, lon, meas_db in st.session_state.measurements:
        if st.session_state.heatmap_data is not None:
            theoretical_value = calculate_heatmap_and_contours(
                st.session_state.speakers, st.session_state.L0, st.session_state.r_max, np.array([[lat]]), np.array([[lon]])
            )[0][0][2]
        else:
            theoretical_value = "N/A"
        table_data.append({
            "緯度": f"{lat:.6f}",
            "経度": f"{lon:.6f}",
            "実測値 (dB)": f"{meas_db:.2f}",
            "理論値 (dB)": f"{theoretical_value:.2f}" if theoretical_value != "N/A" else "N/A"
        })

    st.table(table_data)
else:
    st.write("計測値がまだ追加されていません。")
