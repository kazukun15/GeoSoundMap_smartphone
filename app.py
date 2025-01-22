import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math

# 初期設定
if "map_center" not in st.session_state:
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 17

if "speakers" not in st.session_state:
    st.session_state.speakers = []  # [[lat, lon, [direction1, direction2, ...]], ...]

if "measurements" not in st.session_state:
    st.session_state.measurements = []  # [[lat, lon, db], ...]

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None

# 音圧ヒートマップの計算
def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, dirs = spk
        for i in range(Nx):
            for j in range(Ny):
                r = math.sqrt((grid_lat[i, j] - lat) ** 2 + (grid_lon[i, j] - lon) ** 2) * 111320  # 距離(m)
                if r < 1:  # 最小距離を1mに制限
                    r = 1
                if r > r_max:  # 最大距離を超える場合は無視
                    continue
                # スピーカーの指向性を適用
                bearing = math.atan2(grid_lon[i, j] - lon, grid_lat[i, j] - lat) * 180 / math.pi
                bearing = (bearing + 360) % 360
                power = 0
                for direction in dirs:
                    angle_diff = min(abs(bearing - direction), 360 - abs(bearing - direction))
                    if angle_diff <= 60:  # 指向性の範囲内
                        power += 10 ** ((L0 - 20 * math.log10(r)) / 10)
                power_sum[i, j] += power
    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    sound_grid = np.clip(sound_grid, L0 - 40, L0)  # 範囲外の値をクリップ
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] for i in range(Nx) for j in range(Ny) if not np.isnan(sound_grid[i, j])]
    return heat_data

# 地図の表示
st.title("音圧ヒートマップ表示")
lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01

# ズームレベルに応じた分割数
zoom_factor = 100 + (st.session_state.map_zoom - 17) * 20
grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, zoom_factor), np.linspace(lon_min, lon_max, zoom_factor))

if st.session_state.heatmap_data is None:
    st.session_state.heatmap_data = calculate_heatmap(st.session_state.speakers, 80, 500, grid_lat, grid_lon)

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

# スピーカーのマーカー
for spk in st.session_state.speakers:
    lat, lon, dirs = spk
    popup_text = f"スピーカー: ({lat:.6f}, {lon:.6f})\n方向: {dirs}"
    folium.Marker(location=[lat, lon], popup=popup_text, icon=folium.Icon(color="blue")).add_to(m)

# 計測値のマーカー
for meas in st.session_state.measurements:
    lat, lon, db = meas
    folium.Marker(location=[lat, lon], popup=f"計測値: {db} dB", icon=folium.Icon(color="green")).add_to(m)

# ヒートマップの追加
HeatMap(st.session_state.heatmap_data, radius=15, blur=20, min_opacity=0.4).add_to(m)
st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])

# 地図の中心・ズームを更新
if st_data and "center" in st_data:
    st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
if st_data and "zoom" in st_data:
    if st_data["zoom"] != st.session_state.map_zoom:
        st.session_state.map_zoom = st_data["zoom"]
        zoom_factor = 100 + (st_data["zoom"] - 17) * 20
        grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, zoom_factor), np.linspace(lon_min, lon_max, zoom_factor))
        st.session_state.heatmap_data = calculate_heatmap(st.session_state.speakers, 80, 500, grid_lat, grid_lon)

# 操作パネル
st.subheader("操作パネル")
with st.form(key="controls"):
    st.write("スピーカーの設定")
    col1, col2 = st.columns(2)

    with col1:
        new_speaker = st.text_input("新しいスピーカー (緯度,経度,方向1,方向2...)", placeholder="例: 34.2579,133.2072,0,90")
        if st.form_submit_button("スピーカーを追加"):
            try:
                parts = new_speaker.split(",")
                lat, lon = float(parts[0]), float(parts[1])
                directions = [float(d) for d in parts[2:]]
                st.session_state.speakers.append([lat, lon, directions])
                st.session_state.heatmap_data = None
                st.success(f"スピーカーを追加しました: ({lat}, {lon}), 方向: {directions}")
            except ValueError:
                st.error("入力形式が正しくありません")

    with col2:
        if st.form_submit_button("スピーカーをリセット"):
            st.session_state.speakers = []
            st.session_state.heatmap_data = None
            st.success("スピーカーをリセットしました")

    # 計測値の入力
    st.write("計測値の入力")
    new_measurement = st.text_input("計測値 (緯度,経度,デシベル)", placeholder="例: 34.2579,133.2072,75")
    if st.form_submit_button("計測値を追加"):
        try:
            lat, lon, db = map(float, new_measurement.split(","))
            st.session_state.measurements.append([lat, lon, db])
            st.success(f"計測値を追加しました: ({lat}, {lon}), {db} dB")
        except ValueError:
            st.error("入力形式が正しくありません")

    if st.form_submit_button("計測値をリセット"):
        st.session_state.measurements = []
        st.success("計測値をリセットしました")
