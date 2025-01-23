import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import math
import branca.colormap as cm

# 初期設定
def initialize_session_state():
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
    if "l0" not in st.session_state:
        st.session_state.l0 = 80
    if "r_max" not in st.session_state:
        st.session_state.r_max = 500

initialize_session_state()

# 方角の変換
direction_mapping = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction_to_degrees(direction_str):
    direction_str = direction_str.strip().upper()
    return direction_mapping.get(direction_str, float(direction_str))

# 音圧ヒートマップの計算
def calculate_heatmap(speakers, l0, r_max, grid_lat, grid_lon):
    nx, ny = grid_lat.shape
    power_sum = np.zeros((nx, ny))
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for lat, lon, directions in speakers:
        spk_coords = np.array([lat, lon])
        distances = np.sqrt(np.sum((grid_coords - spk_coords)**2, axis=1)) * 111320
        distances[distances < 1] = 1
        bearings = np.degrees(np.arctan2(grid_coords[:, 1] - lon, grid_coords[:, 0] - lat)) % 360
        power = np.zeros_like(distances)

        for direction in directions:
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            directivity_factor = np.clip(1 - angle_diff / 180, 0, 1)
            power += directivity_factor * 10**((l0 - 20 * np.log10(distances)) / 10)

        power[distances > r_max] = 0
        power_sum += power.reshape(nx, ny)

    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    sound_grid = np.clip(sound_grid, l0 - 40, l0)
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] for i in range(nx) for j in range(ny) if not np.isnan(sound_grid[i, j])]
    return heat_data

# CSVデータの読み込み
def import_csv(file):
    try:
        df = pd.read_csv(file)
        if 'type' not in df.columns or 'lat' not in df.columns or 'lon' not in df.columns:
            st.error("CSVファイルに必要な列 ('type', 'lat', 'lon') が含まれていません。")
            return
        speakers = df[df['type'] == 'speaker'][['lat', 'lon', 'directions']].values.tolist()
        measurements = df[df['type'] == 'measurement'][['lat', 'lon', 'value']].values.tolist()
        st.session_state.speakers = [
            [row[0], row[1], [parse_direction_to_degrees(d) for d in str(row[2]).split(",")]]
            for row in speakers
        ]
        st.session_state.measurements = measurements
        st.session_state.heatmap_data = None
        st.success("CSVファイルのインポートが完了しました。")
    except Exception as e:
        st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")

# 地図の表示
def render_map():
    lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
    lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
    grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))

    if st.session_state.heatmap_data is None and st.session_state.speakers:
        st.session_state.heatmap_data = calculate_heatmap(
            st.session_state.speakers, st.session_state.l0, st.session_state.r_max, grid_lat, grid_lon
        )

    map_obj = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

    # スピーカーのマーカー
    for lat, lon, directions in st.session_state.speakers:
        popup_text = f"""
        <b>スピーカー位置:</b> ({lat:.6f}, {lon:.6f})<br>
        <b>初期音圧レベル:</b> {st.session_state.l0} dB<br>
        <b>最大伝播距離:</b> {st.session_state.r_max} m
        """
        folium.Marker([lat, lon], popup=folium.Popup(popup_text, max_width=300), icon=folium.Icon(color="blue")).add_to(map_obj)

    # 計測値のマーカー
    for lat, lon, value in st.session_state.measurements:
        popup_text = f"""
        <b>計測位置:</b> ({lat:.6f}, {lon:.6f})<br>
        <b>計測値:</b> {value:.2f} dB
        """
        folium.Marker([lat, lon], popup=folium.Popup(popup_text, max_width=300), icon=folium.Icon(color="green")).add_to(map_obj)

    # ヒートマップ
    if st.session_state.heatmap_data:
        HeatMap(st.session_state.heatmap_data, radius=15, blur=20, min_opacity=0.4).add_to(map_obj)

    return st_folium(map_obj, width=700, height=500)

# 操作パネル
def render_controls():
    st.subheader("操作パネル")

    # CSVファイルのインポート
    csv_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    if csv_file:
        import_csv(csv_file)

    # スピーカーの追加
    new_speaker = st.text_input("スピーカー (緯度,経度,方向1,方向2...)", placeholder="例: 34.2579,133.2072,N,E")
    if st.button("スピーカーを追加"):
        try:
            parts = new_speaker.split(",")
            lat, lon = float(parts[0]), float(parts[1])
            directions = [parse_direction_to_degrees(d) for d in parts[2:]]
            st.session_state.speakers.append([lat, lon, directions])
            st.session_state.heatmap_data = None
            st.success("スピーカーを追加しました。")
        except ValueError:
            st.error("入力形式が正しくありません。")

    # 計測値の追加
    new_measurement = st.text_input("計測値 (緯度,経度,デシベル)", placeholder="例: 34.2579,133.2072,75")
    if st.button("計測値を追加"):
        try:
            lat, lon, value = map(float, new_measurement.split(","))
            st.session_state.measurements.append([lat, lon, value])
            st.success("計測値を追加しました。")
        except ValueError:
            st.error("入力形式が正しくありません。")

    # 音圧設定
    st.session_state.l0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.l0)
    st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)

# メイン
st.title("音圧ヒートマップシステム")
render_controls()
render_map()
