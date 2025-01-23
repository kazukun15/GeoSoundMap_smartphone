import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math
import branca.colormap as cm

# 初期設定の確認・設定
def initialize_session_state():
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

# 方角を角度に変換するマッピング
DIRECTION_MAPPING = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction_to_degrees(direction_str):
    """方角文字列を角度に変換"""
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    return float(direction_str)

# ヒートマップを計算
def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for speaker in speakers:
        lat, lon, directions = speaker
        speaker_coords = np.array([lat, lon])
        distances = np.sqrt(np.sum((grid_coords - speaker_coords) ** 2, axis=1)) * 111320
        distances[distances < 1] = 1  # 距離を最小1mに設定

        bearings = np.degrees(np.arctan2(grid_coords[:, 1] - lon, grid_coords[:, 0] - lat)) % 360
        power = np.zeros_like(distances)

        for direction in directions:
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            directivity_factor = np.clip(1 - angle_diff / 180, 0, 1)
            power += directivity_factor * 10 ** ((L0 - 20 * np.log10(distances)) / 10)

        power[distances > r_max] = 0
        power_sum += power.reshape(Nx, Ny)

    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    sound_grid = np.clip(sound_grid, L0 - 40, L0)
    return [
        [grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]]
        for i in range(Nx)
        for j in range(Ny)
        if not np.isnan(sound_grid[i, j])
    ]

# 地図を初期化して表示
def render_map():
    lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
    lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
    zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, zoom_factor),
        np.linspace(lon_min, lon_max, zoom_factor)
    )

    if st.session_state.heatmap_data is None and st.session_state.speakers:
        st.session_state.heatmap_data = calculate_heatmap(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat,
            grid_lon
        )

    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

    # スピーカーのマーカー
    for speaker in st.session_state.speakers:
        lat, lon, directions = speaker
        popup_text = f"""
        <div style="font-size:14px; line-height:1.5;">
            <b>スピーカー位置:</b> ({lat:.6f}, {lon:.6f})<br>
            <b>初期音圧レベル:</b> {st.session_state.L0} dB<br>
            <b>最大伝播距離:</b> {st.session_state.r_max} m<br>
            <b>方向:</b> {directions}
        </div>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="blue")
        ).add_to(m)

    # 計測値のマーカー
    for measurement in st.session_state.measurements:
        lat, lon, db = measurement
        popup_text = f"""
        <div style="font-size:14px; line-height:1.5;">
            <b>計測位置:</b> ({lat:.6f}, {lon:.6f})<br>
            <b>計測値:</b> {db} dB
        </div>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color="green")
        ).add_to(m)

    # ヒートマップ
    if st.session_state.heatmap_data:
        HeatMap(st.session_state.heatmap_data, radius=15, blur=20, min_opacity=0.4).add_to(m)

    st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])
    if st_data:
        if "center" in st_data:
            st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
        if "zoom" in st_data:
            st.session_state.map_zoom = st_data["zoom"]

# 操作パネル
def render_controls():
    st.subheader("操作パネル")
    with st.form(key="controls"):
        st.write("スピーカーの設定")
        col1, col2 = st.columns(2)

        # スピーカー追加
        with col1:
            new_speaker = st.text_input(
                "新しいスピーカー (緯度,経度,方向1,方向2...)",
                placeholder="例: 34.2579,133.2072,N,E"
            )
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

        # スピーカーリセット
        with col2:
            if st.form_submit_button("スピーカーをリセット"):
                st.session_state.speakers = []
                st.session_state.heatmap_data = None
                st.success("スピーカーをリセットしました")

        # 計測値の追加
        st.write("計測値の設定")
        new_measurement = st.text_input("計測値 (緯度,経度,デシベル)", placeholder="例: 34.2579,133.2072,75")
        if st.form_submit_button("計測値を追加"):
            try:
                lat, lon, db = map(float, new_measurement.split(","))
                st.session_state.measurements.append([lat, lon, db])
                st.success(f"計測値を追加しました: ({lat}, {lon}), {db} dB")
            except ValueError:
                st.error("入力形式が正しくありません")

        # 計測値リセット
        if st.form_submit_button("計測値をリセット"):
            st.session_state.measurements = []
            st.success("計測値をリセットしました")

        # 音圧設定
        st.write("音圧設定")
        st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
        if st.form_submit_button("更新"):
            st.session_state.heatmap_data = None
            st.success("ヒートマップを更新しました")

# 凡例の表示
def render_legend():
    st.subheader("音圧レベルの凡例")
    colormap = cm.LinearColormap(
        colors=["blue", "green", "yellow", "red"],
        vmin=st.session_state.L0 - 40,
        vmax=st.session_state.L0,
        caption="音圧レベル (dB)"
    )
    st.markdown(f'<div style="width:100%; text-align:center;">{colormap._repr_html_()}</div>', unsafe_allow_html=True)

# メイン実行
initialize_session_state()
render_map()
render_controls()
render_legend()
