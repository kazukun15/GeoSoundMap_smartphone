import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math
import pandas as pd
from skimage import measure
import branca.colormap as cm
import io

# ─────────────────────────────────────────────────────────────────────────
# セッション初期設定
# ─────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────
# 方角の変換
# ─────────────────────────────────────────────────────────────────────────
DIRECTION_MAPPING = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction_to_degrees(direction_str):
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    try:
        return float(direction_str)
    except ValueError:
        st.error(f"方向 '{direction_str}' の変換に失敗しました。0度に設定します。")
        return 0.0

# ─────────────────────────────────────────────────────────────────────────
# CSV読み込みとエクスポート
# ─────────────────────────────────────────────────────────────────────────
def load_csv(file):
    try:
        df = pd.read_csv(file)
        speakers, measurements = [], []
        for _, row in df.iterrows():
            if not pd.isna(row.get("スピーカー緯度")) and not pd.isna(row.get("スピーカー経度")):
                lat, lon = row["スピーカー緯度"], row["スピーカー経度"]
                directions = [
                    parse_direction_to_degrees(row.get(f"方向{i}", ""))
                    for i in range(1, 4)
                    if not pd.isna(row.get(f"方向{i}"))
                ]
                speakers.append([lat, lon, directions])

            if not pd.isna(row.get("計測位置緯度")) and not pd.isna(row.get("計測位置経度")):
                lat, lon, db = row["計測位置緯度"], row["計測位置経度"], row.get("計測デシベル", 0)
                measurements.append([lat, lon, float(db)])
        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込み中のエラー: {e}")
        return [], []

def export_to_csv(data, columns):
    df = pd.DataFrame(data, columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────
# 地図とヒートマップの計算
# ─────────────────────────────────────────────────────────────────────────
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

    sound_grid = np.full_like(power_sum, np.nan, dtype=float)
    sound_grid[power_sum > 0] = 10 * np.log10(power_sum[power_sum > 0])
    sound_grid = np.clip(sound_grid, L0 - 40, L0)

    heat_data = [
        [grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]]
        for i in range(Nx)
        for j in range(Ny)
        if not np.isnan(sound_grid[i, j])
    ]
    return heat_data, {"60dB": [], "80dB": []}

# ─────────────────────────────────────────────────────────────────────────
# Streamlitアプリケーションのメイン
# ─────────────────────────────────────────────────────────────────────────
st.title("防災スピーカー音圧可視化マップ")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
if uploaded_file:
    new_speakers, new_measurements = load_csv(uploaded_file)
    if new_speakers:
        st.session_state.speakers.extend(new_speakers)
    if new_measurements:
        st.session_state.measurements.extend(new_measurements)
    st.success("CSVデータを読み込みました！")

lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
grid_lat, grid_lon = np.meshgrid(
    np.linspace(lat_min, lat_max, 100),
    np.linspace(lon_min, lon_max, 100)
)

if st.session_state.heatmap_data is None:
    st.session_state.heatmap_data, _ = calculate_heatmap_and_contours(
        st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
    )

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
for spk in st.session_state.speakers:
    lat, lon, dirs = spk
    popup_text = f"スピーカー: ({lat:.6f}, {lon:.6f})<br>方向: {dirs}"
    folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text)).add_to(m)

if st.session_state.heatmap_data:
    HeatMap(st.session_state.heatmap_data).add_to(m)

st_data = st_folium(m, width=700, height=500)

if st_data:
    if "center" in st_data:
        st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    if "zoom" in st_data:
        st.session_state.map_zoom = st_data["zoom"]

# 操作パネル
st.subheader("操作パネル")
# スピーカーの手動追加
new_speaker = st.text_input("新しいスピーカー (緯度,経度,方向1,方向2...)", placeholder="例: 34.2579,133.2072,N,E")
if st.button("スピーカーを追加"):
    try:
        parts = new_speaker.split(",")
        lat, lon = float(parts[0]), float(parts[1])
        directions = [parse_direction_to_degrees(dir_str) for dir_str in parts[2:]]
        st.session_state.speakers.append([lat, lon, directions])
        st.session_state.heatmap_data = None  # ヒートマップ再計算をトリガー
        st.success(f"スピーカーを追加しました: ({lat}, {lon}), 方向: {directions}")
    except ValueError:
        st.error("入力形式が正しくありません。(緯度,経度,方向...) の形式で入力してください。")

# スピーカーリセット
if st.button("スピーカーをリセット"):
    st.session_state.speakers = []
    st.session_state.heatmap_data = None
    st.session_state.contours = {"60dB": [], "80dB": []}
    st.success("スピーカーをリセットしました")

# 音圧設定
st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)

# 計測値の手動追加
new_measurement = st.text_input("計測値 (緯度,経度,デシベル)", placeholder="例: 34.2579,133.2072,75")
if st.button("計測値を追加"):
    try:
        lat, lon, db = map(float, new_measurement.split(","))
        st.session_state.measurements.append([lat, lon, db])
        st.success(f"計測値を追加しました: ({lat}, {lon}), {db} dB")
    except ValueError:
        st.error("入力形式が正しくありません。(緯度,経度,デシベル) の形式で入力してください。")

# 計測値リセット
if st.button("計測値をリセット"):
    st.session_state.measurements = []
    st.success("計測値をリセットしました")

# ヒートマップ更新
if st.button("ヒートマップを更新"):
    if st.session_state.speakers:
        st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat,
            grid_lon
        )
        st.success("ヒートマップを更新しました")
    else:
        st.error("スピーカーがありません。スピーカーを追加してください。")

# CSVエクスポート
st.subheader("データのエクスポート")
col1, col2 = st.columns(2)

with col1:
    csv_data_speakers = export_to_csv(
        st.session_state.speakers,
        ["スピーカー緯度", "スピーカー経度", "方向1", "方向2", "方向3"]
    )
    st.download_button(
        label="スピーカーCSVのダウンロード",
        data=csv_data_speakers,
        file_name="speakers.csv",
        mime="text/csv"
    )

with col2:
    csv_data_measurements = export_to_csv(
        st.session_state.measurements,
        ["計測位置緯度", "計測位置経度", "計測デシベル"]
    )
    st.download_button(
        label="計測値CSVのダウンロード",
        data=csv_data_measurements,
        file_name="measurements.csv",
        mime="text/csv"
    )

# 凡例の表示
st.subheader("音圧レベルの凡例")
colormap = cm.LinearColormap(
    colors=["blue", "green", "yellow", "red"],
    vmin=st.session_state.L0 - 40,
    vmax=st.session_state.L0,
    caption="音圧レベル (dB)"
)
st.markdown(f'<div style="width:100%; text-align:center;">{colormap._repr_html_()}</div>', unsafe_allow_html=True)

