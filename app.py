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

# 計測値の理論値を計算する関数
def calculate_theoretical_value(speakers, L0, r_max, lat, lon):
    """単一地点での理論音圧レベルを計算"""
    total_power = 0.0
    for spk in speakers:
        s_lat, s_lon, dirs = spk
        distance = math.sqrt((lat - s_lat)**2 + (lon - s_lon)**2) * 111320  # 距離をメートルで計算
        if distance < 1:  # 最小距離1m
            distance = 1
        if distance > r_max:  # 最大距離を超えた場合は影響なし
            continue
        bearing = (math.degrees(math.atan2(lon - s_lon, lat - s_lat)) + 360) % 360
        power = 0.0
        for direction in dirs:
            angle_diff = abs(bearing - direction) % 360
            angle_diff = min(angle_diff, 360 - angle_diff)
            directivity_factor = max(1 - angle_diff / 180, 0)  # 指向性の減衰
            power += directivity_factor * 10 ** ((L0 - 20 * math.log10(distance)) / 10)
        total_power += power
    if total_power > 0:
        return 10 * math.log10(total_power)  # 音圧レベル(dB)を計算
    return None  # 範囲外の場合

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
    popup_text = f"スピーカー: ({lat:.6f}, {lon:.6f})<br>初期音圧レベル: {st.session_state.L0} dB<br>最大伝播距離: {st.session_state.r_max} m"
    folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300), icon=folium.Icon(color="blue")).add_to(m)

# 計測値のマーカー
for meas in st.session_state.measurements:
    lat, lon, db = meas
    theoretical_value = calculate_theoretical_value(
        st.session_state.speakers, st.session_state.L0, st.session_state.r_max, lat, lon
    )
    theoretical_text = f"{theoretical_value:.2f} dB" if theoretical_value is not None else "範囲外"
    popup_text = f"""
    <div style="font-size:14px; line-height:1.5;">
        <b>計測位置:</b> ({lat:.6f}, {lon:.6f})<br>
        <b>計測値:</b> {db:.2f} dB<br>
        <b>理論値:</b> {theoretical_text}
    </div>
    """
    folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300), icon=folium.Icon(color="green")).add_to(m)

# ヒートマップの追加
if st.session_state.heatmap_data:
    HeatMap(st.session_state.heatmap_data, radius=15, blur=20, min_opacity=0.4).add_to(m)

# 等高線の追加
for contour in st.session_state.contours["60dB"]:
    folium.PolyLine(locations=contour, color="blue", weight=2, tooltip="60dB").add_to(m)

for contour in st.session_state.contours["80dB"]:
    folium.PolyLine(locations=contour, color="red", weight=2, tooltip="80dB").add_to(m)

# 地図を表示
st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])

# 地図の中心・ズームを更新
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

    # スピーカー追加
    with col1:
        new_speaker = st.text_input("新しいスピーカー (緯度,経度,方向1,方向2...)", placeholder="例: 34.2579,133.2072,N,E")
        if st.form_submit_button("スピーカーを追加"):
            try:
                parts = new_speaker.split(",")
                lat, lon = float(parts[0]), float(parts[1])
                directions = [parse_direction_to_degrees(d) for d in parts[2:]]
                st.session_state.speakers.append([lat, lon, directions])
                st.session_state.heatmap_data = None  # 再計算フラグを設定
                st.success(f"スピーカーを追加しました: ({lat}, {lon}), 方向: {directions}")
            except ValueError:
                st.error("入力形式が正しくありません")

    # スピーカーリセット
    with col2:
        if st.form_submit_button("スピーカーをリセット"):
            st.session_state.speakers = []  # スピーカーをクリア
            st.session_state.heatmap_data = None  # ヒートマップもクリア
            st.session_state.contours = {"60dB": [], "80dB": []}
            st.success("スピーカーをリセットしました")

    # 音圧設定
    st.write("音圧設定")
    st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
    st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)

    # 計測値の設定
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

    # ヒートマップ更新
    if st.form_submit_button("更新"):
        if st.session_state.speakers:
            st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
                st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
            )
            st.success("ヒートマップと等高線を更新しました")
        else:
            st.error("スピーカーが存在しません。")


# 凡例バーを表示
st.subheader("音圧レベルの凡例")
colormap = cm.LinearColormap(
    colors=["blue", "green", "yellow", "red"],
    vmin=st.session_state.L0 - 40,
    vmax=st.session_state.L0,
    caption="音圧レベル (dB)"
)
st.markdown(f'<div style="width:100%; text-align:center;">{colormap._repr_html_()}</div>', unsafe_allow_html=True)
