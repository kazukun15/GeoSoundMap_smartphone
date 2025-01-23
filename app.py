import streamlit as st 
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math
from skimage import measure
import pandas as pd
import branca.colormap as cm
import io  # エクスポート用バッファ

# ─────────────────────────────────────────────────────────────────────────
# セッション初期設定
# ─────────────────────────────────────────────────────────────────────────
if "map_center" not in st.session_state:
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 14  

if "speakers" not in st.session_state:
    # speakers: [[lat, lon, [direction1, direction2, ...]], ...]
    st.session_state.speakers = [
        [34.25741795269067, 133.20450105700033, [0.0, 90.0]]  # 初期サンプル
    ]

if "measurements" not in st.session_state:
    # measurements: [[lat, lon, dB], ...]
    st.session_state.measurements = []

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None

if "contours" not in st.session_state:
    st.session_state.contours = {"60dB": [], "80dB": []}

if "L0" not in st.session_state:
    st.session_state.L0 = 80  # 初期音圧レベル(dB)

if "r_max" not in st.session_state:
    st.session_state.r_max = 500  # 最大伝播距離

# 方角文字列→角度に変換
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
        st.error(f"方向の指定 '{direction_str}' を数値に変換できません。0度として扱います。")
        return 0.0

# ─────────────────────────────────────────────────────────────────────────
# ヒートマップ & 等高線の計算
# ─────────────────────────────────────────────────────────────────────────
def calculate_heatmap_and_contours(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))

    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for spk in speakers:
        lat, lon, dirs = spk
        spk_coords = np.array([lat, lon])

        # lat/lon差分から距離(メートル換算)
        distances = np.sqrt(np.sum((grid_coords - spk_coords)**2, axis=1)) * 111320
        distances[distances < 1] = 1  # 最小1m

        # 方位計算
        bearings = np.degrees(np.arctan2(
            grid_coords[:,1] - lon,
            grid_coords[:,0] - lat
        )) % 360

        power = np.zeros_like(distances)
        for direction in dirs:
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            directivity_factor = np.clip(1 - angle_diff / 180, 0, 1)

            p = 10 ** ((L0 - 20 * np.log10(distances)) / 10)
            power += directivity_factor * p

        power[distances > r_max] = 0
        power_sum += power.reshape(Nx, Ny)

    sound_grid = np.full_like(power_sum, np.nan, dtype=float)
    non_zero_mask = (power_sum > 0)
    sound_grid[non_zero_mask] = 10 * np.log10(power_sum[non_zero_mask])

    sound_grid = np.clip(sound_grid, L0 - 40, L0)

    # ヒートマップ用データ
    heat_data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i,j]
            if not np.isnan(val):
                heat_data.append([grid_lat[i,j], grid_lon[i,j], val])

    # 等高線(60dB/80dB)
    fill_grid = np.where(np.isnan(sound_grid), -9999, sound_grid)
    contours = {"60dB": [], "80dB": []}
    levels = {"60dB": 60, "80dB": 80}

    from skimage import measure
    for key, level in levels.items():
        raw_contours = measure.find_contours(fill_grid, level=level)
        for contour in raw_contours:
            lat_lon_contour = []
            for y, x in contour:
                iy, ix = int(round(y)), int(round(x))
                if 0 <= iy < Nx and 0 <= ix < Ny:
                    lat_lon_contour.append((grid_lat[iy, ix], grid_lon[iy, ix]))
            if len(lat_lon_contour) > 1:
                contours[key].append(lat_lon_contour)

    return heat_data, contours

# ─────────────────────────────────────────────────────────────────────────
# CSV読み込み (インポート)  「種別, 緯度, 経度, データ1, データ2, データ3」形式
# ─────────────────────────────────────────────────────────────────────────
def load_csv(file):
    """
    CSVの列: 「種別」, 「緯度」, 「経度」, 「データ1」, 「データ2」, 「データ3」
    ・種別 = "スピーカ" の場合
       -> (緯度, 経度, [方向1, 方向2, 方向3])として speakers[]へ
    ・種別 = "計測値" の場合
       -> (緯度, 経度, dB値)として measurements[]へ
    """
    try:
        df = pd.read_csv(file)
        speakers = []
        measurements = []

        for _, row in df.iterrows():
            # 各列を取得
            item_type = row.get("種別", None)
            lat = row.get("緯度", None)
            lon = row.get("経度", None)
            d1 = row.get("データ1", None)
            d2 = row.get("データ2", None)
            d3 = row.get("データ3", None)

            # 緯度・経度がNaNならスキップ
            if pd.isna(lat) or pd.isna(lon):
                continue

            # 数値化
            try:
                lat = float(lat)
                lon = float(lon)
            except ValueError:
                st.warning("緯度/経度の値が不正です。該当行をスキップします。")
                continue

            if item_type == "スピーカ":
                # データ列を方角として解釈
                directions = []
                for d in [d1, d2, d3]:
                    if not pd.isna(d) and str(d).strip() != "":
                        directions.append(parse_direction_to_degrees(str(d)))
                speakers.append([lat, lon, directions])

            elif item_type == "計測値":
                # データ1 を dBとして扱う (データ2, データ3は無視または空欄)
                db_val = 0.0
                if not pd.isna(d1):
                    try:
                        db_val = float(d1)
                    except ValueError:
                        st.error(f"計測値を数値に変換できませんでした: {d1}")
                        db_val = 0.0
                measurements.append([lat, lon, db_val])

        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込み中にエラー: {e}")
        return [], []

# ─────────────────────────────────────────────────────────────────────────
# CSV書き出し (エクスポート)  「種別, 緯度, 経度, データ1, データ2, データ3」形式
# ─────────────────────────────────────────────────────────────────────────
def export_to_csv(speakers, measurements):
    """
    1つのCSVに全スピーカ(種別='スピーカ')と全計測値(種別='計測値')をまとめる。
    - 「種別」: スピーカ or 計測値
    - 「緯度」: lat
    - 「経度」: lon
    - 「データ1~3」: スピーカの場合は方角(最大3つ)/計測値の場合はdBだけデータ1へ
    """
    columns = ["種別","緯度","経度","データ1","データ2","データ3"]
    rows = []

    # スピーカ
    for lat, lon, dirs in speakers:
        row = {
            "種別": "スピーカ",
            "緯度": lat,
            "経度": lon,
            "データ1": dirs[0] if len(dirs) > 0 else "",
            "データ2": dirs[1] if len(dirs) > 1 else "",
            "データ3": dirs[2] if len(dirs) > 2 else "",
        }
        rows.append(row)

    # 計測値
    for lat_m, lon_m, db_m in measurements:
        row = {
            "種別": "計測値",
            "緯度": lat_m,
            "経度": lon_m,
            "データ1": db_m,  # データ1列にdBを入れる
            "データ2": "",
            "データ3": "",
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────
# Streamlit表示開始
# ─────────────────────────────────────────────────────────────────────────
st.title("音圧ヒートマップ表示 - 防災スピーカーの非可聴域検出 (CSV列: 種別/緯度/経度/データ1..3)")

# 地図用グリッド
lat_min = st.session_state.map_center[0] - 0.01
lat_max = st.session_state.map_center[0] + 0.01
lon_min = st.session_state.map_center[1] - 0.01
lon_max = st.session_state.map_center[1] + 0.01

zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
grid_lat, grid_lon = np.meshgrid(
    np.linspace(lat_min, lat_max, zoom_factor),
    np.linspace(lon_min, lon_max, zoom_factor)
)
# 必要に応じて転置
grid_lat = grid_lat.T
grid_lon = grid_lon.T

# まだ計算していなければ、一度だけヒートマップ計算
if st.session_state.heatmap_data is None and st.session_state.speakers:
    st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
        st.session_state.speakers,
        st.session_state.L0,
        st.session_state.r_max,
        grid_lat,
        grid_lon
    )

# Foliumマップの生成
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom
)

# スピーカーのマーカー
for lat_s, lon_s, dirs in st.session_state.speakers:
    popup_text = (
        f"<b>スピーカ</b>: ({lat_s:.6f}, {lon_s:.6f})<br>"
        f"<b>初期音圧:</b> {st.session_state.L0} dB<br>"
        f"<b>最大伝播距離:</b> {st.session_state.r_max} m<br>"
        f"<b>方向:</b> {dirs}"
    )
    folium.Marker(
        location=[lat_s, lon_s],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="blue")
    ).add_to(m)

# 計測値のマーカー
for lat_m, lon_m, db_m in st.session_state.measurements:
    popup_text = (
        f"<b>計測位置:</b> ({lat_m:.6f}, {lon_m:.6f})<br>"
        f"<b>計測値:</b> {db_m:.2f} dB"
    )
    folium.Marker(
        location=[lat_m, lon_m],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="green")
    ).add_to(m)

# ヒートマップ
if st.session_state.heatmap_data:
    HeatMap(
        st.session_state.heatmap_data,
        radius=15,
        blur=20,
        min_opacity=0.4
    ).add_to(m)

# 等高線(60dB緑,80dB赤)
for contour_60 in st.session_state.contours["60dB"]:
    folium.PolyLine(locations=contour_60, color="green", weight=2).add_to(m)

for contour_80 in st.session_state.contours["80dB"]:
    folium.PolyLine(locations=contour_80, color="red", weight=2).add_to(m)

# 地図表示
st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])
if st_data:
    if "center" in st_data:
        st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    if "zoom" in st_data:
        st.session_state.map_zoom = st_data["zoom"]

# ─────────────────────────────────────────────────────────────────────────
# 操作パネル
# ─────────────────────────────────────────────────────────────────────────
st.subheader("操作パネル")

uploaded_file = st.file_uploader("CSVをアップロード (種別/緯度/経度/データ1..3列)", type=["csv"])
if uploaded_file:
    speakers_loaded, measurements_loaded = load_csv(uploaded_file)
    if speakers_loaded:
        st.session_state.speakers.extend(speakers_loaded)
    if measurements_loaded:
        st.session_state.measurements.extend(measurements_loaded)
    st.success("CSVを読み込みました。“更新”ボタンでヒートマップに反映可能。")

# スピーカー追加
new_speaker = st.text_input("新しいスピーカ (緯度,経度,方向1,方向2,方向3)",
                            placeholder="例: 34.2579,133.2072,N,E,SE")
if st.button("スピーカを追加"):
    try:
        items = new_speaker.split(",")
        lat_spk = float(items[0])
        lon_spk = float(items[1])
        dirs_spk = []
        for d in items[2:]:
            dirs_spk.append(parse_direction_to_degrees(d))
        st.session_state.speakers.append([lat_spk, lon_spk, dirs_spk])
        st.session_state.heatmap_data = None
        st.success(f"スピーカを追加: ({lat_spk}, {lon_spk}), 方向={dirs_spk}")
    except:
        st.error("入力形式が不正です。(緯度,経度,方向...)の形式で入力してください。")

# 計測値追加
new_measurement = st.text_input("計測値 (緯度,経度,dB)",
                                placeholder="例: 34.2578,133.2075,75")
if st.button("計測値を追加"):
    try:
        items = new_measurement.split(",")
        lat_m = float(items[0])
        lon_m = float(items[1])
        db_m = float(items[2])
        st.session_state.measurements.append([lat_m, lon_m, db_m])
        st.success(f"計測値を追加: ({lat_m}, {lon_m}), {db_m} dB")
    except:
        st.error("入力形式が不正です。(緯度,経度,dB)の形式で入力してください。")

# リセットボタン
if st.button("スピーカをリセット"):
    st.session_state.speakers = []
    st.session_state.heatmap_data = None
    st.session_state.contours = {"60dB": [], "80dB": []}
    st.success("スピーカをリセットしました")

if st.button("計測値をリセット"):
    st.session_state.measurements = []
    st.success("計測値をリセットしました")

# 音圧/距離スライダー
st.session_state.L0 = st.slider("初期音圧レベル(dB)", 50, 100, st.session_state.L0)
st.session_state.r_max = st.slider("最大伝播距離(m)", 100, 2000, st.session_state.r_max)

# 更新ボタン: ヒートマップ再計算
if st.button("更新"):
    if st.session_state.speakers:
        st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat,
            grid_lon
        )
        st.success("ヒートマップと等高線を再計算しました")
    else:
        st.error("スピーカがありません。追加してください。")

# ─────────────────────────────────────────────────────────────────────────
# CSVのエクスポート
# ─────────────────────────────────────────────────────────────────────────
st.subheader("CSVのエクスポート (種別/緯度/経度/データ1..3形式)")

if st.button("CSVをエクスポート"):
    csv_data = export_to_csv(st.session_state.speakers, st.session_state.measurements)
    st.download_button(
        label="CSVファイルのダウンロード",
        data=csv_data,
        file_name="sound_map_data.csv",
        mime="text/csv"
    )

# ─────────────────────────────────────────────────────────────────────────
# 凡例バーの表示
# ─────────────────────────────────────────────────────────────────────────
st.subheader("音圧レベルの凡例")
color_scale = cm.LinearColormap(
    colors=["blue", "green", "yellow", "red"],
    vmin=st.session_state.L0 - 40,
    vmax=st.session_state.L0,
    caption="音圧レベル (dB)"
)
st.markdown(
    f'<div style="width:100%; text-align:center;">{color_scale._repr_html_()}</div>',
    unsafe_allow_html=True
)
