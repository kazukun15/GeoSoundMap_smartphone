import streamlit as st 
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math
from skimage import measure
import pandas as pd
import branca.colormap as cm
import io  # 追加：エクスポート用

# ─────────────────────────────────────────────────────────────────────────
# セッション初期設定
# ─────────────────────────────────────────────────────────────────────────
if "map_center" not in st.session_state:
    # 地図の中心座標(初期値)
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]

if "map_zoom" not in st.session_state:
    # 地図の初期ズームレベル
    st.session_state.map_zoom = 14  

if "speakers" not in st.session_state:
    # スピーカー情報 [ [lat, lon, [方向角度リスト]], ... ]
    st.session_state.speakers = [
        [34.25741795269067, 133.20450105700033, [0.0, 90.0]]  # デフォルトサンプル
    ]

if "measurements" not in st.session_state:
    # 計測値リスト [ [lat, lon, dB], ... ]
    st.session_state.measurements = []

if "heatmap_data" not in st.session_state:
    # ヒートマップ計算結果(座標+音圧レベル)
    st.session_state.heatmap_data = None

if "contours" not in st.session_state:
    # 等高線のリスト {"60dB": [ [ (lat, lon), ... ], ... ], "80dB": [ [...], ... ]}
    st.session_state.contours = {"60dB": [], "80dB": []}

if "L0" not in st.session_state:
    # 初期音圧レベル(dB)
    st.session_state.L0 = 80  

if "r_max" not in st.session_state:
    # 最大伝播距離(m)
    st.session_state.r_max = 500  

# ─────────────────────────────────────────────────────────────────────────
# 方角の変換定義
# ─────────────────────────────────────────────────────────────────────────
DIRECTION_MAPPING = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction_to_degrees(direction_str):
    """方角文字列(N, S, E, W, NE...など)を角度に変換し、数値ならそのままfloatに変換。"""
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    try:
        return float(direction_str)  # それ以外は数値とみなす
    except ValueError:
        st.error(f"方向の指定 '{direction_str}' を数値に変換できません。0度にします。")
        return 0.0

# ─────────────────────────────────────────────────────────────────────────
# ヒートマップ & 等高線の計算
# ─────────────────────────────────────────────────────────────────────────
def calculate_heatmap_and_contours(speakers, L0, r_max, grid_lat, grid_lon):
    """
    複数スピーカーとグリッドから音圧ヒートマップと
    60dB / 80dB の等高線座標を計算して返す。
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))

    # グリッド座標(2次元)を一列化し、各点との距離をまとめて計算
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for spk in speakers:
        lat, lon, dirs = spk
        spk_coords = np.array([lat, lon])

        # lat/lon差分から距離(メートル換算)
        # 1度 ≈ 111,320mとして近似
        distances = np.sqrt(np.sum((grid_coords - spk_coords) ** 2, axis=1)) * 111320
        distances[distances < 1] = 1  # 最小1mに補正

        # 方位計算：bearing(北=0°, 東=90°, 南=180°, 西=270°)
        bearings = np.degrees(np.arctan2(
            grid_coords[:, 1] - lon,
            grid_coords[:, 0] - lat
        )) % 360

        power = np.zeros_like(distances)
        # 指向性(複数方向)の合成
        for direction in dirs:
            # 角度差(0~180)
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)

            # 角度差に応じた減衰 (ここでは最大1倍、180度差で0倍)
            directivity_factor = np.clip(1 - angle_diff / 180, 0, 1)

            # 音圧レベル L = L0 - 20log10(r)
            # パワー比 ~ 10^(L/10)
            p = 10 ** ((L0 - 20 * np.log10(distances)) / 10)
            power += directivity_factor * p

        # 最大伝播距離を越えた点は0に
        power[distances > r_max] = 0

        # 重ね合わせ
        power_sum += power.reshape(Nx, Ny)

    # dBスケールに変換(パワーが0のところはNaNに)
    sound_grid = np.full_like(power_sum, np.nan, dtype=float)
    non_zero_mask = (power_sum > 0)
    sound_grid[non_zero_mask] = 10 * np.log10(power_sum[non_zero_mask])

    # 表示レンジを L0-40 ~ L0 にクリップ
    sound_grid = np.clip(sound_grid, L0 - 40, L0)

    # ヒートマップ用データをリスト化
    heat_data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                heat_data.append([grid_lat[i, j], grid_lon[i, j], val])

    # 等高線(60dB / 80dB)を skimage.measure.find_contours で抽出
    # NaNがあると処理できないので、NaN部分を-9999等で埋める
    fill_grid = np.where(np.isnan(sound_grid), -9999, sound_grid)
    contours = {"60dB": [], "80dB": []}
    levels = {"60dB": 60, "80dB": 80}

    for key, level in levels.items():
        raw_contours = measure.find_contours(fill_grid, level=level)
        for contour in raw_contours:
            lat_lon_contour = []
            for y, x in contour:
                # float→intにすると座標がずれてしまうので丸める
                iy, ix = int(round(y)), int(round(x))
                # 配列の範囲チェック
                if 0 <= iy < Nx and 0 <= ix < Ny:
                    lat_lon_contour.append((grid_lat[iy, ix], grid_lon[iy, ix]))
            if len(lat_lon_contour) > 1:
                contours[key].append(lat_lon_contour)

    return heat_data, contours

# ─────────────────────────────────────────────────────────────────────────
# CSV読み込み(インポート機能)
# ─────────────────────────────────────────────────────────────────────────
def load_csv(file):
    """CSVからスピーカー情報と計測値を読み込み、リストにして返す。"""
    try:
        df = pd.read_csv(file)
        speakers = []
        measurements = []
        for _, row in df.iterrows():
            # スピーカー情報があれば取得
            if not pd.isna(row.get("スピーカー緯度")) and not pd.isna(row.get("スピーカー経度")):
                lat, lon = row["スピーカー緯度"], row["スピーカー経度"]
                directions = []
                # 方向1～3列があれば変換
                for i in range(1, 4):
                    col = f"方向{i}"
                    if col in row and not pd.isna(row[col]):
                        directions.append(parse_direction_to_degrees(row[col]))
                speakers.append([lat, lon, directions])

            # 計測位置があれば取得
            if not pd.isna(row.get("計測位置緯度")) and not pd.isna(row.get("計測位置経度")):
                lat_m, lon_m = row["計測位置緯度"], row["計測位置経度"]
                db_m = row.get("計測デシベル", None)
                if pd.isna(db_m):
                    db_m = 0.0
                measurements.append([lat_m, lon_m, float(db_m)])

        return speakers, measurements
    except Exception as e:
        st.error(f"CSVの読み込みに失敗しました: {e}")
        return [], []

# ─────────────────────────────────────────────────────────────────────────
# CSVエクスポート(ダウンロードボタン)
# ─────────────────────────────────────────────────────────────────────────
def export_speakers_to_csv(speakers):
    """スピーカー情報をCSV形式のバイト列として返す"""
    # 最大で方向は3つ想定
    columns = ["スピーカー緯度", "スピーカー経度", "方向1", "方向2", "方向3"]
    rows = []
    for spk in speakers:
        lat, lon, dirs = spk
        row = {
            "スピーカー緯度": lat,
            "スピーカー経度": lon,
        }
        for i in range(3):
            colname = f"方向{i+1}"
            row[colname] = dirs[i] if i < len(dirs) else ""
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

def export_measurements_to_csv(measurements):
    """計測値リストをCSV形式のバイト列として返す"""
    columns = ["計測位置緯度", "計測位置経度", "計測デシベル"]
    df = pd.DataFrame(measurements, columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────
# Streamlit表示開始
# ─────────────────────────────────────────────────────────────────────────
st.title("音圧ヒートマップ表示 - 防災スピーカーの非可聴域検出")

# 地図用のグリッドを作成
lat_min = st.session_state.map_center[0] - 0.01
lat_max = st.session_state.map_center[0] + 0.01
lon_min = st.session_state.map_center[1] - 0.01
lon_max = st.session_state.map_center[1] + 0.01

# ズームに応じたグリッド密度
zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
grid_lat, grid_lon = np.meshgrid(
    np.linspace(lat_min, lat_max, zoom_factor),
    np.linspace(lon_min, lon_max, zoom_factor)
)
grid_lat = grid_lat.T  # shape合わせのため転置する場合もあるが、好みで
grid_lon = grid_lon.T

# まだヒートマップを計算していない場合、一度だけ計算する
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

# スピーカーのマーカーを追加
for spk in st.session_state.speakers:
    lat_s, lon_s, dirs = spk
    popup_text = (
        f"<b>スピーカー</b>: ({lat_s:.6f}, {lon_s:.6f})<br>"
        f"<b>初期音圧レベル:</b> {st.session_state.L0} dB<br>"
        f"<b>最大伝播距離:</b> {st.session_state.r_max} m<br>"
        f"<b>方向:</b> {dirs}"
    )
    folium.Marker(
        location=[lat_s, lon_s],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="blue")
    ).add_to(m)

# 計測値のマーカーを追加
for meas in st.session_state.measurements:
    lat_m, lon_m, db_m = meas
    popup_text = (
        f"<b>計測位置:</b> ({lat_m:.6f}, {lon_m:.6f})<br>"
        f"<b>計測値:</b> {db_m:.2f} dB"
    )
    folium.Marker(
        location=[lat_m, lon_m],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="green")
    ).add_to(m)

# ヒートマップの追加
if st.session_state.heatmap_data:
    HeatMap(
        st.session_state.heatmap_data,
        radius=15,
        blur=20,
        min_opacity=0.4
    ).add_to(m)

# 等高線の描画(60dB -> 緑, 80dB -> 赤 など色を分ける)
for contour_60 in st.session_state.contours["60dB"]:
    folium.PolyLine(locations=contour_60, color="green", weight=2).add_to(m)

for contour_80 in st.session_state.contours["80dB"]:
    folium.PolyLine(locations=contour_80, color="red", weight=2).add_to(m)

# ストリームリット上に地図を表示
st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])

# ユーザーが地図を動かしたら、その中心・ズームをセッションに保存
if st_data:
    if "center" in st_data:
        st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    if "zoom" in st_data:
        st.session_state.map_zoom = st_data["zoom"]

# ─────────────────────────────────────────────────────────────────────────
# 操作パネル
# ─────────────────────────────────────────────────────────────────────────
st.subheader("操作パネル")

# CSVインポート
uploaded_file = st.file_uploader("スピーカーと計測値のCSVファイルをアップロード", type=["csv"])
if uploaded_file:
    speakers_loaded, measurements_loaded = load_csv(uploaded_file)
    if speakers_loaded:
        st.session_state.speakers.extend(speakers_loaded)
    if measurements_loaded:
        st.session_state.measurements.extend(measurements_loaded)
    st.success("CSVファイルを読み込みました。再度“更新”ボタンを押すとヒートマップに反映されます。")

# スピーカーの手動追加
new_speaker = st.text_input("新しいスピーカー (緯度,経度,方向1,方向2...)",
                            placeholder="例: 34.2579,133.2072,N,E")
if st.button("スピーカーを追加"):
    try:
        parts = new_speaker.split(",")
        lat_spk, lon_spk = float(parts[0]), float(parts[1])
        directions_spk = [parse_direction_to_degrees(d) for d in parts[2:]]
        st.session_state.speakers.append([lat_spk, lon_spk, directions_spk])
        st.session_state.heatmap_data = None  # 再計算させる
        st.success(f"スピーカーを追加: ({lat_spk}, {lon_spk}), 方向={directions_spk}")
    except (ValueError, IndexError):
        st.error("入力形式が正しくありません。(緯度,経度,方向...) の形式で入力してください。")

# スピーカーリセット
if st.button("スピーカーをリセット"):
    st.session_state.speakers = []
    st.session_state.heatmap_data = None
    st.session_state.contours = {"60dB": [], "80dB": []}
    st.success("スピーカーをリセットしました")

# 音圧設定スライダー
st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)

# 計測値の手動追加 (緯度,経度,デシベル)
new_measurement = st.text_input("計測値 (緯度,経度,デシベル)",
                                placeholder="例: 34.2579,133.2072,75")
if st.button("計測値を追加"):
    try:
        lat_m, lon_m, db_m = map(float, new_measurement.split(","))
        st.session_state.measurements.append([lat_m, lon_m, db_m])
        st.success(f"計測値を追加: ({lat_m}, {lon_m}), {db_m} dB")
    except (ValueError, IndexError):
        st.error("入力形式が正しくありません。(緯度,経度,デシベル) の形式で入力してください。")

# 計測値リセット
if st.button("計測値をリセット"):
    st.session_state.measurements = []
    st.success("計測値をリセットしました")

# ヒートマップ・等高線の再計算ボタン
if st.button("更新"):
    if st.session_state.speakers:
        st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
            st.session_state.speakers, st.session_state.L0, st.session_state.r_max, 
            grid_lat, grid_lon
        )
        st.success("ヒートマップと等高線を再計算しました")
    else:
        st.error("スピーカーがありません。追加してください。")

# CSVエクスポート
st.subheader("データのエクスポート")
col1, col2 = st.columns(2)

with col1:
    if st.button("スピーカーCSVをエクスポート"):
        csv_data = export_speakers_to_csv(st.session_state.speakers)
        st.download_button(
            label="スピーカーCSVのダウンロード",
            data=csv_data,
            file_name="speakers.csv",
            mime="text/csv"
        )

with col2:
    if st.button("計測値CSVをエクスポート"):
        csv_data = export_measurements_to_csv(st.session_state.measurements)
        st.download_button(
            label="計測値CSVのダウンロード",
            data=csv_data,
            file_name="measurements.csv",
            mime="text/csv"
        )

# ─────────────────────────────────────────────────────────────────────────
# 凡例バーを表示
# ─────────────────────────────────────────────────────────────────────────
st.subheader("音圧レベルの凡例")
colormap = cm.LinearColormap(
    colors=["blue", "green", "yellow", "red"],
    vmin=st.session_state.L0 - 40,
    vmax=st.session_state.L0,
    caption="音圧レベル (dB)"
)
# folium._repr_html_() 相当のHTMLをStreamlit上で表示
st.markdown(
    f'<div style="width:100%; text-align:center;">{colormap._repr_html_()}</div>',
    unsafe_allow_html=True
)
