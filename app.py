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
    st.session_state.speakers = [
        [34.25741795269067, 133.20450105700033, [0.0, 90.0]]  # 初期サンプル
    ]

if "measurements" not in st.session_state:
    st.session_state.measurements = []

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None

# contoursの初期化を修正: 必要なキーが存在するか確認し、存在しない場合は追加
required_contour_keys = ["L0-20dB", "L0dB"]
if "contours" not in st.session_state:
    st.session_state.contours = {key: [] for key in required_contour_keys}
else:
    for key in required_contour_keys:
        if key not in st.session_state.contours:
            st.session_state.contours[key] = []

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
# 特定1地点での音圧理論値(dB)を計算する関数
# ─────────────────────────────────────────────────────────────────────────
def calc_theoretical_db_for_point(lat, lon, speakers, L0, r_max):
    """
    渡された1点(lat, lon)に対して、複数スピーカの合成音圧レベル(dB)を理論計算。
    calculate_heatmap_and_contours() と同じロジックの簡易版。
    """
    lat, lon = float(lat), float(lon)
    power_sum = 0.0

    for spk_lat, spk_lon, spk_dirs in speakers:
        # 距離 (m換算)
        dx = (lat - spk_lat) * 111320
        dy = (lon - spk_lon) * 111320
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1:
            dist = 1  # 1m以下は1mとみなす

        # 距離が r_max を超える場合は無視
        if dist > r_max:
            continue

        # Bearing (北=0°, 東=90°, ...)
        bearing = math.degrees(math.atan2((lon - spk_lon), (lat - spk_lat))) % 360

        # 指向性を考慮しながらパワー合成
        spk_power = 0.0
        for direction in spk_dirs:
            angle_diff = abs(bearing - direction) % 360
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # 角度差に応じた減衰 [0～1]
            directivity_factor = max(0.0, 1 - angle_diff / 180.0)

            # 距離減衰: L = L0 - 20*log10(r)
            # → パワー比 ~ 10^(L/10)
            p = 10 ** ((L0 - 20 * math.log10(dist)) / 10)
            spk_power += directivity_factor * p

        power_sum += spk_power

    if power_sum <= 0:
        return None  # 音が届いていない(=超微小レベル)

    # dB変換
    db_value = 10 * math.log10(power_sum)
    # ヒートマップと同じく L0-40 ~ L0 でクリップ
    db_value = max(L0 - 40, min(db_value, L0))
    return db_value

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

        distances = np.sqrt(np.sum((grid_coords - spk_coords)**2, axis=1)) * 111320
        distances[distances < 1] = 1  # 最小1m

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

    # ヒートマップ用データの効率的な生成
    valid_indices = ~np.isnan(sound_grid)
    heat_data = np.column_stack((grid_lat[valid_indices], grid_lon[valid_indices], sound_grid[valid_indices])).tolist()

    # 等高線(L0-20dB, L0)
    fill_grid = np.where(np.isnan(sound_grid), -9999, sound_grid)
    contours = {"L0-20dB": [], "L0dB": []}
    levels = {"L0-20dB": L0 - 20, "L0dB": L0}
    for key, level in levels.items():
        raw_contours = measure.find_contours(fill_grid, level=level)
        if not raw_contours:
            st.warning(f"{key}の等高線が見つかりませんでした。レベルを調整してください。")
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
# CSVインポート (種別/緯度/経度/データ1..3 形式)
# ─────────────────────────────────────────────────────────────────────────
def load_csv(file):
    try:
        df = pd.read_csv(file)
        speakers = []
        measurements = []

        for _, row in df.iterrows():
            item_type = row.get("種別", None)
            lat = row.get("緯度", None)
            lon = row.get("経度", None)
            d1 = row.get("データ1", None)
            d2 = row.get("データ2", None)
            d3 = row.get("データ3", None)

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
                directions = []
                for d in [d1, d2, d3]:
                    if not pd.isna(d) and str(d).strip() != "":
                        directions.append(parse_direction_to_degrees(str(d)))
                speakers.append([lat, lon, directions])
            elif item_type == "計測値":
                db_val = 0.0
                if not pd.isna(d1):
                    try:
                        db_val = float(d1)
                    except ValueError:
                        db_val = 0.0
                measurements.append([lat, lon, db_val])

        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込み中にエラー: {e}")
        return [], []

# ─────────────────────────────────────────────────────────────────────────
# CSVエクスポート (種別/緯度/経度/データ1..3 形式)
# ─────────────────────────────────────────────────────────────────────────
def export_to_csv(speakers, measurements):
    columns = ["種別","緯度","経度","データ1","データ2","データ3"]
    rows = []

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

    for lat_m, lon_m, db_m in measurements:
        row = {
            "種別": "計測値",
            "緯度": lat_m,
            "経度": lon_m,
            "データ1": db_m,
            "データ2": "",
            "データ3": "",
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────
# Streamlitメイン表示
# ─────────────────────────────────────────────────────────────────────────
st.title("防災スピーカ音圧ヒートマップ")

# 地図用グリッド
lat_min = st.session_state.map_center[0] - 0.01
lat_max = st.session_state.map_center[0] + 0.01
lon_min = st.session_state.map_center[1] - 0.01
lon_max = st.session_state.map_center[1] + 0.01

zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
zoom_factor = max(100, zoom_factor)  # グリッドの解像度を最低100に設定
grid_lat, grid_lon = np.meshgrid(
    np.linspace(lat_min, lat_max, zoom_factor),
    np.linspace(lon_min, lon_max, zoom_factor)
)
grid_lat = grid_lat.T
grid_lon = grid_lon.T

# 初回ヒートマップ計算
if st.session_state.heatmap_data is None and st.session_state.speakers:
    st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
        st.session_state.speakers,
        st.session_state.L0,
        st.session_state.r_max,
        grid_lat,
        grid_lon
    )

# Foliumマップ生成
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom
)

# ─────────────────────────────────────────────────────────────────────────
# スピーカー・ピン配置 (アイコンとフォント調整)
# ─────────────────────────────────────────────────────────────────────────
for lat_s, lon_s, dirs in st.session_state.speakers:
    directions_str = ", ".join([f"{dir}°" for dir in dirs])
    popup_html = f"""
    <div style="font-size:14px;">
      <b>スピーカ:</b> ({lat_s:.6f}, {lon_s:.6f})<br>
      <b>初期音圧:</b> {st.session_state.L0} dB<br>
      <b>最大伝播距離:</b> {st.session_state.r_max} m<br>
      <b>方向:</b> {directions_str}
    </div>
    """
    folium.Marker(
        location=[lat_s, lon_s],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(icon="volume-up", prefix="fa", color="blue")
    ).add_to(m)

# ─────────────────────────────────────────────────────────────────────────
# 計測値・ピン配置 (アイコン+理論値をポップアップに追加表示)
# ─────────────────────────────────────────────────────────────────────────
for lat_m, lon_m, db_m in st.session_state.measurements:
    # 理論値を計算
    theoretical_db = calc_theoretical_db_for_point(
        lat_m, lon_m,
        st.session_state.speakers,
        st.session_state.L0,
        st.session_state.r_max
    )
    if theoretical_db is not None:
        theo_str = f"{theoretical_db:.2f} dB"
    else:
        theo_str = "N/A"

    popup_html = f"""
    <div style="font-size:14px;">
      <b>計測位置:</b> ({lat_m:.6f}, {lon_m:.6f})<br>
      <b>計測値:</b> {db_m:.2f} dB<br>
      <b>理論値:</b> {theo_str}
    </div>
    """

    folium.Marker(
        location=[lat_m, lon_m],
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(icon="info-circle", prefix="fa", color="green")
    ).add_to(m)

# ─────────────────────────────────────────────────────────────────────────
# ヒートマップ・等高線
# ─────────────────────────────────────────────────────────────────────────
if st.session_state.heatmap_data:
    HeatMap(
        st.session_state.heatmap_data,
        radius=15,
        blur=20,
        min_opacity=0.4
    ).add_to(m)

for key, level in {"L0-20dB": st.session_state.L0 - 20, "L0dB": st.session_state.L0}.items():
    for contour in st.session_state.contours[key]:
        color = "green" if key == "L0-20dB" else "red"
        folium.PolyLine(locations=contour, color=color, weight=2).add_to(m)

# マップをStreamlitに表示し、移動状況を取得
st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])
if st_data:
    if "center" in st_data:
        st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    if "zoom" in st_data:
        st.session_state.map_zoom = st_data["zoom"]
        # グリッドを再生成
        lat_min = st.session_state.map_center[0] - 0.01
        lat_max = st.session_state.map_center[0] + 0.01
        lon_min = st.session_state.map_center[1] - 0.01
        lon_max = st.session_state.map_center[1] + 0.01

        zoom_factor = 100 + (st.session_state.map_zoom - 14) * 20
        zoom_factor = max(100, zoom_factor)
        grid_lat, grid_lon = np.meshgrid(
            np.linspace(lat_min, lat_max, zoom_factor),
            np.linspace(lon_min, lon_max, zoom_factor)
        )
        grid_lat = grid_lat.T
        grid_lon = grid_lon.T

        # ヒートマップ再計算
        if st.session_state.speakers:
            st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
                st.session_state.speakers,
                st.session_state.L0,
                st.session_state.r_max,
                grid_lat,
                grid_lon
            )

# ─────────────────────────────────────────────────────────────────────────
# 操作パネル
# ─────────────────────────────────────────────────────────────────────────
st.subheader("操作パネル")

# CSVアップロード
uploaded_file = st.file_uploader("CSVをアップロード (種別/緯度/経度/データ1..3列)", type=["csv"])
if uploaded_file:
    speakers_loaded, measurements_loaded = load_csv(uploaded_file)
    if speakers_loaded:
        st.session_state.speakers.extend(speakers_loaded)
    if measurements_loaded:
        st.session_state.measurements.extend(measurements_loaded)
    st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
        st.session_state.speakers,
        st.session_state.L0,
        st.session_state.r_max,
        grid_lat,
        grid_lon
    )
    st.success("CSVを読み込み、ヒートマップを更新しました。")

# 新規スピーカ追加
st.markdown("### 新しいスピーカを追加")
new_speaker = st.text_input(
    "スピーカの追加 (緯度, 経度, 方向1, 方向2, 方向3)",
    placeholder="例: 34.2579,133.2072,N,E,SE"
)
if st.button("スピーカを追加"):
    try:
        items = new_speaker.split(",")
        if len(items) < 2:
            raise ValueError("緯度と経度は必須です。")
        lat_spk = float(items[0].strip())
        lon_spk = float(items[1].strip())
        dirs_spk = [parse_direction_to_degrees(d) for d in items[2:] if d.strip() != ""]
        if not dirs_spk:
            st.warning("少なくとも1つの方向を指定してください。")
        else:
            st.session_state.speakers.append([lat_spk, lon_spk, dirs_spk])
            st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
                st.session_state.speakers,
                st.session_state.L0,
                st.session_state.r_max,
                grid_lat,
                grid_lon
            )
            st.success(f"スピーカを追加しました: ({lat_spk}, {lon_spk}), 方向={', '.join([f'{d}°' for d in dirs_spk])}")
    except Exception as e:
        st.error(f"入力エラー: {e}")

# 新規計測値追加
st.markdown("### 新しい計測値を追加")
new_measurement = st.text_input("計測値の追加 (緯度, 経度, dB)", placeholder="例: 34.2578,133.2075,75")
if st.button("計測値を追加"):
    try:
        items = new_measurement.split(",")
        if len(items) < 3:
            raise ValueError("緯度、経度、dBはすべて必須です。")
        lat_m = float(items[0].strip())
        lon_m = float(items[1].strip())
        db_m = float(items[2].strip())
        st.session_state.measurements.append([lat_m, lon_m, db_m])
        st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat,
            grid_lon
        )
        st.success(f"計測値を追加しました: ({lat_m}, {lon_m}), {db_m} dB")
    except Exception as e:
        st.error(f"入力エラー: {e}")

# リセットボタン
st.markdown("### データのリセット")
col1, col2 = st.columns(2)
with col1:
    if st.button("スピーカをリセット"):
        st.session_state.speakers = []
        st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat,
            grid_lon
        )
        st.success("スピーカをリセットしました。ヒートマップを更新しました。")
with col2:
    if st.button("計測値をリセット"):
        st.session_state.measurements = []
        st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat,
            grid_lon
        )
        st.success("計測値をリセットしました。ヒートマップを更新しました。")

# 音圧/距離スライダー
st.markdown("### 音圧と伝播距離の設定")
col3, col4 = st.columns(2)
with col3:
    L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
    st.session_state.L0 = L0
with col4:
    r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max, step=100)
    st.session_state.r_max = r_max

# 更新ボタン: ヒートマップ再計算
if st.button("ヒートマップを更新"):
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

# エクスポート
st.subheader("CSVのエクスポート (種別/緯度/経度/データ1..3形式)")
if st.button("CSVをエクスポート"):
    csv_data = export_to_csv(st.session_state.speakers, st.session_state.measurements)
    st.download_button(
        label="CSVファイルのダウンロード",
        data=csv_data,
        file_name="sound_map_data.csv",
        mime="text/csv"
    )

# 凡例バー
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
