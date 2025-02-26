import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math
from skimage import measure
import pandas as pd
import branca.colormap as cm
import io

# ─────────────────────────────────────────────
# セッション初期化
# ─────────────────────────────────────────────
def init_session_state():
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
    if "contours" not in st.session_state:
        st.session_state.contours = {"60dB": [], "80dB": []}
    if "L0" not in st.session_state:
        st.session_state.L0 = 80  # 初期音圧レベル(dB)
    if "r_max" not in st.session_state:
        st.session_state.r_max = 500  # 最大伝播距離

init_session_state()

# ─────────────────────────────────────────────
# 定数・方向変換の設定
# ─────────────────────────────────────────────
DIRECTION_MAPPING = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction_to_degrees(direction_str: str) -> float:
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    try:
        return float(direction_str)
    except ValueError:
        st.error(f"方向の指定 '{direction_str}' を数値に変換できません。0度として扱います。")
        return 0.0

# ─────────────────────────────────────────────
# 1地点での理論音圧(dB)計算
# ─────────────────────────────────────────────
def calc_theoretical_db_for_point(lat: float, lon: float, speakers: list, L0: float, r_max: float):
    lat, lon = float(lat), float(lon)
    power_sum = 0.0
    for spk_lat, spk_lon, spk_dirs in speakers:
        dx = (lat - spk_lat) * 111320
        dy = (lon - spk_lon) * 111320
        dist = math.sqrt(dx * dx + dy * dy)
        dist = max(dist, 1)  # 最小1m
        if dist > r_max:
            continue
        bearing = math.degrees(math.atan2((lon - spk_lon), (lat - spk_lat))) % 360
        spk_power = 0.0
        for direction in spk_dirs:
            angle_diff = abs(bearing - direction) % 360
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            directivity_factor = max(0.0, 1 - angle_diff / 180.0)
            p = 10 ** ((L0 - 20 * math.log10(dist)) / 10)
            spk_power += directivity_factor * p
        power_sum += spk_power
    if power_sum <= 0:
        return None
    db_value = 10 * math.log10(power_sum)
    return max(L0 - 40, min(db_value, L0))

# ─────────────────────────────────────────────
# ヒートマップ＆等高線計算
# ─────────────────────────────────────────────
def calculate_heatmap_and_contours(speakers: list, L0: float, r_max: float, grid_lat: np.ndarray, grid_lon: np.ndarray):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)
    for spk_lat, spk_lon, spk_dirs in speakers:
        spk_coords = np.array([spk_lat, spk_lon])
        distances = np.sqrt(np.sum((grid_coords - spk_coords) ** 2, axis=1)) * 111320
        distances = np.maximum(distances, 1)
        bearings = np.degrees(np.arctan2(
            grid_coords[:, 1] - spk_lon,
            grid_coords[:, 0] - spk_lat
        )) % 360
        power = np.zeros_like(distances)
        for direction in spk_dirs:
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            directivity_factor = np.clip(1 - angle_diff / 180, 0, 1)
            p = 10 ** ((L0 - 20 * np.log10(distances)) / 10)
            power += directivity_factor * p
        power[distances > r_max] = 0
        power_sum += power.reshape(Nx, Ny)
    sound_grid = np.full_like(power_sum, np.nan, dtype=float)
    mask = power_sum > 0
    sound_grid[mask] = 10 * np.log10(power_sum[mask])
    sound_grid = np.clip(sound_grid, L0 - 40, L0)
    heat_data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                heat_data.append([grid_lat[i, j], grid_lon[i, j], val])
    fill_grid = np.where(np.isnan(sound_grid), -9999, sound_grid)
    contours = {"60dB": [], "80dB": []}
    for key, level in {"60dB": 60, "80dB": 80}.items():
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

# ─────────────────────────────────────────────
# CSV入出力処理
# ─────────────────────────────────────────────
def load_csv(file) -> (list, list):
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
                try:
                    db_val = float(d1) if not pd.isna(d1) else 0.0
                except ValueError:
                    db_val = 0.0
                measurements.append([lat, lon, db_val])
        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込み中にエラー: {e}")
        return [], []

def export_to_csv(speakers: list, measurements: list) -> bytes:
    columns = ["種別", "緯度", "経度", "データ1", "データ2", "データ3"]
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

# ─────────────────────────────────────────────
# グリッド生成（地図表示用）
# ─────────────────────────────────────────────
def create_grid(map_center: list, map_zoom: int, lat_offset: float = 0.01, lon_offset: float = 0.01):
    lat_min = map_center[0] - lat_offset
    lat_max = map_center[0] + lat_offset
    lon_min = map_center[1] - lon_offset
    lon_max = map_center[1] + lon_offset
    zoom_factor = 100 + (map_zoom - 14) * 20
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, zoom_factor),
        np.linspace(lon_min, lon_max, zoom_factor)
    )
    return grid_lat.T, grid_lon.T

# ─────────────────────────────────────────────
# Foliumマップにマーカー・ヒートマップ・等高線を追加
# ─────────────────────────────────────────────
def add_speaker_markers(m: folium.Map, speakers: list, L0: float, r_max: float):
    for lat, lon, dirs in speakers:
        popup_html = f"""
        <div style="font-size:14px;">
          <b>スピーカ:</b> ({lat:.6f}, {lon:.6f})<br>
          <b>初期音圧:</b> {L0} dB<br>
          <b>最大伝播距離:</b> {r_max} m<br>
          <b>方向:</b> {dirs}
        </div>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(icon="volume-up", prefix="fa", color="blue")
        ).add_to(m)

def add_measurement_markers(m: folium.Map, measurements: list, speakers: list, L0: float, r_max: float):
    for lat, lon, db_m in measurements:
        theoretical_db = calc_theoretical_db_for_point(lat, lon, speakers, L0, r_max)
        theo_str = f"{theoretical_db:.2f} dB" if theoretical_db is not None else "N/A"
        popup_html = f"""
        <div style="font-size:14px;">
          <b>計測位置:</b> ({lat:.6f}, {lon:.6f})<br>
          <b>計測値:</b> {db_m:.2f} dB<br>
          <b>理論値:</b> {theo_str}
        </div>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(icon="info-circle", prefix="fa", color="green")
        ).add_to(m)

def add_heatmap_and_contours(m: folium.Map, heat_data: list, contours: dict):
    if heat_data:
        HeatMap(heat_data, radius=15, blur=20, min_opacity=0.4).add_to(m)
    for contour in contours["60dB"]:
        folium.PolyLine(locations=contour, color="green", weight=2).add_to(m)
    for contour in contours["80dB"]:
        folium.PolyLine(locations=contour, color="red", weight=2).add_to(m)

# ─────────────────────────────────────────────
# メイン処理（Streamlit UI）
# ─────────────────────────────────────────────
def main():
    st.title("防災スピーカー音圧ヒートマップ")
    
    # マップ中心とズームに合わせたグリッド生成
    grid_lat, grid_lon = create_grid(st.session_state.map_center, st.session_state.map_zoom)
    
    # ヒートマップデータが未計算の場合は再計算する
    if st.session_state.heatmap_data is None and st.session_state.speakers:
        st.session_state.heatmap_data, st.session_state.contours = calculate_heatmap_and_contours(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat,
            grid_lon
        )
    
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom
    )
    
    add_speaker_markers(m, st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
    add_measurement_markers(m, st.session_state.measurements, st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
    
    if st.session_state.heatmap_data:
        add_heatmap_and_contours(m, st.session_state.heatmap_data, st.session_state.contours)
    
    # マップの移動・ズーム情報の取得と更新
    st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])
    if st_data:
        new_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
        new_zoom = st_data["zoom"]
        if new_center != st.session_state.map_center or new_zoom != st.session_state.map_zoom:
            st.session_state.map_center = new_center
            st.session_state.map_zoom = new_zoom
            # ヒートマップ用のグリッドを更新するためにデータをリセット
            st.session_state.heatmap_data = None
    
    st.subheader("操作パネル")
    
    # CSVアップロード
    uploaded_file = st.file_uploader("CSVをアップロード (種別/緯度/経度/データ1..3列)", type=["csv"])
    if uploaded_file:
        speakers_loaded, measurements_loaded = load_csv(uploaded_file)
        if speakers_loaded:
            st.session_state.speakers.extend(speakers_loaded)
        if measurements_loaded:
            st.session_state.measurements.extend(measurements_loaded)
        st.success("CSVを読み込みました。『更新』ボタンでヒートマップに反映可能です。")
    
    # 新規スピーカ追加
    new_speaker = st.text_input(
        "新しいスピーカ (緯度,経度,方向1,方向2,方向3)",
        placeholder="例: 34.2579,133.2072,N,E,SE"
    )
    if st.button("スピーカを追加"):
        try:
            items = new_speaker.split(",")
            lat_spk = float(items[0])
            lon_spk = float(items[1])
            dirs_spk = [parse_direction_to_degrees(d) for d in items[2:]]
            st.session_state.speakers.append([lat_spk, lon_spk, dirs_spk])
            st.session_state.heatmap_data = None
            st.success(f"スピーカを追加しました: ({lat_spk}, {lon_spk}), 方向 = {dirs_spk}")
        except Exception as e:
            st.error(f"入力エラー: {e}")
    
    # 計測値追加
    new_measurement = st.text_input("計測値 (緯度,経度,dB)", placeholder="例: 34.2578,133.2075,75")
    if st.button("計測値を追加"):
        try:
            items = new_measurement.split(",")
            lat_m = float(items[0])
            lon_m = float(items[1])
            db_m = float(items[2])
            st.session_state.measurements.append([lat_m, lon_m, db_m])
            st.success(f"計測値を追加しました: ({lat_m}, {lon_m}), {db_m} dB")
        except Exception as e:
            st.error(f"入力エラー: {e}")
    
    if st.button("スピーカをリセット"):
        st.session_state.speakers = []
        st.session_state.heatmap_data = None
        st.session_state.contours = {"60dB": [], "80dB": []}
        st.success("スピーカ情報をリセットしました")
    
    if st.button("計測値をリセット"):
        st.session_state.measurements = []
        st.success("計測値情報をリセットしました")
    
    st.session_state.L0 = st.slider("初期音圧レベル(dB)", 50, 100, st.session_state.L0)
    st.session_state.r_max = st.slider("最大伝播距離(m)", 100, 2000, st.session_state.r_max)
    
    # 「更新」ボタンによるヒートマップ再計算
    if st.button("更新"):
        if st.session_state.speakers:
            grid_lat, grid_lon = create_grid(st.session_state.map_center, st.session_state.map_zoom)
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
    
    st.subheader("CSVのエクスポート (種別/緯度/経度/データ1..3形式)")
    if st.button("CSVをエクスポート"):
        csv_data = export_to_csv(st.session_state.speakers, st.session_state.measurements)
        st.download_button(
            label="CSVファイルのダウンロード",
            data=csv_data,
            file_name="sound_map_data.csv",
            mime="text/csv"
        )
    
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

if __name__ == "__main__":
    main()
