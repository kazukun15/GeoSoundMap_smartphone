import streamlit as st
import folium
from folium.plugins import HeatMap, Draw
from streamlit_folium import st_folium
import numpy as np
import math
from skimage import measure
import pandas as pd
import branca.colormap as cm
import io  # エクスポート用バッファ
from scipy.ndimage import gaussian_filter  # データの滑らかさ向上
import geopandas as gpd
import xml.etree.ElementTree as ET  # XMLパース用
import json
import rasterio.transform

from shapely.geometry import LineString

from rasterio.transform import Affine

# ─────────────────────────────────────────────────────────────────────────
# セッション初期設定
# ─────────────────────────────────────────────────────────────────────────
if "map_center" not in st.session_state:
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]  # 初期中心位置（例）

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 14  # 初期ズームレベル

if "speakers" not in st.session_state:
    st.session_state.speakers = [
        {"lat": 34.25741795269067, "lon": 133.20450105700033, "directions": [0.0, 90.0]}  # 初期スピーカーサンプル
    ]

if "measurements" not in st.session_state:
    st.session_state.measurements = []

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None

# contoursの初期化を修正: 必要なキーが存在するか確認し、存在しない場合は追加
required_contour_keys = ["L0-20dB", "L0-1dB"]
if "contours" not in st.session_state:
    st.session_state.contours = {key: [] for key in required_contour_keys}
else:
    for key in required_contour_keys:
        if key not in st.session_state.contours:
            st.session_state.contours[key] = []

if "L0" not in st.session_state:
    st.session_state.L0 = 80  # 初期音圧レベル(dB)

if "r_max" not in st.session_state:
    st.session_state.r_max = 500  # 最大伝播距離(m)

# 建築材質とその音減衰率（dB）の定義を session_state に移行
if "MATERIAL_ATTENUATION" not in st.session_state:
    st.session_state.MATERIAL_ATTENUATION = {
        "コンクリート": 30,  # 例: コンクリート壁は30dB減衰
        "木材": 20,         # 例: 木製壁は20dB減衰
        "ガラス": 25,       # 例: ガラス窓は25dB減衰
        "断熱材": 35,       # 例: 断熱材は35dB減衰
        "石膏ボード": 28,   # 例: 石膏ボード壁は28dB減衰
        "Unknown": 10,      # 不明な材質の場合
        # 必要に応じて追加
    }

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

# 家の中での音圧レベルの計算を関数内で行う
def calc_indoor_db(theoretical_db, material):
    """
    理論音圧から材質ごとの減衰量を引いて家内音圧を計算します。
    """
    attenuation = st.session_state.MATERIAL_ATTENUATION.get(material, 10)  # デフォルト10dB
    indoor_db = theoretical_db - attenuation
    return max(indoor_db, st.session_state.L0 - 40)  # 最小値はL0-40dBにクリップ

# レイ・トレーシング関数
def is_obstructed(speaker, receiver, buildings, dem_data, dem_transform):
    """
    スピーカーから受信点への音波が地形や建物によって遮断されているかを判定します。
    speaker: [x, y] 経度, 緯度
    receiver: [x, y] 経度, 緯度
    buildings: GeoDataFrame
    dem_data: NumPy配列
    dem_transform: rasterio.transform.Affine
    """
    line = LineString([speaker, receiver])

    # 建物による遮断判定
    if buildings.intersects(line).any():
        return True  # 建物により遮断されている

    # 地形による遮断判定
    # サンプルポイント間隔を設定（例: 10mごと）
    num_samples = int(line.length * 0.1)  # 10mごとのサンプル
    if num_samples < 2:
        num_samples = 2

    for i in range(1, num_samples):
        point = line.interpolate(float(i) / num_samples, normalized=True)
        # 緯度経度をDEMのインデックスに変換
        try:
            row, col = rasterio.transform.rowcol(dem_transform, point.x, point.y)
            elevation = dem_data[row, col]
        except IndexError:
            elevation = 0  # データ外の場合は0とする

        # 音波の直線上の高度
        # スピーカーと受信点の高度が不明な場合は、DEMから取得
        # ここでは簡単化のため、スピーカーと受信点の高度を0と仮定
        speaker_elevation = 0  # 必要に応じて修正
        receiver_elevation = 0  # 必要に応じて修正
        expected_elevation = speaker_elevation + (receiver_elevation - speaker_elevation) * (i / num_samples)

        if elevation > expected_elevation:
            return True  # 地形により遮断されている

    return False  # 遮断なし

# 特定1地点での音圧理論値(dB)を計算する関数（障害物考慮版）
def calc_theoretical_db_with_obstruction(lat, lon, speakers, L0, r_max, buildings, dem_data, dem_transform):
    """
    渡された1点(lat, lon)に対して、障害物を考慮した複数スピーカの合成音圧レベル(dB)を理論計算。
    """
    receiver = [lon, lat]  # shapelyでは経度がx, 緯度がy
    power_sum = 0.0

    for spk in speakers:
        spk_lat, spk_lon, spk_dirs = spk["lat"], spk["lon"], spk["directions"]
        speaker = [spk_lon, spk_lat]  # shapelyでは経度がx, 緯度がy

        # 距離の計算
        dx = (lon - spk_lon) * 111320  # 緯度経度をメートルに変換（近似）
        dy = (lat - spk_lat) * 111320
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1:
            dist = 1  # 1m以下は1mとみなす

        if dist > r_max:
            continue

        # 遮断判定
        if is_obstructed(speaker, receiver, buildings, dem_data, dem_transform):
            continue  # 遮断されている場合は音源を無視

        # Bearing
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
        return None

    # dB変換
    db_value = 10 * math.log10(power_sum)
    db_value = max(L0 - 40, min(db_value, L0))
    return db_value

# ヒートマップ & 等高線の計算（障害物考慮版）
def calculate_heatmap_and_contours_with_obstruction(speakers, L0, r_max, grid_lat, grid_lon, buildings, dem_data, dem_transform):
    Nx, Ny = grid_lat.shape
    sound_grid = np.full((Nx, Ny), np.nan, dtype=float)

    # 各グリッドポイントで音圧を計算
    for i in range(Nx):
        for j in range(Ny):
            lat = grid_lat[i, j]
            lon = grid_lon[i, j]
            db = calc_theoretical_db_with_obstruction(lat, lon, speakers, L0, r_max, buildings, dem_data, dem_transform)
            if db is not None:
                sound_grid[i, j] = db

    # データの滑らかさを向上
    sound_grid_smoothed = gaussian_filter(sound_grid, sigma=1)
    fill_grid = np.where(np.isnan(sound_grid_smoothed), -9999, sound_grid_smoothed)

    # ヒートマップ用データの生成
    valid_indices = ~np.isnan(sound_grid_smoothed)
    heat_data = np.column_stack((grid_lat[valid_indices], grid_lon[valid_indices], sound_grid_smoothed[valid_indices])).tolist()

    # 等高線(L0-20dB, L0-1dB)
    contours = {"L0-20dB": [], "L0-1dB": []}
    levels = {"L0-20dB": L0 - 20, "L0-1dB": L0 - 1}

    # データ範囲の確認
    min_db = np.nanmin(sound_grid_smoothed)
    max_db = np.nanmax(sound_grid_smoothed)
    st.write(f"Sound Grid Min: {min_db:.2f} dB")
    st.write(f"Sound Grid Max: {max_db:.2f} dB")

    for key, level in levels.items():
        # レベルがデータ範囲内か確認
        if not (min_db <= level <= max_db):
            st.warning(f"{key} のレベル {level}dB はデータ範囲外です。")
            continue

        raw_contours = measure.find_contours(fill_grid, level=level)
        if not raw_contours:
            # 近似レベルで再試行
            adjusted_level = level - 0.5  # 例として0.5dB下げる
            if min_db <= adjusted_level <= max_db:
                raw_contours = measure.find_contours(fill_grid, level=adjusted_level)
                if raw_contours:
                    st.warning(f"{key} の等高線が見つかりませんでした。近似レベル {adjusted_level:.1f}dB で再試行しました。")
                    key_adjusted = f"{key}-approx"
                    contours[key_adjusted] = []
                    for contour in raw_contours:
                        lat_lon_contour = []
                        for y, x in contour:
                            iy, ix = int(round(y)), int(round(x))
                            if 0 <= iy < Nx and 0 <= ix < Ny:
                                lat_lon_contour.append((grid_lat[iy, ix], grid_lon[iy, ix]))
                        if len(lat_lon_contour) > 1:
                            contours[key_adjusted].append(lat_lon_contour)
            else:
                st.warning(f"{key} の近似レベル {adjusted_level:.1f}dB はデータ範囲外です。")
            continue

        for contour in raw_contours:
            lat_lon_contour = []
            for y, x in contour:
                iy, ix = int(round(y)), int(round(x))
                if 0 <= iy < Nx and 0 <= ix < Ny:
                    lat_lon_contour.append((grid_lat[iy, ix], grid_lon[iy, ix]))
            if len(lat_lon_contour) > 1:
                contours[key].append(lat_lon_contour)

    return heat_data, contours, sound_grid_smoothed

# XML DEMデータの読み込み関数
@st.cache_data
def load_dem_xml(xml_file):
    """
    XML形式のDEMデータを読み込み、NumPy配列とトランスフォーム情報を返します。
    XMLの具体的な構造に応じて、解析方法を調整してください。
    ここでは、簡単なグリッド形式のXMLを仮定します。
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Grid情報を取得
        grid = root.find('.//Grid')
        rows = int(grid.get('rows'))
        cols = int(grid.get('cols'))

        # Origin情報を取得
        origin = grid.find('Origin')
        origin_lat = float(origin.get('lat'))
        origin_lon = float(origin.get('lon'))

        # CellSizeを取得
        cell_size = float(grid.find('CellSize').text)

        # Elevation値を取得
        elevations = []
        for elev in grid.find('.//Elevations').iter('elevation'):
            elevations.append(float(elev.text))

        # NumPy配列に変換
        dem_data = np.array(elevations).reshape((rows, cols))

        # トランスフォーム情報の作成
        dem_transform = Affine.translation(origin_lon, origin_lat) * Affine.scale(cell_size, cell_size)

        return dem_data, dem_transform
    except Exception as e:
        st.error(f"XML DEMデータの読み込み中にエラーが発生しました: {e}")
        return np.array([]), None

# 建物データの読み込み関数
@st.cache_data
def load_buildings(buildings_geojson_path):
    """
    GeoJSON形式の建物データを読み込み、GeoDataFrameとして返します。
    """
    try:
        buildings = gpd.read_file(buildings_geojson_path)
        return buildings
    except Exception as e:
        st.error(f"建物データの読み込みに失敗しました: {e}")
        return gpd.GeoDataFrame()

# CSVインポート (種別/緯度/経度/データ1..4 形式)
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
            material = row.get("データ4", "Unknown")  # 材質情報をデータ4として仮定。欠落時は "Unknown"

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
                speakers.append({"lat": lat, "lon": lon, "directions": directions})
            elif item_type == "計測値":
                db_val = 0.0
                if not pd.isna(d1):
                    try:
                        db_val = float(d1)
                    except ValueError:
                        db_val = 0.0
                measurements.append({"lat": lat, "lon": lon, "db": db_val, "material": material})

        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込み中にエラー: {e}")
        return [], []

# CSVエクスポート (種別/緯度/経度/データ1..4 形式)
def export_to_csv(speakers, measurements):
    columns = ["種別","緯度","経度","データ1","データ2","データ3","データ4"]
    rows = []

    for spk in speakers:
        dirs = spk["directions"]
        row = {
            "種別": "スピーカ",
            "緯度": spk["lat"],
            "経度": spk["lon"],
            "データ1": dirs[0] if len(dirs) > 0 else "",
            "データ2": dirs[1] if len(dirs) > 1 else "",
            "データ3": dirs[2] if len(dirs) > 2 else "",
            "データ4": ""  # スピーカには材質情報がない
        }
        rows.append(row)

    for meas in measurements:
        row = {
            "種別": "計測値",
            "緯度": meas["lat"],
            "経度": meas["lon"],
            "データ1": meas["db"],
            "データ2": "",
            "データ3": "",
            "データ4": meas["material"],
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# マップにDEMと建物を追加する関数
def add_dem_to_map(m, dem_data, dem_transform, contour_interval=100):
    """
    DEMデータの等高線をマップに追加します。
    """
    try:
        contours = measure.find_contours(dem_data, level=range(int(np.min(dem_data)), int(np.max(dem_data)), contour_interval))
    except Exception as e:
        st.error(f"DEMデータの等高線生成に失敗しました: {e}")
        return

    for contour in contours:
        lat_lon_contour = []
        for y, x in contour:
            try:
                lon, lat = rasterio.transform.xy(dem_transform, y, x)
                lat_lon_contour.append((lat, lon))
            except Exception as e:
                st.warning(f"DEMデータの座標変換に失敗しました: {e}")
                continue
        if len(lat_lon_contour) > 1:
            folium.PolyLine(locations=lat_lon_contour, color='gray', weight=1, opacity=0.5).add_to(m)

def add_buildings_to_map(m, buildings):
    """
    建物データをマップに追加します。
    """
    try:
        for _, building in buildings.iterrows():
            folium.GeoJson(
                building.geometry,
                style_function=lambda x: {'fillColor': 'orange', 'color': 'black', 'weight': 1, 'fillOpacity': 0.5}
            ).add_to(m)
    except Exception as e:
        st.error(f"建物データのマップへの追加に失敗しました: {e}")

def add_dem_and_buildings_to_map(m, dem_data, dem_transform, buildings):
    # DEMの等高線を追加
    add_dem_to_map(m, dem_data, dem_transform, contour_interval=100)

    # 建物を追加
    add_buildings_to_map(m, buildings)

# スピーカーおよび計測値の編集・削除機能
def manage_entities():
    st.markdown("### スピーカーの管理")
    if st.session_state.speakers:
        for idx, spk in enumerate(st.session_state.speakers):
            with st.expander(f"スピーカー {idx+1}: ({spk['lat']}, {spk['lon']})"):
                cols = st.columns(4)
                with cols[0]:
                    new_lat = st.text_input(f"緯度", value=str(spk['lat']), key=f"spk_lat_{idx}")
                with cols[1]:
                    new_lon = st.text_input(f"経度", value=str(spk['lon']), key=f"spk_lon_{idx}")
                with cols[2]:
                    new_dirs = st.text_input(f"方向 (カンマ区切り)", value=','.join([str(d) for d in spk['directions']]), key=f"spk_dirs_{idx}")
                with cols[3]:
                    if st.button("更新", key=f"update_spk_{idx}"):
                        try:
                            spk['lat'] = float(new_lat)
                            spk['lon'] = float(new_lon)
                            dirs = [parse_direction_to_degrees(d) for d in new_dirs.split(",") if d.strip() != ""]
                            if dirs:
                                spk['directions'] = dirs
                                st.success(f"スピーカー {idx+1} を更新しました。")
                                # ヒートマップの再計算
                                recalculate_heatmap()
                            else:
                                st.warning("少なくとも1つの方向を指定してください。")
                        except ValueError:
                            st.error("緯度または経度の値が不正です。")

                    if st.button("削除", key=f"delete_spk_{idx}"):
                        del st.session_state.speakers[idx]
                        st.success(f"スピーカー {idx+1} を削除しました。")
                        recalculate_heatmap()
                        st.experimental_rerun()
    else:
        st.info("現在、スピーカーは登録されていません。")

    st.markdown("### 計測値の管理")
    if st.session_state.measurements:
        for idx, meas in enumerate(st.session_state.measurements):
            with st.expander(f"計測値 {idx+1}: ({meas['lat']}, {meas['lon']})"):
                cols = st.columns(4)
                with cols[0]:
                    new_lat = st.text_input(f"緯度", value=str(meas['lat']), key=f"meas_lat_{idx}")
                with cols[1]:
                    new_lon = st.text_input(f"経度", value=str(meas['lon']), key=f"meas_lon_{idx}")
                with cols[2]:
                    new_db = st.text_input(f"dB", value=str(meas['db']), key=f"meas_db_{idx}")
                with cols[3]:
                    new_material = st.selectbox(f"材質", list(st.session_state.MATERIAL_ATTENUATION.keys()), index=list(st.session_state.MATERIAL_ATTENUATION.keys()).index(meas['material']) if meas['material'] in st.session_state.MATERIAL_ATTENUATION else 0, key=f"meas_material_{idx}")
                cols_del = st.columns(2)
                with cols_del[0]:
                    if st.button("更新", key=f"update_meas_{idx}"):
                        try:
                            meas['lat'] = float(new_lat)
                            meas['lon'] = float(new_lon)
                            meas['db'] = float(new_db)
                            meas['material'] = new_material
                            st.success(f"計測値 {idx+1} を更新しました。")
                            # ヒートマップの再計算
                            recalculate_heatmap()
                        except ValueError:
                            st.error("緯度、経度、またはdBの値が不正です。")
                with cols_del[1]:
                    if st.button("削除", key=f"delete_meas_{idx}"):
                        del st.session_state.measurements[idx]
                        st.success(f"計測値 {idx+1} を削除しました。")
                        recalculate_heatmap()
                        st.experimental_rerun()
    else:
        st.info("現在、計測値は登録されていません。")

# ヒートマップの詳細設定
def heatmap_settings():
    st.markdown("### ヒートマップの設定")
    radius = st.slider("ヒートマップの半径", min_value=1, max_value=50, value=getattr(st.session_state, 'heatmap_radius', 15), step=1)
    blur = st.slider("ヒートマップのブラー", min_value=1, max_value=50, value=getattr(st.session_state, 'heatmap_blur', 20), step=1)
    min_opacity = st.slider("ヒートマップの最小不透明度", min_value=0.0, max_value=1.0, value=getattr(st.session_state, 'heatmap_min_opacity', 0.4), step=0.1)
    st.session_state.heatmap_radius = radius
    st.session_state.heatmap_blur = blur
    st.session_state.heatmap_min_opacity = min_opacity

    st.markdown("#### カラーレジェンドのカスタマイズ")
    color_choice = st.multiselect("色の順序を選択", options=["blue", "green", "yellow", "red"], default=["blue", "green", "yellow", "red"])
    if len(color_choice) >= 2:
        st.session_state.heatmap_gradient = {i/(len(color_choice)-1): color for i, color in enumerate(color_choice)}
    else:
        st.warning("少なくとも2色を選択してください。デフォルトのグラデーションを使用します。")
        st.session_state.heatmap_gradient = {0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'red'}

# アプリケーションの状態保存と読み込み
def save_load_state():
    st.markdown("### アプリケーションの状態管理")
    save_load_choice = st.selectbox("操作を選択", ["保存", "読み込み"])

    if save_load_choice == "保存":
        if st.button("現在の状態を保存"):
            state = {
                "speakers": st.session_state.speakers,
                "measurements": st.session_state.measurements,
                "materials": st.session_state.MATERIAL_ATTENUATION,
                "L0": st.session_state.L0,
                "r_max": st.session_state.r_max
            }
            st.session_state.saved_state = json.dumps(state)
            st.success("現在の状態を保存しました。")

    elif save_load_choice == "読み込み":
        if "saved_state" in st.session_state:
            if st.button("保存された状態を読み込む"):
                try:
                    state = json.loads(st.session_state.saved_state)
                    st.session_state.speakers = state.get("speakers", [])
                    st.session_state.measurements = state.get("measurements", [])
                    st.session_state.MATERIAL_ATTENUATION = state.get("materials", st.session_state.MATERIAL_ATTENUATION)
                    st.session_state.L0 = state.get("L0", st.session_state.L0)
                    st.session_state.r_max = state.get("r_max", st.session_state.r_max)
                    st.success("保存された状態を読み込みました。ヒートマップを更新してください。")
                    recalculate_heatmap()
                except json.JSONDecodeError:
                    st.error("保存された状態の読み込みに失敗しました。")
        else:
            st.info("保存された状態がありません。")

# フィルタリング機能
def filter_measurements():
    st.markdown("### 計測値のフィルタリング")
    if st.session_state.measurements:
        min_db = st.number_input("最小dB値", value=st.session_state.L0 - 40, step=1)
        max_db = st.number_input("最大dB値", value=st.session_state.L0, step=1)
        selected_materials = st.multiselect("表示する材質", options=list(st.session_state.MATERIAL_ATTENUATION.keys()), default=list(st.session_state.MATERIAL_ATTENUATION.keys()))

        filtered_measurements = [
            meas for meas in st.session_state.measurements
            if min_db <= meas["db"] <= max_db and meas["material"] in selected_materials
        ]

        st.session_state.filtered_measurements = filtered_measurements
        st.write(f"フィルタリング後の計測値: {len(filtered_measurements)} 件")

        return filtered_measurements
    else:
        st.info("現在、計測値は登録されていません。")
        return []

# ヘルプセクション
def help_section():
    st.markdown("### ヘルプ")
    st.markdown("""
    **防災スピーカ音圧ヒートマップアプリケーションへようこそ！**

    - **データのアップロード**
        - サイドバーからXML形式のDEMデータとGeoJSON形式の建物データをアップロードしてください。

    - **スピーカーの追加**
        - 新しいスピーカーを手動で追加するか、CSVファイルからインポートできます。

    - **計測値の追加**
        - 計測位置の緯度・経度、dB値、材質を入力して新しい計測値を追加できます。

    - **材質の管理**
        - 音圧減衰量を指定して新しい材質を追加できます。

    - **スピーカーおよび計測値の管理**
        - 既存のスピーカーや計測値を編集・削除できます。

    - **ヒートマップの設定**
        - ヒートマップの視覚的なパラメータを調整できます。

    - **アプリケーションの状態管理**
        - 現在の設定を保存し、後で読み込むことができます。

    - **計測値のフィルタリング**
        - dB値や材質に基づいて計測値をフィルタリングできます。

    - **CSVのエクスポート**
        - スピーカーと計測値をCSVファイルとしてエクスポートできます。

    **操作手順**
    1. 必要なデータをアップロードします。
    2. スピーカーや計測値を追加・管理します。
    3. ヒートマップの設定を調整します。
    4. ヒートマップを更新して結果を確認します。
    5. 必要に応じてデータをエクスポートします。

    **サポート**
    問題が発生した場合や質問がある場合は、サポートまでお問い合わせください。
    """)

# ヒートマップの再計算関数
def recalculate_heatmap():
    if st.session_state.speakers and st.session_state.dem_data is not None and not st.session_state.buildings.empty:
        with st.spinner("音圧計算中..."):
            st.session_state.heatmap_data, st.session_state.contours, sound_grid_smoothed = calculate_heatmap_and_contours_with_obstruction(
                st.session_state.speakers,
                st.session_state.L0,
                st.session_state.r_max,
                st.session_state.grid_lat,
                st.session_state.grid_lon,
                st.session_state.buildings,
                st.session_state.dem_data,
                st.session_state.dem_transform
            )
        st.success("ヒートマップと等高線を再計算しました。")

# Streamlitメイン表示
st.title("防災スピーカ音圧ヒートマップ（地形・建物考慮版）")

# DEMデータのアップロード
st.sidebar.header("データのアップロード")
uploaded_dem = st.sidebar.file_uploader("DEMデータをアップロード（XML形式）", type=["xml"])
uploaded_buildings = st.sidebar.file_uploader("建物データをアップロード（GeoJSON形式）", type=["geojson"])

if uploaded_dem and uploaded_buildings:
    # DEMデータの読み込み
    dem_data, dem_transform = load_dem_xml(uploaded_dem)
    if dem_data.size > 0 and dem_transform is not None:
        st.sidebar.success("DEMデータを読み込みました。")
        st.session_state.dem_data = dem_data
        st.session_state.dem_transform = dem_transform
    else:
        st.sidebar.error("DEMデータの読み込みに失敗しました。")

    # 建物データの読み込み
    buildings = load_buildings(uploaded_buildings)
    if not buildings.empty:
        st.sidebar.success("建物データを読み込みました。")
        st.session_state.buildings = buildings
    else:
        st.sidebar.error("建物データの読み込みに失敗しました。")

    # 地図用グリッドの設定
    lat_min = st.session_state.map_center[0] - 0.01
    lat_max = st.session_state.map_center[0] + 0.01
    lon_min = st.session_state.map_center[1] - 0.01
    lon_max = st.session_state.map_center[1] + 0.01

    zoom_factor = 200 + (st.session_state.map_zoom - 14) * 30  # グリッド解像度の調整
    zoom_factor = min(max(200, zoom_factor), 1000)  # 最低200、最高1000に設定
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, zoom_factor),
        np.linspace(lon_min, lon_max, zoom_factor)
    )
    grid_lat = grid_lat.T
    grid_lon = grid_lon.T

    st.session_state.grid_lat = grid_lat
    st.session_state.grid_lon = grid_lon

    # 初回ヒートマップ計算
    if st.session_state.heatmap_data is None and st.session_state.speakers:
        with st.spinner("音圧計算中..."):
            st.session_state.heatmap_data, st.session_state.contours, sound_grid_smoothed = calculate_heatmap_and_contours_with_obstruction(
                st.session_state.speakers,
                st.session_state.L0,
                st.session_state.r_max,
                grid_lat,
                grid_lon,
                buildings,
                dem_data,
                dem_transform
            )
        st.success("音圧計算が完了しました。")

    # Foliumマップ生成
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom
    )

    # DEMと建物をマップに追加
    add_dem_and_buildings_to_map(m, dem_data, dem_transform, buildings)

    # スピーカー・ピン配置 (アイコンとフォント調整)
    for idx, spk in enumerate(st.session_state.speakers):
        lat_s, lon_s, dirs = spk["lat"], spk["lon"], spk["directions"]
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

    # 計測値・ピン配置 (アイコン+理論値と家内音圧をポップアップに追加表示)
    filtered_measurements = st.session_state.measurements  # 初期は全て表示
    if "filtered_measurements" in st.session_state:
        filtered_measurements = st.session_state.filtered_measurements

    for idx, meas in enumerate(filtered_measurements):
        lat_m, lon_m, db_m, material = meas["lat"], meas["lon"], meas["db"], meas["material"]
        # 理論値を計算
        theoretical_db = calc_theoretical_db_with_obstruction(
            lat_m, lon_m,
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            buildings,
            dem_data,
            dem_transform
        )
        if theoretical_db is not None:
            theo_str = f"{theoretical_db:.2f} dB"
            # 家内音圧の計算
            indoor_db = calc_indoor_db(theoretical_db, material)
            indoor_str = f"{indoor_db:.2f} dB"
        else:
            theo_str = "N/A"
            indoor_str = "N/A"

        popup_html = f"""
        <div style="font-size:14px;">
          <b>計測位置:</b> ({lat_m:.6f}, {lon_m:.6f})<br>
          <b>計測値:</b> {db_m:.2f} dB<br>
          <b>理論値:</b> {theo_str}<br>
          <b>家内音圧:</b> {indoor_str}<br>
          <b>材質:</b> {material}
        </div>
        """

        folium.Marker(
            location=[lat_m, lon_m],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(icon="info-circle", prefix="fa", color="green")
        ).add_to(m)

    # ヒートマップ・等高線
    if st.session_state.heatmap_data:
        HeatMap(
            st.session_state.heatmap_data,
            radius=getattr(st.session_state, 'heatmap_radius', 15),
            blur=getattr(st.session_state, 'heatmap_blur', 20),
            min_opacity=getattr(st.session_state, 'heatmap_min_opacity', 0.4),
            gradient=getattr(st.session_state, 'heatmap_gradient', {0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'red'})
        ).add_to(m)

    for key, level in {"L0-20dB": st.session_state.L0 - 20, "L0-1dB": st.session_state.L0 - 1}.items():
        for contour in st.session_state.contours.get(key, []):
            color = "green" if key == "L0-20dB" else "red"
            folium.PolyLine(locations=contour, color=color, weight=2).add_to(m)

    # マップに描画ツールを追加
    draw = Draw(
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "rectangle": False,
            "marker": True,
            "circlemarker": False,
        },
        edit=True,
        remove=True,
    )
    draw.add_to(m)

    # マップをStreamlitに表示し、移動状況を取得
    st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom", "all_drawings"])
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

            zoom_factor = 200 + (st.session_state.map_zoom - 14) * 30
            zoom_factor = min(max(200, zoom_factor), 1000)
            grid_lat, grid_lon = np.meshgrid(
                np.linspace(lat_min, lat_max, zoom_factor),
                np.linspace(lon_min, lon_max, zoom_factor)
            )
            grid_lat = grid_lat.T
            grid_lon = grid_lon.T

            st.session_state.grid_lat = grid_lat
            st.session_state.grid_lon = grid_lon

            # ヒートマップ再計算
            if st.session_state.speakers and uploaded_dem and uploaded_buildings:
                with st.spinner("ヒートマップ再計算中..."):
                    st.session_state.heatmap_data, st.session_state.contours, sound_grid_smoothed = calculate_heatmap_and_contours_with_obstruction(
                        st.session_state.speakers,
                        st.session_state.L0,
                        st.session_state.r_max,
                        grid_lat,
                        grid_lon,
                        buildings,
                        dem_data,
                        dem_transform
                    )
                st.success("ヒートマップと等高線を再計算しました。")

# データのアップロードが完了していない場合
else:
    st.warning("DEMデータと建物データをサイドバーからアップロードしてください。")

# ─────────────────────────────────────────────────────────────────────────
# 操作パネル
# ─────────────────────────────────────────────────────────────────────────
if uploaded_dem and uploaded_buildings:
    st.subheader("操作パネル")

    # CSVアップロード
    st.markdown("### CSVのインポート")
    uploaded_file = st.file_uploader("CSVをアップロード (種別/緯度/経度/データ1..4列)", type=["csv"])
    if uploaded_file:
        speakers_loaded, measurements_loaded = load_csv(uploaded_file)
        if speakers_loaded:
            st.session_state.speakers.extend(speakers_loaded)
            st.success(f"{len(speakers_loaded)} 件のスピーカを追加しました。")
        if measurements_loaded:
            st.session_state.measurements.extend(measurements_loaded)
            st.success(f"{len(measurements_loaded)} 件の計測値を追加しました。")
        if st.session_state.speakers:
            recalculate_heatmap()

    # スピーカーおよび計測値の管理
    manage_entities()

    # 新規スピーカー追加
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
                st.session_state.speakers.append({"lat": lat_spk, "lon": lon_spk, "directions": dirs_spk})
                if uploaded_dem and uploaded_buildings:
                    recalculate_heatmap()
                st.success(f"スピーカを追加しました: ({lat_spk}, {lon_spk}), 方向={', '.join([f'{d}°' for d in dirs_spk])}")
        except Exception as e:
            st.error(f"入力エラー: {e}")

    # 新規計測値追加（ドロップダウンメニュー使用）
    st.markdown("### 新しい計測値を追加")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        lat_input = st.text_input("緯度", placeholder="例: 34.2578", key="new_meas_lat")
    with col2:
        lon_input = st.text_input("経度", placeholder="例: 133.2075", key="new_meas_lon")
    with col3:
        db_input = st.text_input("dB", placeholder="例: 75", key="new_meas_db")
    with col4:
        material_input = st.selectbox("材質", list(st.session_state.MATERIAL_ATTENUATION.keys()), key="new_meas_material")
    if st.button("計測値を追加"):
        try:
            lat_m = float(lat_input.strip())
            lon_m = float(lon_input.strip())
            db_m = float(db_input.strip())
            material = material_input.strip()
            st.session_state.measurements.append({"lat": lat_m, "lon": lon_m, "db": db_m, "material": material})
            if uploaded_dem and uploaded_buildings:
                recalculate_heatmap()
            st.success(f"計測値を追加しました: ({lat_m}, {lon_m}), {db_m} dB, 材質={material}")
        except Exception as e:
            st.error(f"入力エラー: {e}")

    # 材質の管理
    st.markdown("### 材質の管理")
    col1, col2 = st.columns(2)
    with col1:
        new_material = st.text_input("新しい材質名", placeholder="例: タイル", key="new_material_name")
    with col2:
        new_attenuation = st.number_input("減衰量 (dB)", min_value=0, max_value=100, step=1, key="new_material_attenuation")
    if st.button("材質を追加"):
        if new_material and new_attenuation:
            if new_material in st.session_state.MATERIAL_ATTENUATION:
                st.warning(f"材質 '{new_material}' は既に存在します。")
            else:
                st.session_state.MATERIAL_ATTENUATION[new_material] = new_attenuation
                st.success(f"材質 '{new_material}' を追加しました。減衰量: {new_attenuation} dB")
        else:
            st.error("材質名と減衰量を入力してください。")

    # ヒートマップの詳細設定
    heatmap_settings()

    # アプリケーションの状態保存と読み込み
    save_load_state()

    # フィルタリング機能
    filtered_measurements = filter_measurements()

    # リセットボタン
    st.markdown("### データのリセット")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("スピーカをリセット"):
            st.session_state.speakers = []
            if uploaded_dem and uploaded_buildings:
                recalculate_heatmap()
            st.success("スピーカをリセットしました。ヒートマップを更新しました。")
    with col2:
        if st.button("計測値をリセット"):
            st.session_state.measurements = []
            if uploaded_dem and uploaded_buildings:
                recalculate_heatmap()
            st.success("計測値をリセットしました。ヒートマップを更新しました。")

    # 更新ボタン: ヒートマップ再計算
    if st.button("ヒートマップを更新"):
        if st.session_state.speakers and uploaded_dem and uploaded_buildings:
            recalculate_heatmap()
        else:
            st.error("スピーカがありません。または、DEM・建物データをアップロードしてください。")

    # エクスポート
    st.markdown("### CSVのエクスポート (種別/緯度/経度/データ1..4形式)")
    if st.button("CSVをエクスポート"):
        csv_data = export_to_csv(st.session_state.speakers, st.session_state.measurements)
        st.download_button(
            label="CSVファイルのダウンロード",
            data=csv_data,
            file_name="sound_map_data.csv",
            mime="text/csv"
        )

    # 凡例バー
    st.markdown("### 音圧レベルの凡例")
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

    # ヘルプセクション
    help_section()
