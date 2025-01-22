import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import branca.colormap as cm
import json
from skimage import measure
import math

# -------------------------------------------------------------
# 1) セッションステートの初期化 (初回のみ)
# -------------------------------------------------------------
if "map_center" not in st.session_state:
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 17

if "speakers" not in st.session_state:
    st.session_state.speakers = [[34.25741795269067, 133.20450105700033, [0.0, 0.0]]]

if "measurements" not in st.session_state:
    st.session_state.measurements = []

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None
if "iso_60" not in st.session_state:
    st.session_state.iso_60 = []
if "iso_80" not in st.session_state:
    st.session_state.iso_80 = []

if "prev_l0" not in st.session_state:
    st.session_state.prev_l0 = None
if "prev_r_max" not in st.session_state:
    st.session_state.prev_r_max = None

# -------------------------------------------------------------
# 2) 関数の定義
# -------------------------------------------------------------
def parse_direction_to_degrees(dir_str):
    dir_str = dir_str.strip().upper()
    mapping = {
        "N": 0, "NE": 45, "E": 90, "SE": 135,
        "S": 180, "SW": 225, "W": 270, "NW": 315
    }
    return mapping.get(dir_str, float(dir_str))

def fix_speakers_dir(speakers):
    fixed = []
    for spk in speakers:
        lat, lon, dir_list = spk
        dir_fixed = [parse_direction_to_degrees(d) for d in dir_list]
        fixed.append([lat, lon, dir_fixed])
    return fixed

def calc_bearing_deg(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    diff_lon_rad = math.radians(lon2 - lon1)
    x = math.sin(diff_lon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(diff_lon_rad)
    bearing = math.atan2(x, y)
    return (math.degrees(bearing) + 360) % 360

def directivity_factor(bearing_spk_to_point, horn_angle, half_angle=60):
    diff = abs((bearing_spk_to_point - horn_angle + 180) % 360 - 180)
    return 1.0 if diff <= half_angle else 0.001

def calculate_heatmap_and_contours(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny), dtype=float)
    for spk in speakers:
        lat, lon, dir_list = spk
        for i in range(Nx):
            for j in range(Ny):
                r = math.sqrt((grid_lat[i, j] - lat)**2 + (grid_lon[i, j] - lon)**2) * 111320
                if r == 0: r = 1
                bearing_deg = calc_bearing_deg(lat, lon, grid_lat[i, j], grid_lon[i, j])
                power = sum(directivity_factor(bearing_deg, dir, 60) * 10**((L0 - 20 * math.log10(r)) / 10) for dir in dir_list)
                power_sum[i, j] += power
    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] for i in range(Nx) for j in range(Ny) if not np.isnan(sound_grid[i, j])]
    return heat_data

# -------------------------------------------------------------
# 3) サイドバーとレイアウト
# -------------------------------------------------------------
st.title("スマートフォン対応版 音圧シミュレーター")

col1, col2 = st.columns(2)
with col1:
    st.subheader("スピーカー追加")
    new_speaker = st.text_input("緯度,経度,ホーン1,ホーン2 (例: 34.2579,133.2072,N,SW)")
    if st.button("スピーカーを追加"):
        try:
            lat, lon, dir1, dir2 = new_speaker.split(",")
            lat, lon = float(lat.strip()), float(lon.strip())
            dir1, dir2 = parse_direction_to_degrees(dir1.strip()), parse_direction_to_degrees(dir2.strip())
            st.session_state.speakers.append([lat, lon, [dir1, dir2]])
            st.session_state.heatmap_data = None
            st.success(f"スピーカーを追加: ({lat}, {lon})")
        except Exception:
            st.error("入力形式が正しくありません")

with col2:
    st.subheader("スピーカー削除")
    if st.session_state.speakers:
        spk_idx = st.selectbox("削除するスピーカー", range(len(st.session_state.speakers)), format_func=lambda i: f"{i + 1}: {st.session_state.speakers[i]}")
        if st.button("削除"):
            st.session_state.speakers.pop(spk_idx)
            st.session_state.heatmap_data = None
            st.success("スピーカーを削除しました")

L0 = st.slider("初期音圧レベル (dB)", 50, 100, 80)
r_max = st.slider("最大伝播距離 (m)", 100, 2000, 500)

# -------------------------------------------------------------
# 4) 地図の表示
# -------------------------------------------------------------
lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))

if st.session_state.heatmap_data is None:
    st.session_state.heatmap_data = calculate_heatmap_and_contours(st.session_state.speakers, L0, r_max, grid_lat, grid_lon)

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
HeatMap(st.session_state.heatmap_data, radius=15).add_to(m)
st_data = st_folium(m, width=600, height=400, returned_objects=["center", "zoom"])

if st_data and "center" in st_data:
    st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
if st_data and "zoom" in st_data:
    st.session_state.map_zoom = st_data["zoom"]

