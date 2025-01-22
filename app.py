import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import math

# 初期設定
if "map_center" not in st.session_state:
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]

if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 17

if "speakers" not in st.session_state:
    st.session_state.speakers = [[34.25741795269067, 133.20450105700033, [0.0, 0.0]]]

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None

# 関数: ヒートマップデータ計算
def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, dirs = spk
        for i in range(Nx):
            for j in range(Ny):
                r = math.sqrt((grid_lat[i, j] - lat) ** 2 + (grid_lon[i, j] - lon) ** 2) * 111320
                if r == 0: r = 1
                power = 10 ** ((L0 - 20 * math.log10(r)) / 10)
                power_sum[i, j] += power
    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] for i in range(Nx) for j in range(Ny) if not np.isnan(sound_grid[i, j])]
    return heat_data

# サイドバー
st.sidebar.title("操作パネル")
L0 = st.sidebar.slider("初期音圧レベル (dB)", 50, 100, 80)
r_max = st.sidebar.slider("最大伝播距離 (m)", 100, 2000, 500)

new_speaker = st.sidebar.text_input("スピーカー (緯度,経度) 例: 34.25,133.20")
if st.sidebar.button("スピーカー追加"):
    try:
        lat, lon = map(float, new_speaker.split(","))
        st.session_state.speakers.append([lat, lon, [0.0]])
        st.session_state.heatmap_data = None
        st.sidebar.success("スピーカーを追加しました")
    except:
        st.sidebar.error("入力が正しくありません")

if st.sidebar.button("スピーカーをリセット"):
    st.session_state.speakers = []
    st.session_state.heatmap_data = None

# 地図サイズ
container = st.container()
map_width = st.sidebar.slider("地図の幅 (px)", 300, 1200, 600)
map_height = st.sidebar.slider("地図の高さ (px)", 300, 800, 400)

# ヒートマップデータ計算
lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))

if st.session_state.heatmap_data is None:
    st.session_state.heatmap_data = calculate_heatmap(st.session_state.speakers, L0, r_max, grid_lat, grid_lon)

# 地図表示
with container:
    st.subheader("ヒートマップ表示")
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
    HeatMap(st.session_state.heatmap_data, radius=15).add_to(m)
    st_data = st_folium(m, width=map_width, height=map_height, returned_objects=["center", "zoom"])

# 地図の中心とズームを更新
if st_data and "center" in st_data:
    st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
if st_data and "zoom" in st_data:
    st.session_state.map_zoom = st_data["zoom"]
