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
    st.session_state.speakers = []

if "measurements" not in st.session_state:
    st.session_state.measurements = []

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None

if "action_mode" not in st.session_state:
    st.session_state.action_mode = None  # None, "add_speaker", "add_measurement"

# 関数: ヒートマップデータ計算
def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon = spk[:2]
        for i in range(Nx):
            for j in range(Ny):
                r = math.sqrt((grid_lat[i, j] - lat) ** 2 + (grid_lon[i, j] - lon) ** 2) * 111320
                if r == 0: r = 1
                power = 10 ** ((L0 - 20 * math.log10(r)) / 10)
                power_sum[i, j] += power
    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] for i in range(Nx) for j in range(Ny) if not np.isnan(sound_grid[i, j])]
    return heat_data

# 地図の設定
st.title("地図上で操作可能な音圧シミュレーター")
lat_min, lat_max = st.session_state.map_center[0] - 0.01, st.session_state.map_center[0] + 0.01
lon_min, lon_max = st.session_state.map_center[1] - 0.01, st.session_state.map_center[1] + 0.01
grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))

if st.session_state.heatmap_data is None:
    st.session_state.heatmap_data = calculate_heatmap(st.session_state.speakers, 80, 500, grid_lat, grid_lon)

# 地図表示
m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)

# スピーカーと計測値を地図上にプロット
for spk in st.session_state.speakers:
    folium.Marker(location=[spk[0], spk[1]], popup="スピーカー", icon=folium.Icon(color="blue")).add_to(m)

for meas in st.session_state.measurements:
    folium.Marker(location=[meas[0], meas[1]], popup=f"計測値: {meas[2]} dB", icon=folium.Icon(color="green")).add_to(m)

HeatMap(st.session_state.heatmap_data, radius=15).add_to(m)

st_data = st_folium(m, width=700, height=500, returned_objects=["click", "center", "zoom"])

# 地図操作の結果を反映
if st_data:
    if "center" in st_data:
        st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    if "zoom" in st_data:
        st.session_state.map_zoom = st_data["zoom"]
    if "click" in st_data and st_data["click"]:
        clicked_lat, clicked_lon = st_data["click"]["lat"], st_data["click"]["lng"]
        if st.session_state.action_mode == "add_speaker":
            st.session_state.speakers.append([clicked_lat, clicked_lon])
            st.session_state.heatmap_data = None
            st.success(f"スピーカーを追加: ({clicked_lat:.6f}, {clicked_lon:.6f})")
        elif st.session_state.action_mode == "add_measurement":
            meas_value = st.number_input("計測値を入力 (dB)", min_value=0, max_value=120, step=1, value=60)
            st.session_state.measurements.append([clicked_lat, clicked_lon, meas_value])
            st.success(f"計測値を追加: ({clicked_lat:.6f}, {clicked_lon:.6f}), {meas_value} dB")

# 操作パネル
st.subheader("操作パネル")
action = st.radio("操作モード", options=["スピーカーを置く", "計測値を置く", "何もしない"], index=2)

if action == "スピーカーを置く":
    st.session_state.action_mode = "add_speaker"
    st.info("地図をクリックしてスピーカーを追加してください")
elif action == "計測値を置く":
    st.session_state.action_mode = "add_measurement"
    st.info("地図をクリックして計測値を追加してください")
else:
    st.session_state.action_mode = None
    st.info("現在、何も追加しないモードです")

if st.button("スピーカーをリセット"):
    st.session_state.speakers = []
    st.session_state.heatmap_data = None
    st.success("すべてのスピーカーをリセットしました")

if st.button("計測値をリセット"):
    st.session_state.measurements = []
    st.success("すべての計測値をリセットしました")
