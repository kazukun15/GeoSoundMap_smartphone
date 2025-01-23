import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import io
import branca.colormap as cm

# ─────────────────────────────────────────────────────────────────────────
# 初期設定
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
# CSVエクスポート用の関数
# ─────────────────────────────────────────────────────────────────────────
def export_to_csv(data, columns):
    """任意のデータリストをCSV形式でエクスポート"""
    try:
        # データ形式を検証
        if not all(isinstance(row, (list, tuple)) for row in data):
            raise ValueError("データの形式が正しくありません。リストのリスト形式で指定してください。")

        df = pd.DataFrame(data, columns=columns)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")
    except Exception as e:
        st.error(f"CSVエクスポート中にエラーが発生しました: {e}")
        return b""  # 空のバイト列を返す

# ─────────────────────────────────────────────────────────────────────────
# 操作パネル
# ─────────────────────────────────────────────────────────────────────────
st.title("防災スピーカー音圧可視化マップ")

# CSVエクスポート
st.subheader("データのエクスポート")
col1, col2 = st.columns(2)

with col1:
    try:
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
    except ValueError as e:
        st.error(f"スピーカーCSVのエクスポートでエラー: {e}")

with col2:
    try:
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
    except ValueError as e:
        st.error(f"計測値CSVのエクスポートでエラー: {e}")

# ─────────────────────────────────────────────────────────────────────────
# 地図の表示
# ─────────────────────────────────────────────────────────────────────────
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom
)

# スピーカーのマーカー
for spk in st.session_state.speakers:
    lat, lon, dirs = spk
    popup_text = f"""
    <b>スピーカー</b>: ({lat:.6f}, {lon:.6f})<br>
    <b>初期音圧レベル:</b> {st.session_state.L0} dB<br>
    <b>最大伝播距離:</b> {st.session_state.r_max} m
    """
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color="blue")
    ).add_to(m)

# 計測値のマーカー
for meas in st.session_state.measurements:
    lat, lon, db = meas
    popup_text = f"""
    <b>計測位置:</b> ({lat:.6f}, {lon:.6f})<br>
    <b>計測値:</b> {db:.2f} dB
    """
    folium.Marker(
        location=[lat, lon],
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

# 地図を表示
st_data = st_folium(m, width=700, height=500, returned_objects=["center", "zoom"])

# 地図の中心・ズームを更新
if st_data:
    if "center" in st_data:
        st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    if "zoom" in st_data:
        st.session_state.map_zoom = st_data["zoom"]
