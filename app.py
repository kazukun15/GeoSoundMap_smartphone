import streamlit as st
import folium
from streamlit_folium import st_folium

st.title("Folium マップのテスト")

# 簡単なFoliumマップを作成
m = folium.Map(location=[34.2574, 133.2045], zoom_start=14)

# マーカーを追加
folium.Marker([34.2574, 133.2045], popup="初期スピーカ").add_to(m)

# マップを表示
st_folium(m, width=700, height=500)
