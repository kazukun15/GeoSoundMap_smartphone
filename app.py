import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
import math
from skimage import measure
from scipy.ndimage import gaussian_filter
import pandas as pd
import branca.colormap as cm

st.title("DEM と建物データのテストアプリ")

# DEM データのアップロード
uploaded_dem = st.file_uploader("DEMデータをアップロード（.tif形式）", type=["tif"])
uploaded_buildings = st.file_uploader("建物データをアップロード（GeoJSON形式）", type=["geojson"])

if uploaded_dem and uploaded_buildings:
    try:
        # DEM データの読み込み
        with rasterio.open(uploaded_dem) as dem_dataset:
            dem_data = dem_dataset.read(1)
            dem_transform = dem_dataset.transform
        st.success("DEM データを正常に読み込みました。")
        
        # 建物データの読み込み
        buildings = gpd.read_file(uploaded_buildings)
        st.success("建物データを正常に読み込みました。")
        
        # Folium マップの作成
        m = folium.Map(location=[34.2574, 133.2045], zoom_start=14)
        
        # DEM の等高線を追加
        contours = measure.find_contours(dem_data, level=range(int(np.min(dem_data)), int(np.max(dem_data)), 100))
        for contour in contours:
            lat_lon_contour = []
            for y, x in contour:
                lon, lat = rasterio.transform.xy(dem_transform, y, x)
                lat_lon_contour.append((lat, lon))
            if len(lat_lon_contour) > 1:
                folium.PolyLine(locations=lat_lon_contour, color='gray', weight=1, opacity=0.5).add_to(m)
        
        # 建物をマップに追加
        folium.GeoJson(
            buildings,
            style_function=lambda x: {'fillColor': 'orange', 'color': 'black', 'weight': 1, 'fillOpacity': 0.5}
        ).add_to(m)
        
        # マップを表示
        st_data = st_folium(m, width=700, height=500)
        
    except Exception as e:
        st.error(f"データの処理中にエラーが発生しました: {e}")
else:
    st.warning("DEMデータと建物データをアップロードしてください。")
