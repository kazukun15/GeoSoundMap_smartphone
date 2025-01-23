if st.button("ヒートマップを計算して追加"):
    if st.session_state.speakers:
        with st.spinner("ヒートマップを計算中..."):
            heat_data, contours, sound_grid_smoothed = calculate_heatmap_and_contours_with_obstruction(
                st.session_state.speakers,
                st.session_state.L0,
                st.session_state.r_max,
                st.session_state.grid_lat,
                st.session_state.grid_lon,
                st.session_state.buildings,
                st.session_state.dem_data,
                st.session_state.dem_transform
            )
            st.session_state.heatmap_data = heat_data
            st.session_state.contours = contours

        # ヒートマップをマップに追加
        HeatMap(
            st.session_state.heatmap_data,
            radius=getattr(st.session_state, 'heatmap_radius', 15),
            blur=getattr(st.session_state, 'heatmap_blur', 20),
            min_opacity=getattr(st.session_state, 'heatmap_min_opacity', 0.4),
            gradient=getattr(st.session_state, 'heatmap_gradient', {0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'red'})
        ).add_to(m)

        # 等高線をマップに追加
        for key, contour_list in st.session_state.contours.items():
            color = "green" if key == "L0-20dB" else "red"
            for contour in contour_list:
                folium.PolyLine(locations=contour, color=color, weight=2).add_to(m)

        st.success("ヒートマップと等高線をマップに追加しました。")
    else:
        st.error("スピーカーが登録されていません。スピーカーを追加してください。")
