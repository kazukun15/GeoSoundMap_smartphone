def calculate_heatmap_and_contours_with_obstruction(speakers, L0, r_max, grid_lat, grid_lon, buildings, dem_data, dem_transform):
    Nx, Ny = grid_lat.shape
    sound_grid = np.full((Nx, Ny), np.nan, dtype=float)

    # プログレスバーの設定
    progress_bar = st.progress(0)
    total = Nx

    for i in range(Nx):
        for j in range(Ny):
            lat = grid_lat[i, j]
            lon = grid_lon[i, j]
            db = calc_theoretical_db_with_obstruction(lat, lon, speakers, L0, r_max, buildings, dem_data, dem_transform)
            if db is not None:
                sound_grid[i, j] = db
        # プログレスバーの更新
        progress_bar.progress((i + 1) / total)

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
    # st.write(f"Sound Grid Min: {min_db:.2f} dB")  # コメントアウト
    # st.write(f"Sound Grid Max: {max_db:.2f} dB")  # コメントアウト

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
