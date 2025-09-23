import rasterio
import rasterio.features
import pandas as pd
import numpy as np
from shapely.geometry import box
from rasterio.transform import from_bounds
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import geopandas as gpd

def show_maps_grid(all_cols, map_df, standardize=True, show_plots=True):
    """
    Display grid maps for specified columns.
    
    Parameters:
    -----------
    all_cols : list
        List of column names to plot
    map_df : pandas.DataFrame
        DataFrame containing data with 'row', 'col', and 'center_lat' columns
    standardize : bool, default True
        If True, standardize values to z-scores and clip to ±3 std
        If False, use original data range for color mapping
    show_plots : bool, default True
        If True, display plots with plt.show()
        If False, don't display plots (useful for saving only)
    """
    map_df = map_df.copy()
    
    if "row" not in map_df.columns or "col" not in map_df.columns:
        raise ValueError("map_df must contain 'row' and 'col' columns")
    
    # Infer grid shape
    max_row = map_df["row"].max()
    max_col = map_df["col"].max()
    grid_shape = (max_row + 1, max_col + 1)
    
    lat_mean = map_df['center_lat'].mean()  
    aspect_correction = 1 / np.cos(np.radians(lat_mean))
    
    for col in all_cols:
        values = pd.to_numeric(map_df[col], errors="coerce")
        assert values.notna().any(), f"All values for {col} are NaN"
        
        if standardize:
            # Standardize and clip to ±3 std
            mean = values.mean()
            std = values.std()
            plot_values = ((values - mean) / std).clip(lower=-3, upper=3)
            vmin, vmax = -3, 3
        else:
            # Use original data range
            plot_values = values
            valid_values = values.dropna()
            vmin = valid_values.min()
            vmax = valid_values.max()
        
        # Create empty grid filled with NaN
        grid = np.full(grid_shape, np.nan)
        
        # Place each value at its specified (row, col) position
        for idx, (_, row_data) in enumerate(map_df.iterrows()):
            row_pos = int(row_data["row"])
            col_pos = int(row_data["col"])
            if not pd.isna(plot_values.iloc[idx]):
                grid[row_pos, col_pos] = plot_values.iloc[idx]
        
        # Normalize for color mapping
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('RdBu_r')
        
        width = 10
        height = width * aspect_correction
        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(grid, cmap=cmap, norm=norm, origin='upper', aspect=aspect_correction)
        
        # Add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, shrink=0.35)
        cbar.ax.tick_params(labelsize=14)
        
        ax.axis('off')
        
        if show_plots:
            plt.show()
        
        
def show_maps_geo(all_cols, map_df, standardize=True, show_plots=False):
    """
    Create map visualizations for specified columns using geographic shapes.
    
    Parameters:
    -----------
    all_cols : list
        List of column names to plot
    map_df : pandas.DataFrame
        DataFrame containing data with 'geometry' column
    standardize : bool, default True
        If True, standardize values to z-scores and clip to ±3 std
        If False, use original data range for color mapping
    show_plots : bool, default True
        If True, display plots with plt.show()
        If False, don't display plots (useful for saving only)
    """
    map_df = map_df.copy()
    tolerance = 0.01
    map_df["geometry"] = map_df["geometry"].apply(lambda geom: geom.simplify(tolerance, preserve_topology=True))
    
    for col in all_cols:
        values = pd.to_numeric(map_df[col], errors="coerce")
        assert values.notna().any(), f"All values for {col} are NaN"
        
        if standardize:
            # Standardize and clip to ±3 std
            mean = values.mean()
            std = values.std()
            plot_values = ((values - mean) / std).clip(lower=-3, upper=3)
            vmin, vmax = -3, 3
            colors = plt.get_cmap('RdBu_r')(Normalize(vmin=vmin, vmax=vmax)(plot_values.fillna(0)))  # fallback to 0 for missing
        else:
            # Use original data range
            plot_values = values
            valid_values = values.dropna()
            vmin = valid_values.min()
            vmax = valid_values.max()
            colors = plt.get_cmap('RdBu_r')(Normalize(vmin=vmin, vmax=vmax)(plot_values.fillna(vmin)))  # fallback to vmin for missing
        
        # Normalize for color mapping
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('RdBu_r')
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        map_df.plot(ax=ax, color=colors, edgecolor='black', linewidth=0.0)
        
        # Force matplotlib to draw the figure to get accurate measurements
        fig.canvas.draw()
        
        # Get the actual bounds of the plotted map and axis limits
        map_bounds = map_df.total_bounds  # [minx, miny, maxx, maxy]
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        
        # Calculate the data range vs axis range ratios
        map_height = map_bounds[3] - map_bounds[1]
        axis_height = ax_ylim[1] - ax_ylim[0]
        
        map_width = map_bounds[2] - map_bounds[0]  
        axis_width = ax_xlim[1] - ax_xlim[0]
        
        # Use the smaller ratio to account for padding/margins
        height_ratio = map_height / axis_height
        width_ratio = map_width / axis_width
        shrink = min(height_ratio, width_ratio) * 0.65  # Further reduce
        
        # Add colorbar with height matching the map
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []  # dummy array for colorbar
        cbar = fig.colorbar(sm, ax=ax, shrink=shrink)
        
        ax.axis('off')
        
        if show_plots:
            plt.show()

        
def grid_gdf_quarter_degree(gdf, value_column, bounds=None, output_path=None):
    """
    Grid GeoDataFrame data into 0.25° x 0.25° cells using area-weighted averaging.
    
    Parameters:
    - gdf: GeoDataFrame with geometries and values
    - value_column: column name in gdf containing values to grid
    - bounds: optional tuple (minx, miny, maxx, maxy) to define grid extent.
              If None, uses bounds of the GeoDataFrame
    - output_path: optional path to save the output GeoDataFrame
    
    Returns:
    - grid_gdf: GeoDataFrame with 0.25° grid cells and weighted average values
    """
    
    # Ensure GDF is in a geographic CRS (WGS84)
    if gdf.crs is None:
        # print("Warning: GeoDataFrame has no CRS. Assuming WGS84 (EPSG:4326)")
        gdf = gdf.set_crs('EPSG:4326')
    elif not gdf.crs.is_geographic:
        # print(f"Converting from {gdf.crs} to WGS84 for degree-based gridding")
        gdf = gdf.to_crs('EPSG:4326')
    
    # Remove invalid geometries and values
    gdf_clean = gdf[
        ~gdf.geometry.is_empty & 
        ~pd.isna(gdf[value_column]) &
        gdf.geometry.notna()
    ].copy()
    
    if len(gdf_clean) == 0:
        print("No valid geometries found")
        return gpd.GeoDataFrame()
    
    # Get bounds for grid creation
    if bounds is None:
        total_bounds = gdf_clean.total_bounds
        minx, miny, maxx, maxy = total_bounds
    else:
        minx, miny, maxx, maxy = bounds
    
    # Expand bounds to align with 0.25° grid
    grid_size = 0.25
    minx = np.floor(minx / grid_size) * grid_size
    miny = np.floor(miny / grid_size) * grid_size
    maxx = np.ceil(maxx / grid_size) * grid_size  
    maxy = np.ceil(maxy / grid_size) * grid_size
    
    # print(f"Grid bounds: ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})")
    
    # Create grid cells
    grid_cells = []
    grid_ids = []
    
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)
    
    # print(f"Creating {len(x_coords)} x {len(y_coords)} = {len(x_coords) * len(y_coords)} grid cells")
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append(cell)
            grid_ids.append(f"cell_{i}_{j}")
    
    # Create grid GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({
        'grid_id': grid_ids,
        'geometry': grid_cells
    }, crs='EPSG:4326')
    
    # Add centroid coordinates for reference
    centroids = grid_gdf.geometry.centroid
    grid_gdf['center_lon'] = centroids.x
    grid_gdf['center_lat'] = centroids.y
    
    # Calculate area-weighted averages for each grid cell
    weighted_values = []
    total_areas = []
    geometry_counts = []
    
    # print("Calculating area-weighted averages...")
    
    for idx, grid_cell in enumerate(grid_gdf.geometry):
        # if idx % 1000 == 0:
            # print(f"Processing cell {idx + 1}/{len(grid_gdf)}")
            
        weighted_sum = 0.0
        total_area = 0.0
        geom_count = 0
        
        # Find intersecting geometries using spatial index for efficiency
        possible_matches_index = list(gdf_clean.sindex.intersection(grid_cell.bounds))
        possible_matches = gdf_clean.iloc[possible_matches_index]
        
        for _, row in possible_matches.iterrows():
            geometry = row.geometry
            value = row[value_column]
            
            try:
                # Calculate intersection
                intersection = geometry.intersection(grid_cell)
                
                if not intersection.is_empty:
                    intersection_area = intersection.area
                    weighted_sum += value * intersection_area
                    total_area += intersection_area
                    geom_count += 1
                    
            except Exception as e:
                # Handle any geometry errors silently
                continue
        
        # Calculate weighted average
        if total_area > 0:
            weighted_avg = weighted_sum / total_area
        else:
            weighted_avg = np.nan
            
        weighted_values.append(weighted_avg)
        total_areas.append(total_area)
        geometry_counts.append(geom_count)
    
    # Add results to grid
    grid_gdf['weighted_avg'] = weighted_values
    grid_gdf['total_area'] = total_areas
    grid_gdf['geometry_count'] = geometry_counts
    
    # Rename the weighted average column to match the original column name
    grid_gdf = grid_gdf.rename(columns={'weighted_avg': f'{value_column}'})
    
    # Remove cells with no data if desired (uncomment next line)
    # grid_gdf = grid_gdf[~pd.isna(grid_gdf[f'{value_column}_weighted'])]
    
    # print(f"Grid complete. {(~pd.isna(grid_gdf[f'{value_column}'])).sum()} cells contain data")
    
    # Save output if path provided
    if output_path:
        grid_gdf.to_file(output_path)
        # print(f"Grid saved to {output_path}")
    
    return grid_gdf


def grid_gdf_quarter_degree_str(gdf, value_column, bounds=None, output_path=None, aggregation='largest_area'):
    """
    Grid GeoDataFrame data into 0.25° x 0.25° cells for string values.
    
    Parameters:
    - gdf: GeoDataFrame with geometries and values
    - value_column: column name in gdf containing values to grid
    - bounds: optional tuple (minx, miny, maxx, maxy) to define grid extent.
              If None, uses bounds of the GeoDataFrame
    - output_path: optional path to save the output GeoDataFrame
    - aggregation: method for handling multiple string values in a cell:
                  'most_common' - most frequent value (default)
                  'largest_area' - value from geometry with largest intersection area
                  'first' - first value encountered
                  'concatenate' - concatenate all unique values with separator
    
    Returns:
    - grid_gdf: GeoDataFrame with 0.25° grid cells and aggregated string values
    """
    
    # Ensure GDF is in a geographic CRS (WGS84)
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif not gdf.crs.is_geographic:
        gdf = gdf.to_crs('EPSG:4326')
    
    # Remove invalid geometries and values
    gdf_clean = gdf[
        ~gdf.geometry.is_empty & 
        ~pd.isna(gdf[value_column]) &
        gdf.geometry.notna() &
        (gdf[value_column] != '') &  # Also remove empty strings
        (gdf[value_column].astype(str) != 'nan')  # Remove string 'nan'
    ].copy()
    
    if len(gdf_clean) == 0:
        print("No valid geometries found")
        return gpd.GeoDataFrame()
    
    # Get bounds for grid creation
    if bounds is None:
        total_bounds = gdf_clean.total_bounds
        minx, miny, maxx, maxy = total_bounds
    else:
        minx, miny, maxx, maxy = bounds
    
    # Expand bounds to align with 0.25° grid
    grid_size = 0.25
    minx = np.floor(minx / grid_size) * grid_size
    miny = np.floor(miny / grid_size) * grid_size
    maxx = np.ceil(maxx / grid_size) * grid_size  
    maxy = np.ceil(maxy / grid_size) * grid_size
    
    # Create grid cells
    grid_cells = []
    grid_ids = []
    
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)
    
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            cell = box(x, y, x + grid_size, y + grid_size)
            grid_cells.append(cell)
            grid_ids.append(f"cell_{i}_{j}")
    
    # Create grid GeoDataFrame
    grid_gdf = gpd.GeoDataFrame({
        'grid_id': grid_ids,
        'geometry': grid_cells
    }, crs='EPSG:4326')
    
    # Add centroid coordinates for reference
    centroids = grid_gdf.geometry.centroid
    grid_gdf['center_lon'] = centroids.x
    grid_gdf['center_lat'] = centroids.y
    
    # Add row and col indices for consistent gridding
    grid_gdf['row'] = grid_gdf['grid_id'].str.extract(r'cell_(\d+)_\d+').astype(int)
    grid_gdf['col'] = grid_gdf['grid_id'].str.extract(r'cell_\d+_(\d+)').astype(int)
    
    # Process each grid cell based on aggregation method
    aggregated_values = []
    geometry_counts = []
    
    for idx, grid_cell in enumerate(grid_gdf.geometry):
        possible_matches_index = list(gdf_clean.sindex.intersection(grid_cell.bounds))
        possible_matches = gdf_clean.iloc[possible_matches_index]
        
        intersecting_values = []
        intersecting_areas = []
        
        for _, row in possible_matches.iterrows():
            geometry = row.geometry
            value = str(row[value_column])  # Ensure it's a string
            
            try:
                intersection = geometry.intersection(grid_cell)
                if not intersection.is_empty:
                    intersecting_values.append(value)
                    intersecting_areas.append(intersection.area)
            except Exception:
                continue
        
        # Aggregate based on chosen method
        if len(intersecting_values) == 0:
            final_value = np.nan
        elif aggregation == 'most_common':
            # Get most frequent value
            value_counts = pd.Series(intersecting_values).value_counts()
            final_value = value_counts.index[0]
        elif aggregation == 'largest_area':
            # Get value from geometry with largest intersection
            max_area_idx = np.argmax(intersecting_areas)
            final_value = intersecting_values[max_area_idx]
        elif aggregation == 'first':
            # Get first value encountered
            final_value = intersecting_values[0]
        elif aggregation == 'concatenate':
            # Concatenate all unique values
            unique_values = list(set(intersecting_values))
            final_value = ' | '.join(sorted(unique_values))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        aggregated_values.append(final_value)
        geometry_counts.append(len(intersecting_values))
    
    # Add results to grid
    grid_gdf[value_column] = aggregated_values
    grid_gdf['geometry_count'] = geometry_counts
    
    # Save output if path provided
    if output_path:
        grid_gdf.to_file(output_path)
    
    return grid_gdf