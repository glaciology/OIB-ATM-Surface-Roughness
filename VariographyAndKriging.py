"""
Kriging Interpolation Script for Operation IceBridge Airborne Topographic Mapper L2 Icessn Elevation, Slope, and Roughness, v.2 Data

This script processes geospatial data to perfrom kriging interpolation, 
including data loading, gridding, normal score transformation, variogram
analysis, and plotting results. The aim is to generate a spatial representation 
of surface roughness based on ATM data. 
"""

# Import necessary libraries
import numpy as np 
import pandas as pd
import geopandas as gpd 
import matplotlib.pyplot as plt 
import gstatsim as gs 
import skgstat as skg
import seaborn as sns 
from skgstat import models 
from scipy.spatial.distance import pdist 
from scipy.stats import skew 
from sklearn.preprocessing import QuantileTransformer
from matplotlib.colors import LogNorm
from shapely.geometry import Point
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_data(): 
    """
    Load and prepare shapefiles and CSV data for analyses, visualizations, etc.

    Returns: 
    - ice_extent (gpd.GeoDataFrame): Greenland ice extent GeoDataFrame. 
    - coastline (gpd.GeoDataFrame): Greenland coastline GeoDataFrame. 
    - anomaly (pd.DataFrame): CSV data of kriged anomaly. 
    - atm_gdf (gpd.GeoDataFrame): ATM data in GeoDataFrame format with appropriate CRS. 
    """
    # Define paths to data files 
    ice_extent_path = '/Users/jamiegood/Desktop/OperationIceBridge/greenland_ice_extent/greenland_ice_extent.shp'
    coastline_path = '/Users/jamiegood/Desktop/OperationIceBridge/greenland_ice_extent/greenland_polygon.shp'
    anomaly_path = '/Users/jamiegood/Desktop/OperationIceBridge/krigedData.csv'
    atm_path = '/Users/jamiegood/Desktop/OperationIceBridge/oib_ATM_masked_spring/2009/2009.csv'

    # Load shapefiles and CSV data, converting coordinate reference system to EPSG:3413 (Polar Stereographic)  
    ice_extent = gpd.read_file(ice_extent_path).to_crs('epsg:3413')
    coastline = gpd.read_file(coastline_path).to_crs('epsg:3413')
    anomaly = pd.read_csv(anomaly_path)
    atm_csv = pd.read_csv(atm_path)

    # Convert ATM data to GeoDataFrame and change CRS to ESPG:3413
    atm_gdf = gpd.GeoDataFrame(atm_csv, geometry=gpd.points_from_xy(atm_csv['Longitude(deg)'], atm_csv['Latitude(deg)']), crs='epsg:4326')
    atm_gdf = atm_gdf.to_crs(epsg=3413)

    return ice_extent, coastline, anomaly, atm_gdf

def setup_grid(ice_extent, res=10000): 
    """
    Setup a grid based on the Greenland ice extent shapefile.
    
    Parameters: 
    - ice_extent (gpd.Geodataframe): Greenland ice extent GeoDataFrame. 
    - res (int): Grid resolution in meters (default is 10 km). 
    
    Returns: 
    - Pred_grid_xy (np.ndarray): Array of grid points within the ice extent.
    """
    # Get boudns of the ice extent and create a grid of points.  
    xmin, ymin, xmax, ymax = ice_extent.total_bounds 
    x_points = np.arange(xmin, xmax, res)
    y_points = np.arange(ymin, ymax, res)

    grid_points = [
        (x, y) for x in x_points for y in y_points
        if ice_extent.contains(Point(x, y)).any()
    ]

    return np.array(grid_points)

def filter_data(atm_gdf, ice_extent): 
    """
    Filtered ATM data to points within the ice extent bounds.

    Parameters:
    - atm_gdf (gpd.GeoDataFrame): GeoDataFrame containing ATM data. 
    - ice_extent (gpd.GeoDataFrame): Greenland ice extent GeoDataFrame. 

    Returns: 
    - atm (pd.DataFrame): DataFrame of filtered ATM data with 'X', 'Y', and 'SR' (surface roughness) columns. 
    """
    atm_gdf['Easting'] = atm_gdf.geometry.x
    atm_gdf['Northing'] = atm_gdf.geometry.y 

    # Spatial join to retain points within the ice extent. 
    atm_gdf_ice = gpd.sjoin(atm_gdf, ice_extent, predicate='within')

    # Create a filtered DataFrame for further analysis. 
    atm = pd.DataFrame({
        'X': atm_gdf_ice['Easting'],
        'Y': atm_gdf_ice['Northing'],
        'SR': atm_gdf_ice['RMS_Fit(cm)']
    })

    return atm

def grid_atm(atm): 
    """
    Grid the filtered ATM data to the specified resolution. 

    Parameters: 
    - atm (pd.DataFrame): DataFrame containing X, Y, SR values. 

    Returns: 
    - df_grid (pd.DataFrame): Gridded ATM data. 

    """
    df_grid, _, _, _ = gs.Gridding.grid_data(atm, 'X', 'Y', 'SR', res=10000)
    df_grid = df_grid.dropna().rename(columns={"Z": "SR"})

    return df_grid

def transform_data(df_grid): 
    """
    Apply a normal score transform to the gridded data. 

    Parameters: 
    - df_grid (pd.DataFrame): Gridded ATM data. 

    Returns: 
    - df_grid (pd.DataFrame): Gridded data with an additional 'NSR' (normal score transformed sruface roughness) column. 
    - nst_trans (QuantileTranformer): Transformer used for the normal score transformation. 
    """
    data_reshaped = df_grid['SR'].values.reshape(-1, 1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution='normal').fit(data_reshaped)
    df_grid['NSR'] = nst_trans.transform(data_reshaped)

    return df_grid, nst_trans

def max_distance(df_grid): 
    """

    Compute the maximum Euclidean distance between grid points. 

    Parameters: 
    - df_grid (pd.DataFrame): Gridded data. 

    Returns: 
    - max_distance (float): mMximum euclidean distance between points.
    - maxlag_2 (float): Half of the maximum distance.
    - maxlag_3 (float): One-third of the maximum distance.
    """
    coords = df_grid[['X', 'Y']].values
    max_distance = pdist(coords).max()

    return max_distance, max_distance / 2, max_distance / 3

def plot_histogram(df_grid):
    """
    Plot histogram of SR and NSR.

    Parameters: 
    - df_grid (pd.DataFrame): Gridded data with SR and NSR columns.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df_grid['SR'], kde=True, label='Surface Roughness')
    sns.histplot(df_grid['NSR'], kde=True, label='Normalize')
    plt.title('Distribuation of Surface Roughness Values')
    plt.xlabel('RMS Fit (cm)')
    plt.ylabel('Frequency')
    plt.legend(title='Transformation')
    plt.show()

def check_skew(df_grid):
    """
    Calculate and print skewness of SR and NSR values. 

    Parameters:
    - df_grid (pd.DataFrame): Gridded ATM data. 

    Returns: 
    - sr_skew (float): Skewness of raw SR values.
    - nsr_skew (float): Skewness of normalized SR values.
    """
    sr_skew = skew(df_grid['SR'])
    nsr_skew = skew(df_grid['NSR'])

    print(f"Skewness of SR: {sr_skew:.2f}")
    print(f"Skewness of NSR: {nsr_skew:.2f}")

def variogram_analysis(df_grid, maxlag_3): 
    """
    Perform variogram analysis and fit different variogram models. 

    Parameters: 
    - df_grid (pd.DataFrame): Gridded data with NSR values.
    - maxlag_3 (float): One-third of the maximum lag distance.

    Returns: 
    - V1 (skg.Variogram): Fitted variogram object (Exponential model).
    - xdata (np.ndarray): Experimental variogram lag ditances 
    - ydata (np.ndarray): Experimetnal variogram values .
    - xi (np.ndarray): Lags for model fitting. 
    - y_exp (list): Exponentail variogram model values.
    - y_gauss (list): Gaussian variogram model values 
    - y_sph (list): Spherical variogtam model values 
    """

    coords = df_grid[['X', 'Y']].values
    values = df_grid['NSR']

    V1 = skg.Variogram(coords, values, bin_func='even', n_lags=100, maxlag=maxlag_3, normalize=False)

    xdata = V1.bins
    ydata = V1.experimental

    # Fit various models 
    V1.model = 'exponential'
    xi = np.linspace(0, V1.bins[-1], 100)
    y_exp = [models.exponential(h, V1.cof[0], V1.cof[1], V1.cof[2]) for h in xi]
    y_gauss = [models.gaussian(h, V1.cof[0], V1.cof[1], V1.cof[2]) for h in xi]
    y_sph = [models.spherical(h, V1.cof[0], V1.cof[1], V1.cof[2]) for h in xi]

    return V1, xdata, ydata, xi, y_exp, y_gauss, y_sph

def plot_variogram(xdata, ydata, xi, y_exp, y_gauss, y_sph): 
    """
    Plot the experimental variogram and fitted models. 

    Parameters: 
    - V1 (skg.Variogram): Fitted variogram object.
    - xdata (np.ndarray): Experimental variogram lag distances.
    - ydata (np.ndarray): Experimental variogram values.
    - xi (np.ndarray): Lags for model fitting.
    - y_exp (list): Exponential variogram model values.
    - y_gauss (list): Gaussian variogram model values.
    - y_sph (list): Spherical variogram model values.
    """
    plt.figure()
    plt.plot(xdata / 1000, ydata, 'og', label='Experimental Variogram')
    plt.plot(xi / 1000, y_gauss, 'b--', label='Gaussian Variogram')
    plt.plot(xi / 1000, y_exp, 'b-', label='Exponential Variogram')
    plt.plot(xi/ 1000, y_sph, 'b*-', label='Spherical Variogram')
    plt.title('Isotropic Variogram')
    plt.xlabel('Lag [km]')
    plt.ylabel('Semivariance')
    plt.legend(loc='lower right')
    plt.show() 

def kriging_interp(Pred_grid_xy, df_grid, nst_trans, V1, maxlag_3):
    """
    Perform ordinary kriging interpolation and back-transform the results. 

    Parameters: 
    - Pred_grid_xy (np.ndarray): Array of grid points for prediction.
    - df_grid (pd.DataFrame): DataFrame with gridded data. 
    - nst_trans (QuantileTranforrmer): Transfomer used for normal score transform.
    - V1 (skg.Variogram): Fitted variogram object. 
    - maxlag_3 (float): Maximum lag distance for variogram. 

    Returns: 
    - pred_trans (np.ndarray): Back-transformed predicted surface roughness.
    - std_trans (np.ndarray): Back-transformed standard deviation.
    """
    azimuth = 0 
    nugget = V1.parameters[2] 
    major_range = V1.parameters[0]
    minor_range = V1.parameters[0]
    sill = V1.parameters[1]
    vtype = 'Exponential'
    vario = [azimuth, nugget, major_range, minor_range, sill, vtype]
    k = 100 # Number of neighbors for kriging
    rad = maxlag_3 # Search radius

    # Perform ordinary kriging using GSTools
    est_OK, var_OK = gs.Interpolation.okrige(Pred_grid_xy, df_grid, 'X', 'Y', 'NSR', k, vario, rad)

    # Set negative variance estimates to zero
    var_OK[var_OK < 0] = 0
    std_OK = np.sqrt(var_OK)

    # Reshape estimates and standard deviation to match the grid
    est = est_OK.reshape(-1, 1)
    std = std_OK.reshape(-1, 1)

    # Back-transform the predictions and standard deviations
    pred_trans = nst_trans.inverse_transform(est)
    std_trans = nst_trans.inverse_transform(std) - np.min(nst_trans.inverse_transform(std))

    return pred_trans, std_trans 

def plot_kriging(Pred_grid_xy, pred_trans, df_grid, coastline, anomaly): 
    """
    Plot  the kriged results and anomalies. 

    Parameters: 
    - Pred_grid_xy (np.ndarray): Array of grid points for prediction. 
    - pred_trans (np.ndarray): Back-transformed predicted surface roughness.
    - df_grid (pd.DataFrame): Gridded data for comparison (e.g., flight lines). 
    - coastline (GeoDataFrame): GeoDataFrame of Greenland's coastline. 
    - anomaly (pd.DataFrame): Anomaly values for comparison. 
    """
    # Difference between kriged surface and anomaly
    diff = pred_trans - anomaly

    # Plot kriged surface roughness 
    fig, ax = plt.subplots(figsize=(5, 7))
    norm = LogNorm(vmin=1, vmax=1000)
    coastline.plot(ax=ax, color='gray', linewidth=1, alpha=0.5, zorder=0, label='Greenland Coastline')
    im = ax.scatter(Pred_grid_xy[:, 0], Pred_grid_xy[:, 1], c=pred_trans, norm=norm, marker='.', s=10, cmap='Blues')
    ax.set_title('Ordinary Kriging')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.axis('scaled')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Surface Roughness [cm]', rotation=270, labelpad=15)
    cbar.set_ticks([1, 10, 100, 1000])
    plt.show()

    # Plot kriged surface roughness with flight lines
    fig, ax = plt.subplots(figsize=(5, 7))
    norm = LogNorm(vmin=1, vmax=1000)
    coastline.plot(ax=ax, color='gray', linewidth=1, alpha=0.5, zorder=0, label='Greenland Coastline')
    im = ax.scatter(Pred_grid_xy[:, 0], Pred_grid_xy[:, 1], c=pred_trans, norm=norm, marker='.', s=10, cmap='Blues')
    ax.set_title('Ordinary Kriging with Flightlines')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.axis('scaled')
    ax.scatter(df_grid['X'], df_grid['Y'], c='gray', marker='.', s=1, alpha=0.5, label='Flightlines')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([1, 10, 100, 1000])
    cbar.set_label('Surface Roughness [cm]', rotation=270, labelpad=15)
    plt.legend(loc='upper right')
    plt.show()

    # Plot anomaly map
    fig, ax = plt.subplots(figsize=(5, 7))
    coastline.plot(ax=ax, color='gray', linewidth=1, alpha=0.5, zorder=0, label='Greenland Coastline')
    im = ax.scatter(Pred_grid_xy[:, 0], Pred_grid_xy[:, 1], c=diff['Kriged_SR'], marker='.', s=10, cmap='seismic_r', vmin=-500, vmax=500)
    ax.set_title('Anomaly Map')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.axis('scaled')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Anomaly [cm]', rotation=270, labelpad=15)
    plt.show()

if __name__ == "__main__":
    ice_extent, coastline, anomaly, atm_gdf = load_data()
    Pred_grid_xy = setup_grid(ice_extent)
    atm = filter_data(atm_gdf, ice_extent)
    df_grid = grid_atm(atm)
    df_grid, nst_trans = transform_data(df_grid)
    max_distance, maxlag_2, maxlag_3 = max_distance(df_grid)
    plot_histogram(df_grid)
    check_skew(df_grid)
    V1, xdata, ydata, xi, y_exp, y_gauss, y_sph = variogram_analysis(df_grid, maxlag_3)
    plot_variogram(xdata, ydata, xi, y_exp, y_gauss, y_sph)
    pred_trans, std_trans = kriging_interp(Pred_grid_xy, df_grid, nst_trans, V1, maxlag_3)
    plot_kriging(Pred_grid_xy, pred_trans, df_grid, coastline, anomaly)