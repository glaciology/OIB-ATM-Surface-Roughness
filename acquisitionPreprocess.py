# Download command (run in directory where you want data saved)
wget --http-user=<username> --http-password=<password> --load-cookies
~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies
--no-check-certificate --auth-no-challenge=on -r --reject "index.html*"
--reject "*.xml" -q -np -e robots=off
https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/ILATM2.002/
# Imports
import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import os
# Function to process each day’s .csv files
def day(fname):
files=glob.glob(fname+’/*.csv’) # Get files
if len(files)==0: # If none found, exit
return
date=fname.split(’/’)[-1].split(’.’) # Get date
format_dif=0
if date[0]==’2019’:
format_dif=1
os.makedirs(’../../oib_ATM_test/{}/{}’.format(date[0],date[1]),
exist_ok=True) # Make dir YYMM
f=open(files[0],’r’) # Open read files
f_new=open(’../../oib_ATM_test/{}/{}/{}.csv’.format(date[0],date[1],
date[2]),’w’) # Open write files
for i in files:
f=open(i,’r’)
if first==False: # If not first, skip header
for j in range(10+format_dif):
f.readline()
else: # If first, write header
for j in range(9+format_dif):
f.readline()
f_new.write(f.readline()[2:-1]+’, Year, Month, Day\n’)
first=False
for j in f: # Add rows + YYMMDD
f_new.write(j[:-1]+’,{},{},{}\n’.format(date[0],date[1],
date[2]))
f.close() # Close read files
f_new.close() # Close write files
# Process all days
files=glob.glob(’./*’)
for i in files:
day(i)
# Move processed files to YYMM folder
files = glob.glob(’./*/*/*.csv’)
for i in files:
name = i.split(’.’)[1].split(’/’)
newname = ’{}{}’.format(name[-2], name[-1]) # Set file name
os.rename(i,’./{}/{}/{}.csv’.format(name[1],name[2],newname))
# Function to concatenate each month and filter to ice extent
def make_years(base_path, output_file, ice_extent_shp):
ice_extent = gpd.read_file(ice_extent_shp) # Load .shp
ice_extent = ice_extent.to_crs(epsg=3413) # Set CRS
month_folders = glob.glob(os.path.join(base_path, ’*/*’))
yearly_data = {}
for month_folder in month_folders:
parts = month_folder.split(os.sep) # Get YYMM
year = parts[-2]
month = parts[-1]
if year not in yearly_data:
yearly_data[year] = pd.DataFrame()
csv_files = glob.glob(os.path.join(month_folder, ’*.csv’))
month_df = pd.DataFrame()
for file in csv_files: # Concatenate folders
df = pd.read_csv(file)
month_df = pd.concat([month_df, df], ignore_index=True)
yearly_data[year] = pd.concat([yearly_data[year], month_df],
ignore_index=True) # Append months to year
for year, data in yearly_data.items():
gdf = gpd.GeoDataFrame(data, geometry =
gpd.points_from_xy(data[’Longitude(deg)’],
data[’Latitude(deg)’]), crs=’epsg:4326’) # Convert to gdf
gdf = gdf.to_crs(epsg=3413) # Set CRS
gdf[’Easting’] = gdf.geometry.x
gdf[’Northing’] = gdf.geometry.y
gdf_in = gpd.sjoin(gdf, ice_extent, op=’within’) # Filter
year_output_dir = os.path.join(output_dir, year) # Save
os.makedirs(year_output_dir, exist_ok=True)
output_file = os.path.join(year_output_dir, f’{year}.csv’)
gdf_in.drop(columns=’geometry’, index=False)
gdf_in.to_csv(output_file, index=False)
print(f"Saved data for {year} to {output_file}")
# Filter and concatenate
ice_extent_shp = ’./greenland_ice_extent/greenland_ice_extent.shp’
base_path = ’./’
output_dir = ’./yearly_data’
os.makedirs(output_dir, exist_ok=True)
make_years(base_path, output_dir, ice_extent_shp)