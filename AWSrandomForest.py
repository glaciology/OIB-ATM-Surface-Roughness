# Imports
import math
import numpy as np
import pandas as pd
import seaborn as sns
import pypromice as pp
from scipy import stats
import pypromice.get as pget
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.feature_selection import RFE, SelectFromModel
import geopandas as gpd
from matplotlib.colors import LogNorm
from sklearn.linear_model import LinearRegression
stations = {
"cen1": ("cen1_day.csv", "./CEN/CEN.csv"),
"cp1": ("cp1_day.csv", "./Crawford Point/Crawford Point.csv"),
"dy2": ("dy2_day.csv", "./DYE-II/DYE-II.csv"),
"egp": ("egp_day.csv", "./EGP/EGP.csv"),
"hum": ("hum_day.csv", "./Humboldt/Humboldt.csv"),
"jar": ("jar_day.csv", "./JAR/JAR.csv"),
"kan_b": ("kan_b_day.csv", "./KAN_B/KAN_B.csv"),
"kan_l": ("kan_l_day.csv", "./KAN_L/KAN_L.csv"),
"kan_m": ("kan_m_day.csv", "./KAN_M/KAN_M.csv"),
"kan_u": ("kan_u_day.csv", "./KAN_U/KAN_U.csv"),
"kpc_l": ("kpc_l_day.csv", "./KPC_L/KPC_L.csv"),
"kpc_u": ("kpc_u_day.csv", "./KPC_U/KPC_U.csv"),
"nem": ("nem_day.csv", "./NEEM/NEEM.csv"),
"nse": ("nse_day.csv", "./NASA-SE/NASA-SE.csv"),
"nuk_l": ("nuk_l_day.csv", "./NUK_L/NUK_L.csv"),
"nuk_n": ("nuk_n_day.csv", "./NUK_N/NUK_N.csv"),
"nuk_u": ("nuk_u_day.csv", "./NUK_U/NUK_U.csv"),
"qas_l": ("qas_l_day.csv", "./QAS_L/QAS_L.csv"),
"qas_m": ("qas_m_day.csv", "./QAS_M/QAS_M.csv"),
"qas_u": ("qas_u_day.csv", "./QAS_U/QAS_U.csv"),
"sdl": ("sdl_day.csv", "./Saddle/Saddle.csv"),
"sco_l": ("sco_l_day.csv", "./SCO_L/SCO_L.csv"),
"sco_u": ("sco_u_day.csv", "./SCO_U/SCO_U.csv"),
"swc": ("swc_day.csv", "./Swiss Camp/Swiss Camp.csv"),
"tas_u": ("tas_u_day.csv", "./TAS_U/TAS_U.csv"),
"thu_l": ("thu_l_day.csv", "./THU_L/THU_L.csv"),
"thu_u": ("thu_u_day.csv", "./THU_U/THU_U.csv"),
"thu_u2": ("thu_u2_day.csv", "./THU_U2/THU_U2.csv"),
"tun": ("tun_day.csv", "./Tunu-N/Tunu-N.csv"),
"upe_u": ("upe_u_day.csv", "/UPE_U/UPE_U.csv")
}
# Function to load AWS/ATM data for each station
def load_station_data(station_name, station_file):
aws_data_file, atm_file = station_file
aws_data = pget.aws_data(aws_data_file) # Get AWS data with pyromice API
aws_data.index = pd.to_datetime(aws_data.index)
atm_data = pd.read_csv(atm_file) # Load local ATM
atm_data[’Date’] = pd.to_datetime(atm_data[[’Year’, ’Month’, ’Day’]])
return aws_data, atm_data
station_data = {}
for station, file in stations.items():
aws_data, atm_data = load_station_data(station, file)
station_data[station] = {’aws’: aws_data, ’atm’: atm_data}
day_offsets = [(0, 7), (7, 14), (14, 21)] # (start_day, end_day) pairs
variables = [’p_u’, ’t_u’, ’rh_u_cor’, ’qh_u’, ’wspd_u’,
’dsr_cor’, ’cc’, ’dlhf_u’, ’dshf_u’,
’wdir_u’, ’albedo’, ’t_surf’, ’z_boom_u’]
def process_dataset(atm_data, dataset, day_offsets, variables):
comp_list = []
for date in atm_data[’Date’].unique():
data_for_date = {’Date’: date}
for start_offset, end_offset in day_offsets:
start_range = date - timedelta(days=end_offset)
end_range = date - timedelta(days=start_offset)
filt_df = dataset[(dataset.index >= start_range) &
(dataset.index <= end_range)]
for var in variables:
if var in dataset.columns:
if var != ’z_boom_u’:
data_for_date[f’{var}_{end_offset}_avg’] =
filt_df[var].mean()
data_for_date[f’{var}_{end_offset}_max’] =
filt_df[var].max()
data_for_date[f’{var}_{end_offset}_min’] =
filt_df[var].min()
if var == ’z_boom_u’:
data_for_date[f’{var}_{end_offset}_diff’] =
filt_df[var].max() - filt_df[var].min()
else:
data_for_date[f’{var}_{end_offset}_stats’] = float(’nan’)
comp_list.append(pd.DataFrame([data_for_date]))
comp_df = pd.concat(comp_list, ignore_index=True)
rms_df = atm_data.groupby(’Date’)[’RMS_Fit(cm)’].mean().reset_index()
height_df = atm_data.groupby(’Date’)[’WGS84_Ellipsoid_Height(m)’].mean()
.reset_index()
comp_df = pd.merge(comp_df, rms_df, on=’Date’, how=’left’)
comp_df = pd.merge(comp_df, height_df, on=’Date’, how=’left’)
return comp_df
all_elev_datasets = {
’cen’: (station_data[’cen1’][’atm’], station_data[’cen1’][’aws’]),
’cp’: (station_data[’cp1’][’atm’], station_data[’cp1’][’aws’]),
’dy2’: (station_data[’dy2’][’atm’], station_data[’dy2’][’aws’]),
’egp’: (station_data[’egp’][’atm’], station_data[’egp’][’aws’]),
’hum’: (station_data[’hum’][’atm’], station_data[’hum’][’aws’]),
’jar’: (station_data[’jar’][’atm’], station_data[’jar’][’aws’]),
’kan_b’: (station_data[’kan_b’][’atm’], station_data[’kan_b’][’aws’]),
’kan_l’: (station_data[’kan_l’][’atm’], station_data[’kan_l’][’aws’]),
’kan_m’: (station_data[’kan_m’][’atm’], station_data[’kan_m’][’aws’]),
’kan_u’: (station_data[’kan_u’][’atm’], station_data[’kan_u’][’aws’]),
’kpc_l’: (station_data[’kpc_l’][’atm’], station_data[’kpc_l’][’aws’]),
’kpc_u’: (station_data[’kpc_u’][’atm’], station_data[’kpc_u’][’aws’]),
’nem’: (station_data[’nem’][’atm’], station_data[’nem’][’aws’]),
’nse’: (station_data[’nse’][’atm’], station_data[’nse’][’aws’]),
’nuk_l’: (station_data[’nuk_l’][’atm’], station_data[’nuk_l’][’aws’]),
’nuk_n’: (station_data[’nuk_n’][’atm’], station_data[’nuk_n’][’aws’]),
’nuk_u’: (station_data[’nuk_u’][’atm’], station_data[’nuk_u’][’aws’]),
’qas_l’: (station_data[’qas_l’][’atm’], station_data[’qas_l’][’aws’]),
’qas_m’: (station_data[’qas_m’][’atm’], station_data[’qas_m’][’aws’]),
’qas_u’: (station_data[’qas_u’][’atm’], station_data[’qas_u’][’aws’]),
’sdl’: (station_data[’sdl’][’atm’], station_data[’sdl’][’aws’]),
’sco_l’: (station_data[’sco_l’][’atm’], station_data[’sco_l’][’aws’]),
’sco_u’: (station_data[’sco_u’][’atm’], station_data[’sco_u’][’aws’]),
’swc’: (station_data[’swc’][’atm’], station_data[’swc’][’aws’]),
’tas_u’: (station_data[’tas_u’][’atm’], station_data[’tas_u’][’aws’]),
’thu_l’: (station_data[’thu_l’][’atm’], station_data[’thu_l’][’aws’]),
’thu_u’: (station_data[’thu_u’][’atm’], station_data[’thu_u’][’aws’]),
’thu_u2’: (station_data[’thu_u2’][’atm’], station_data[’thu_u2’][’aws’]),
’tun’: (station_data[’tun’][’atm’], station_data[’tun’][’aws’]),
’upe_u’: (station_data[’upe_u’][’atm’], station_data[’upe_u’][’aws’]),
}
extracted_data = pd.DataFrame()
results = {}
for name, (atm_data, dataset) in all_elev_datasets.items(): # Process all
results[name] = process_dataset(atm_data, dataset, day_offsets, variables)
extracted_data = pd.concat(results.values(), ignore_index=True)
extracted_data = extracted_data.dropna(axis=1, how=’all’)
extracted_data = extracted_data.dropna()
extracted_data = extracted_data.drop(columns=’Date’)
total = extracted_data
above = extracted_data[extracted_data[’Elevation [m]’] >= 1500]
below = extracted_data[extracted_data[’Elevation [m]’] < 1500]
def plot_spearman_correlations(extracted_data, target_variable, label):
spearman_corr = extracted_data.corr(method=’spearman’)[target_variable].
drop(target_variable).sort_values(ascending=False)
low_corr_vars = spearman_corr[abs(spearman_corr) < 0.3]
print(low_corr_vars.index.tolist())
plt.figure(figsize=(20, 10))
sns.barplot(x=spearman_corr.index, y=spearman_corr.values, palette=
’coolwarm’)
plt.title(f’Spearman Correlation Coefficients with Surface Roughness
{label}’, fontsize=20)
plt.xlabel(’Variables’, fontsize=20)
plt.ylabel(’Spearman Correlation Coefficient’, fontsize=20)
plt.xticks(rotation=90)
plt.axhline(-0.3, color=’black’, linestyle=’--’)
plt.axhline(0.3, color=’black’, linestyle=’--’)
plt.tight_layout()
plt.show()
plot_spearman_correlations(total, ’Surface Roughness [cm]’, ’(All
Elevations)’)
plot_spearman_correlations(above, ’Surface Roughness [cm]’, ’(> 1500 m)’)
plot_spearman_correlations(below, ’Surface Roughness [cm]’, ’(< 1500 m)’)
def perform_rf_analysis(data, label, threshold="mean"): # RFR
feature_columns = [col for col in data.columns if col not in [’Surface
Roughness [cm]’, ’Date’]]
X = data[feature_columns]
y = data[’Surface Roughness [cm]’]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42) # Initial RF
rf.fit(X_train, y_train)
selector = SelectFromModel(rf, threshold=threshold, prefit=True)
X_selected_train = selector.transform(X_train)
X_selected_test = selector.transform(X_test)
selected_features = np.array(feature_columns)[selector.get_support()]
rf_selected = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selected.fit(X_selected_train, y_train) # Retrain
param_grid = {
’n_estimators’: [100, 200, 300], ’max_depth’: [None, 10, 20, 30],
’min_samples_split’: [2, 5, 10], ’min_samples_leaf’: [1, 2, 4],
’bootstrap’: [True, False]
}
grid_search = GridSearchCV(estimator=rf_selected, param_grid=param_grid,
cv=5, n_jobs=-1, verbose=2, scoring=’neg_mean_squared_error’)
grid_search.fit(X_selected_train, y_train) # Hypertune
best_rf = grid_search.best_estimator_
best_rf.fit(X_selected_train, y_train) # Cross-validate
y_pred = best_rf.predict(X_selected_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f’{label} - MAE: {mae}, RMSE: {rmse}, R2: {r2}’)
feature_importances = best_rf.feature_importances_
importance_df = pd.DataFrame({’Feature’: selected_features, ’Importance’:
feature_importances})
importance_df = importance_df.sort_values(by=’Importance’,
ascending=False)
plt.figure(figsize=(10, 5))
plt.bar(importance_df[’Feature’], importance_df[’Importance’],
color=’skyblue’)
plt.xlabel(’Features’, fontsize=12)
plt.ylabel(’Importance’, fontsize=12)
plt.title(f’Feature Importance in Predicting Surface Roughness {label}’,
fontsize=12)
plt.xticks(rotation=90)
plt.show()
lin_reg = LinearRegression() # Model fit
y_test_reshaped = np.array(y_test).reshape(-1, 1)
y_pred_reshaped = np.array(y_pred).reshape(-1, 1)
lin_reg.fit(y_test_reshaped, y_pred_reshaped)
line = lin_reg.predict(y_test_reshaped)
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, c=’blue’, marker=’o’)
plt.plot(y_test, line, color=’red’, lw=2)
plt.xlabel(’Actual Surface Roughness [cm]’, fontsize=10)
plt.ylabel(’Predicted Surface Roughness [cm]’, fontsize=10)
plt.title(f’Random Forest Predicted vs Actual {label}’, fontsize=12)
plt.text(0.05, 0.95, f’R²: {r2:.2f}’, transform=plt.gca().transAxes,
fontsize=12, verticalalignment=’top’, horizontalalignment=’left’)
plt.tight_layout()
plt.show()
residuals = y_test - y_pred
plt.figure() # Fit residuals
sns.histplot(residuals, kde=True)
plt.xlabel(’Residuals’, fontsize=12)
plt.ylabel(’Frequency’, fontsize=12)
plt.title(f’Random Forest Residuals Distribution {label}’, fontsize=12)
plt.show()
perform_rf_analysis(total, "(All Elevations)")
perform_rf_analysis(above, "(> 1500 m)")
perform_rf_analysis(below, "(< 1500 m)")