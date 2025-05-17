import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import h5py
from PyQt5 import QtGui, QtCore, QtWidgets
import time
import argparse




parser = argparse.ArgumentParser(description="Process date/time strings and an integer.")
    
parser.add_argument('-p', '--passn', type=int, required=True,
                    help="An integer parameter.")

args = parser.parse_args()


warn=False

def in_outbound(df_input):
    df_copy = df_input.copy()
    df_copy[('X_diff [km]')] = df_copy.loc[:,'GSE_X [km]'].diff().interpolate(limit_direction='both')
    point = dir_change_point(df_copy)
    return df_copy[:point[0]] , df_copy[point[1]:]
def dir_change_point(df):
    # gets the point where thevelocity in the gse X direction changes, 
    # showing the change from inbound to outbound
    sign_list = np.sign(df['X_diff [km]'])
    print(sign_list)
    for i in range(len(sign_list)-1):
        if sign_list.iloc[i] * sign_list.iloc[i+1] == -1:
            point = (sign_list.index[i], sign_list.index[i+1])
            break
    return point

def get_full_series(pass_series_input, full_df):
    full_series_output = full_df.loc[pass_series_input.index[0]:pass_series_input.index[-1]]
    return full_series_output

def closest_date(date, series):
    #takes a pd.Timestamp and finds the closest time in the given series
    closest_index = series.index[np.argmin(np.abs(series.index - date))]
    return closest_index
def closest_date_index(date, indexes):
    #takes a pd.Timestamp and finds the closest time in the given series
    closest_index = indexes[np.argmin(np.abs(indexes - date))]
    return closest_index
def get_keys(inp):
    #inp = filepath as string
    f = h5py.File(inp, 'r')
    print(list(f.keys()))
    del f
def get_xy(time, df):
    return (df.loc[time]["GSE_X [km]"],df.loc[time]["GSE_Y [km]"])
def closest_date(date, series):
    #takes a pd.Timestamp and finds the closest time in the given series
    closest_index = series.index[np.argmin(np.abs(series.index - date))]
    return closest_index

df_passes = pd.read_hdf('../datasets/BS_pass.h5', key = 'pass')
f = h5py.File('../datasets/BS_pass.h5', 'r')
print(f.keys())
f.close()

#merged_df = pd.read_hdf('../datasets/df_merged_for_bs.h5', key = '1995')

merged_df = pd.read_hdf('../datasets/gse_position_1997.h5')

f = h5py.File('../datasets/QTN_merged_1997.h5')

    

file_data_95 = '../datasets/QTN_merged_1997.h5'
file_tnr_95 = "../datasets/TNR_results_1997.h5"
tnr_95 = pd.read_hdf(file_tnr_95, key = 'results')






pass_num = args.passn
pass_dat = df_passes.query(f"`{'pass'}` == {pass_num}")

print(pass_dat.index[0].month)

print(merged_df[pass_dat.index[0]:pass_dat.index[-1]])

try:
    inbound, outbound = in_outbound(merged_df[pass_dat.index[0]:pass_dat.index[-1]])
except:
    if int(pass_dat.index[0].year) < 1997:
        print("adding previous year..")
        del merged_df
        m_df1 = pd.read_hdf('../datasets/gse_position_1996.h5')
        m_df2 = pd.read_hdf('../datasets/gse_position_1997.h5')
        merged_df = pd.concat([m_df1, m_df2]).sort_index()
        print("done!")
        inbound, outbound = in_outbound(merged_df[pass_dat.index[0]:pass_dat.index[-1]])
    else:
        exit()
    

mfi = pd.read_hdf('../datasets/MFI_GSE_merged_one_sec.h5', key='1s')
start_date = datetime.datetime(inbound.index[0].year, 1, 1)
end_date = datetime.datetime(outbound.index[-1].year, 12, 31, 23, 59, 59)

filtered_df = mfi[(mfi.index >= start_date) & (mfi.index <= end_date)]
columns_of_interest = ["B_X [nT]", "B_Y [nT]", "B_Z [nT]"]
mfi_selection = filtered_df[columns_of_interest]

filtered_df.drop(filtered_df.columns.difference(columns_of_interest), axis=1, inplace=True)

del mfi

#timestamp = pd.Timestamp("12-20-1995 17:10:40")

_key = f"{int(pass_dat.index[0].month):02}"
if 'all' in str(f.keys()):
    _key = 'all'
if inbound.index[0].month == outbound.index[-1].month:
    data_95_11 = pd.read_hdf(file_data_95, key = _key).sort_index()
else:
    data_95_11_1 = pd.read_hdf(file_data_95, key = _key).sort_index()
    data_95_11_2 = pd.read_hdf(file_data_95, key = _key).sort_index()
    data_95_11 = pd.concat([data_95_11_1, data_95_11_2]).sort_index()
    del data_95_11_2, data_95_11_1

data_95_11.columns = ['V2' if 'V2' in col else col for col in data_95_11.columns]


#---------------------------------------------------------------------
# REQUIRED:
# DataFrames and variables must be defined beforehand:
# - data_95_11: must contain 'V2' and 'v_sw [m/s]' data.
# - merged_df: must have a DateTime index and columns 'B_X [nT]', 'B_Y [nT]', 'B_Z [nT]',
#              'GSE_X [km]', and 'GSE_Y [km]'.
# - tnr_95: must have a DateTime index, a 'QF' column, and a 'fit n_e' column.
# - inbound, outbound: trajectory data with columns 'GSE_X [km]' and 'GSE_Y [km]'.
#---------------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")
_start = time.time()
mpl.use('QtAgg')

# --------------- animation writer ---------------
writer = animation.FFMpegWriter(
    fps=15,
    metadata=dict(artist='Me'),
    bitrate=1000,
    extra_args=[
        '-vcodec', 'h264_nvenc',
        '-preset', 'fast',
        '-b:v', '1M'
    ]
)

# --------------- Prepare Data ---------------
dframe = data_95_11['V2'].loc[closest_date_index(inbound.index[0], data_95_11.index):closest_date_index(inbound.index[-1], data_95_11.index)]
tnr_95.fillna(method='ffill',inplace=True)
print(len(dframe))
freq   = np.logspace(np.log10(4e3), np.log10(256e3), 96)
dframe.columns = [f'V2_{i}' for i in range(len(dframe.columns))]

tnr_max = tnr_95.max
tnr_min = tnr_95.min

tnr_good = tnr_95[tnr_95['QF'] > 2]

print(f"NaNs in dframe: {dframe.isnull().sum().sum()}")
print(f"Freq bins: {len(freq)}, cols: {dframe.shape[1]}")

total_frames = len(dframe)

# --------------- Precompute nearest-timestamp mapping ---------------
merged_df_unique = merged_df[~merged_df.index.duplicated(keep='first')]
nearest_pos = merged_df_unique.index.get_indexer(dframe.index, method='nearest')
closest_mapping = dict(zip(dframe.index, merged_df_unique.index[nearest_pos]))

# --------------- Figure & Subplots Layout ---------------
fig = plt.figure(figsize=(12, 9), dpi=80)
gs  = fig.add_gridspec(3, 2, height_ratios=[1,1,1.2])

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax2 = fig.add_subplot(gs[2,0])

# --------------- ax0: Frequency Plot ---------------
line1, = ax0.plot([], [], 'k-')
ax0.set_xscale('log'); ax0.set_yscale('log')
ax0.set_xlim(4e3,256e3); ax0.set_ylim(1e-17,1e-8)
ax0.grid(True)
ax0.set_xlabel("Frequency (Hz)"); ax0.set_ylabel("Power (dB)")
title = ax0.set_title("")
plt.setp(ax0.xaxis.get_majorticklabels(), rotation=30, ha="right")

# --------------- ax1: Magnetic Field Plot ---------------
line_bx, = ax1.plot([], [], label='B_X [nT]')
line_by, = ax1.plot([], [], label='B_Y [nT]')
line_bz, = ax1.plot([], [], label='B_Z [nT]')
vline     = ax1.axvline(color='k', ls='--')
ax1.set_xlabel("Time"); ax1.set_ylabel("Magnetic Field (nT)")
ax1.legend(loc='upper left')
ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
margin_pct = 0.40

# --------------- ax3: Solar Wind Speed ---------------
line_vsw, = ax3.plot([], [], 'b-')
ax3.set_title("Solar Wind Speed"); ax3.set_ylabel("v_sw [m/s]")
ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
v_std = data_95_11["v_sw [m/s]"].std()
ax3.set_yscale('linear')

# --------------- ax4: fit n_e Scatter ---------------
scatter_ne = ax4.scatter([], [], c='g', s=25)
ax4.set_title("Fit n_e (QF>2)"); ax4.set_ylabel("fit n_e")
ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")

# --------------- ax2: Trajectory Plot ---------------
x_vals = np.arange(500)-250
dist, factor = 30,0.015
ax2.plot(x_vals, ((-x_vals+dist)/factor)**0.5, 'k--')
ax2.plot(x_vals, -(((-x_vals+dist)/factor)**0.5), 'k--')
dist_in, factor_in = 5,0.12
ax2.plot(x_vals, ((-x_vals+dist_in)/factor_in)**0.5, 'k--')
ax2.plot(x_vals, -(((-x_vals+dist_in)/factor_in)**0.5), 'k--')

ax2.plot(inbound['GSE_X [km]']/6371., inbound['GSE_Y [km]']/6371., 'k-')
ax2.plot(outbound['GSE_X [km]']/6371., outbound['GSE_Y [km]']/6371., 'k-')
ax2.plot(0,0,'o',ms=4)
ax2.set_xlim(-45,45); ax2.set_ylim(-45,45)
ax2.set_xlabel("GSE X (Earth Radii)"); ax2.set_ylabel("GSE Y (Earth Radii)")
ax2.legend(['Trajectory'], loc='upper right')
ax2.grid(True)
animated_point = ax2.scatter([], [], c='r', s=50)

# --------------- Animation init & update ---------------
def init():
    line1.set_data([],[])
    line_bx.set_data([],[]); line_by.set_data([],[]); line_bz.set_data([],[])
    vline.set_xdata([0,0]); title.set_text("")
    line_vsw.set_data([],[])
    animated_point.set_offsets(np.empty((0,2)))
    scatter_ne.set_offsets(np.empty((0,2)))
    return [line1,line_bx,line_by,line_bz,vline,title,
            line_vsw,scatter_ne,animated_point]

def update(frame):
    if frame % 2 == 0:
        print(f"\rProgress: {(((float)(frame)/total_frames)*100):.2f}%   ", end='', flush=True)
    t = dframe.index[frame]

    # ax0
    power = 10**(dframe.iloc[frame]/10.)
    line1.set_data(freq, power)
    title.set_text(f"Time: {t}")

    # ax1
    t_min = t - pd.Timedelta(minutes=10)
    t_max = t + pd.Timedelta(minutes=10)
    ## change to here
    df_slice = mfi_selection.loc[t_min:t_max]
    line_bx.set_data(df_slice.index, df_slice["B_X [nT]"])
    line_by.set_data(df_slice.index, df_slice["B_Y [nT]"])
    line_bz.set_data(df_slice.index, df_slice["B_Z [nT]"])
    combined_min = df_slice[["B_X [nT]","B_Y [nT]","B_Z [nT]"]].min().min()
    combined_max = df_slice[["B_X [nT]","B_Y [nT]","B_Z [nT]"]].max().max()
    y_margin = (combined_max - combined_min) * margin_pct
    if y_margin == 0:
        y_margin = abs(combined_max) * margin_pct if combined_max != 0 else 0.1
    try:
        ax1.set_ylim(combined_min - y_margin, combined_max + y_margin)
    except:
        print("ax1 y lim fail. continue ", end='')
        print(f"low {combined_min - y_margin}, high {combined_max+y_margin}")
    vline.set_xdata([t,t])
    ax1.set_xlim(t_min,t_max)

    # ax3
    vsw = data_95_11["v_sw [m/s]"].loc[t_min:t_max]
    line_vsw.set_data(vsw.index, vsw.values)
    #ax3.set_ylim(vsw.mean() - v_std,vsw.mean() + v_std)
    ax3.set_xlim(t_min,t_max)
    ax3.relim()
    ax3.autoscale_view()
    #if not vsw.empty:
    #    ax3.set_ylim(vsw.min(), vsw.max())

    # ax4
    ne = tnr_good.loc[(tnr_good.index >= t_min) & (tnr_good.index <= t_max), 'fit n_e']
    if ne.empty:
        scatter_ne.set_offsets(np.empty((0, 2)))
    else:
        # Convert x datetime -> float
        xdata = mdates.date2num(ne.index.to_pydatetime())
        ydata = ne.values

        # Update scatter
        scatter_ne.set_offsets(np.c_[xdata, ydata])


    ax4.set_xlim(mdates.date2num(t_min.to_pydatetime()), mdates.date2num(t_max.to_pydatetime()))
    y_margin = (ne.max() - ne.min()) * 0.1 if not ne.empty else 1
    try:
        ax4.set_ylim(ne.min()-y_margin, ne.max()+y_margin) if not ne.empty else ax4.set_ylim(0,1)
    except:
        print("set y failed ax4, continuing")
  #  if not ne.empty:
  #      try:
  #          ax4.relim()
  #          ax4.autoscale_view()
 #       except:
#            pass

    # ax2
    ct = closest_mapping[t]
    x_pt = merged_df_unique.loc[ct,'GSE_X [km]']/6371.
    y_pt = merged_df_unique.loc[ct,'GSE_Y [km]']/6371.
    animated_point.set_offsets([[x_pt, y_pt]])

    return [line1,line_bx,line_by,line_bz,vline,title,
            line_vsw,scatter_ne,animated_point]

# --------------- Run & Save ---------------
ani = animation.FuncAnimation(
    fig, update, frames=len(dframe),
    init_func=init, interval=100, blit=True
)

_end = time.time()
print(f"Preprocessing time: {_end-_start:.1f}s")
iteration = 0
if not os.path.exists(f"pass{pass_num}"):
    os.makedirs(f"pass{pass_num}") 
filename = f"pass{pass_num}/animated_plot{iteration}.mp4"
while os.path.exists(filename):
    iteration += 1
    filename = f"pass{pass_num}/animated_plot{iteration}.mp4"
_start = time.time()

ani.save(filename, writer=writer)
_end = time.time()
print(f"Animation render time: {_end-_start:.1f}s")

plt.close('all')






















warnings.filterwarnings("ignore")
_start = time.time()
mpl.use('QtAgg')

# --------------- animation writer ---------------
writer = animation.FFMpegWriter(
    fps=15,
    metadata=dict(artist='Me'),
    bitrate=1000,
    extra_args=[
        '-vcodec', 'h264_nvenc',
        '-preset', 'fast',
        '-b:v', '1M'
    ]
)

# --------------- Prepare Data ---------------
dframe = data_95_11['V2'].loc[closest_date_index(outbound.index[0], data_95_11.index):closest_date_index(outbound.index[-1], data_95_11.index)]
tnr_95.fillna(method='ffill',inplace=True)
print(len(dframe))
freq   = np.logspace(np.log10(4e3), np.log10(256e3), 96)
dframe.columns = [f'V2_{i}' for i in range(len(dframe.columns))]

tnr_max = tnr_95.max
tnr_min = tnr_95.min

tnr_good = tnr_95[tnr_95['QF'] > 2]

print(f"NaNs in dframe: {dframe.isnull().sum().sum()}")
print(f"Freq bins: {len(freq)}, cols: {dframe.shape[1]}")

total_frames = len(dframe)

# --------------- Precompute nearest-timestamp mapping ---------------
merged_df_unique = merged_df[~merged_df.index.duplicated(keep='first')]
nearest_pos = merged_df_unique.index.get_indexer(dframe.index, method='nearest')
closest_mapping = dict(zip(dframe.index, merged_df_unique.index[nearest_pos]))

# --------------- Figure & Subplots Layout ---------------
fig = plt.figure(figsize=(12, 9), dpi=80)
gs  = fig.add_gridspec(3, 2, height_ratios=[1,1,1.2])

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])
ax2 = fig.add_subplot(gs[2,0])

# --------------- ax0: Frequency Plot ---------------
line1, = ax0.plot([], [], 'k-')
ax0.set_xscale('log'); ax0.set_yscale('log')
ax0.set_xlim(4e3,256e3); ax0.set_ylim(1e-17,1e-8)
ax0.grid(True)
ax0.set_xlabel("Frequency (Hz)"); ax0.set_ylabel("Power (dB)")
title = ax0.set_title("")
plt.setp(ax0.xaxis.get_majorticklabels(), rotation=30, ha="right")

# --------------- ax1: Magnetic Field Plot ---------------
line_bx, = ax1.plot([], [], label='B_X [nT]')
line_by, = ax1.plot([], [], label='B_Y [nT]')
line_bz, = ax1.plot([], [], label='B_Z [nT]')
vline     = ax1.axvline(color='k', ls='--')
ax1.set_xlabel("Time"); ax1.set_ylabel("Magnetic Field (nT)")
ax1.legend(loc='upper left')
ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
margin_pct = 0.40

# --------------- ax3: Solar Wind Speed ---------------
line_vsw, = ax3.plot([], [], 'b-')
ax3.set_title("Solar Wind Speed"); ax3.set_ylabel("v_sw [m/s]")
ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
v_std = data_95_11["v_sw [m/s]"].std()
ax3.set_yscale('linear')

# --------------- ax4: fit n_e Scatter ---------------
scatter_ne = ax4.scatter([], [], c='g', s=25)
ax4.set_title("Fit n_e (QF>2)"); ax4.set_ylabel("fit n_e")
ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")

# --------------- ax2: Trajectory Plot ---------------
x_vals = np.arange(500)-250
dist, factor = 30,0.015
ax2.plot(x_vals, ((-x_vals+dist)/factor)**0.5, 'k--')
ax2.plot(x_vals, -(((-x_vals+dist)/factor)**0.5), 'k--')
dist_in, factor_in = 5,0.12
ax2.plot(x_vals, ((-x_vals+dist_in)/factor_in)**0.5, 'k--')
ax2.plot(x_vals, -(((-x_vals+dist_in)/factor_in)**0.5), 'k--')

ax2.plot(inbound['GSE_X [km]']/6371., inbound['GSE_Y [km]']/6371., 'k-')
ax2.plot(outbound['GSE_X [km]']/6371., outbound['GSE_Y [km]']/6371., 'k-')
ax2.plot(0,0,'o',ms=4)
ax2.set_xlim(-45,45); ax2.set_ylim(-45,45)
ax2.set_xlabel("GSE X (Earth Radii)"); ax2.set_ylabel("GSE Y (Earth Radii)")
ax2.legend(['Trajectory'], loc='upper right')
ax2.grid(True)
animated_point = ax2.scatter([], [], c='r', s=50)

# --------------- Animation init & update ---------------
def init():
    line1.set_data([],[])
    line_bx.set_data([],[]); line_by.set_data([],[]); line_bz.set_data([],[])
    vline.set_xdata([0,0]); title.set_text("")
    line_vsw.set_data([],[])
    animated_point.set_offsets(np.empty((0,2)))
    scatter_ne.set_offsets(np.empty((0,2)))
    return [line1,line_bx,line_by,line_bz,vline,title,
            line_vsw,scatter_ne,animated_point]

def update(frame):
    if frame % 2 == 0:
        print(f"\rProgress: {(((float)(frame)/total_frames)*100):.2f}%", end='', flush=True)
    t = dframe.index[frame]

    # ax0
    power = 10**(dframe.iloc[frame]/10.)
    line1.set_data(freq, power)
    title.set_text(f"Time: {t}")

    # ax1
    t_min = t - pd.Timedelta(minutes=10)
    t_max = t + pd.Timedelta(minutes=10)
    df_slice = mfi_selection.loc[t_min:t_max]
    line_bx.set_data(df_slice.index, df_slice["B_X [nT]"])
    line_by.set_data(df_slice.index, df_slice["B_Y [nT]"])
    line_bz.set_data(df_slice.index, df_slice["B_Z [nT]"])
    combined_min = df_slice[["B_X [nT]","B_Y [nT]","B_Z [nT]"]].min().min()
    combined_max = df_slice[["B_X [nT]","B_Y [nT]","B_Z [nT]"]].max().max()
    y_margin = (combined_max - combined_min) * margin_pct
    if y_margin == 0:
        y_margin = abs(combined_max) * margin_pct if combined_max != 0 else 0.1
    try:
        ax1.set_ylim(combined_min - y_margin, combined_max + y_margin)
    except:
        print("ax1 limits failed, continuing..")
    vline.set_xdata([t,t])
    ax1.set_xlim(t_min,t_max)

    # ax3
    vsw = data_95_11["v_sw [m/s]"].loc[t_min:t_max]
    line_vsw.set_data(vsw.index, vsw.values)
    #ax3.set_ylim(vsw.mean() - v_std,vsw.mean() + v_std)
    ax3.set_xlim(t_min,t_max)
    ax3.relim()
    ax3.autoscale_view()
    #if not vsw.empty:
    #    ax3.set_ylim(vsw.min(), vsw.max())

    # ax4
    ne = tnr_good.loc[(tnr_good.index >= t_min) & (tnr_good.index <= t_max), 'fit n_e']
    if ne.empty:
        scatter_ne.set_offsets(np.empty((0, 2)))
    else:
        # Convert x datetime -> float
        xdata = mdates.date2num(ne.index.to_pydatetime())
        ydata = ne.values

        # Update scatter
        scatter_ne.set_offsets(np.c_[xdata, ydata])


    ax4.set_xlim(mdates.date2num(t_min.to_pydatetime()), mdates.date2num(t_max.to_pydatetime()))
    y_margin = (ne.max() - ne.min()) * 0.1 if not ne.empty else 1
    try:
        ax4.set_ylim(ne.min()-y_margin, ne.max()+y_margin) if not ne.empty else ax4.set_ylim(0,1)
    except:
        print("set y failed ax4, continuing")
    
    
    
    
    
    
    
  #  if not ne.empty:
  #      try:
  #          ax4.relim()
  #          ax4.autoscale_view()
 #       except:
#            pass

    # ax2
    ct = closest_mapping[t]
    x_pt = merged_df_unique.loc[ct,'GSE_X [km]']/6371.
    y_pt = merged_df_unique.loc[ct,'GSE_Y [km]']/6371.
    animated_point.set_offsets([[x_pt, y_pt]])

    return [line1,line_bx,line_by,line_bz,vline,title,
            line_vsw,scatter_ne,animated_point]

# --------------- Run & Save ---------------
ani = animation.FuncAnimation(
    fig, update, frames=len(dframe),
    init_func=init, interval=100, blit=True
)

_end = time.time()
print(f"Preprocessing time: {_end-_start:.1f}s")
iteration = 0
filename = f"pass{pass_num}/animated_plot{iteration}.mp4"
while os.path.exists(filename):
    iteration += 1
    filename = f"pass{pass_num}/animated_plot{iteration}.mp4"
_start = time.time()

ani.save(filename, writer=writer)
_end = time.time()
print(f"Animation render time: {_end-_start:.1f}s")