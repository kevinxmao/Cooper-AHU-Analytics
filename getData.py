import math
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dateutil.relativedelta import relativedelta
import psychrolib as pl
import tensorflow as tf
from tensorflow import keras
from plotly.offline import plot
# set Imperial units
pl.SetUnitSystem(pl.IP)


def getValidRows(df, cols):
    df_p = df.copy()
    for c in cols:
        df_p = df_p[(~df_p[c].isna()) & (df_p[c].astype('str') != 'No Data')]
    return df_p

def getPsychroLoads(OAT, OARH, SAT, SARH, CFM, rh_vlv):
    # enthalpy calcs for outside air and supply air
    # Assume constant at P_atm for now
    hum_ratio = pl.GetHumRatioFromRelHum(OAT, OARH, 14.7)
    OAH = pl.GetMoistAirEnthalpy(OAT, hum_ratio)

    hum_ratio = pl.GetHumRatioFromRelHum(SAT, SARH, 14.7)
    SAH = pl.GetMoistAirEnthalpy(SAT, hum_ratio)

    # cooling coil load in tons
    CC_load = 4.5 * CFM * (OAH - SAH) / 12000
    # zone reheat load calc for all filled in RH valve cells
    zone_reheat_load = (rh_cap * rh_vlv / 100).sum()
    return CC_load, zone_reheat_load


# historical AHU data from April 2019
ahu = pd.read_csv('AHU3_April2019_formatted.csv')
ahu.columns = ['Date', 'Time', 'SAT', 'CFM', 'SP', 'OAT_RF', 'OAT_BLRS', 'OAWB', 'CCT', 'KW']
ahu['Datetime'] = pd.to_datetime(ahu['Date'] + ' ' + ahu['Time'])
ahu = ahu.set_index('Datetime')

# datapoint column names
df_datapoints = pd.read_csv('datapoints.csv')
datapoints = {}
for i in range(len(df_datapoints)):
    datapoints[df_datapoints.loc[i, 'Point']] = df_datapoints.loc[i, 'Column Name']

# 2020 YTD dashboard input data
df_input = pd.read_csv('AHU03trenddataSP.csv', skiprows=152)

df_input.rename(columns=datapoints, inplace=True)
df_input['Datetime'] = pd.to_datetime(df_input['Date'] + ' ' + df_input['Time'])
df_input = df_input.set_index('Datetime')
df_psych = getValidRows(df_input, ['OAT_1', 'OARH', 'SAT', 'SARH', 'CFM', 'KW'])
df_psych = df_psych.astype({'OAT_1':'float', 'OARH':'float', 'SAT':'float', 'OARH':'float', 'SARH':'float', 'OADP':'float', 'CFM':'float', 'KW':'float'})

# 3M vs 12M % change data, currently can only be for month of April
dt_start = pd.to_datetime('4/1/2020 12:00:00')
dt_end = min(pd.to_datetime('2020-05-01 23:55:00'), pd.to_datetime(str(df_psych.index[len(df_psych)-1])[:11] + '12:00:00')) #'4/13/2020 12:00:00'  # most recent date
dt_range = pd.date_range(start=dt_start, end=dt_end, freq='D')
kw_cur = df_input.loc[dt_range, 'KW'].astype('float')
pc = pd.DataFrame(index=kw_cur.index, columns=['3', '12', 'Q'])

dt_range_3 = pd.DatetimeIndex([kw_cur.index[i] + relativedelta(months=-3) for i in range(len(kw_cur))])
dt_range_12 = pd.DatetimeIndex([kw_cur.index[i] + relativedelta(years=-1) for i in range(len(kw_cur))])

kw_cur = kw_cur.to_numpy()
kw_3 = df_input.loc[dt_range_3, 'KW'].astype('float').to_numpy()
kw_12 = ahu.loc[dt_range_12, 'KW'].astype('float').to_numpy()

# percent change
pc_3 = (kw_cur - kw_3) / kw_3 * 100
pc_12 = (kw_cur - kw_12) / kw_12 * 100
pc['3'] = pc_3
pc['12'] = pc_12

# each quadrant corresponds to a different color
pc.loc[(pc['3'] < 0) & (pc['12'] < 0), 'Q'] = 0
pc.loc[(pc['3'] > 0) & (pc['12'] < 0), 'Q'] = 1
pc.loc[(pc['3'] > 0) & (pc['12'] > 0), 'Q'] = 2
pc.loc[(pc['3'] < 0) & (pc['12'] > 0), 'Q'] = 3
colors = ['limegreen', 'yellow', 'red', 'orange']

fig_pc = go.Figure()
for i in range(4):
    # plot data in each quadrant in a separate color
    fig_pc.add_trace(go.Scatter(x=pc.loc[pc['Q'] == i, '3'], y=pc.loc[pc['Q'] == i, '12'], mode='markers',
                                marker=dict(size=15, color=colors[i]),
                                text=pc.loc[pc['Q'] == i].index))

# plot axis lines
fig_pc.add_trace(go.Scatter(x=[-100, 100], y=[0, 0], mode='lines', marker=dict(color='black')))
fig_pc.add_trace(go.Scatter(x=[0, 0], y=[-100, 100], mode='lines', marker=dict(color='black')))

fig_pc.update_layout(
    title="Fan kW: 3 Month vs 12 Month % Change",
    title_x=0.5,
    xaxis_title="3 Month % Change",
    yaxis_title="12 Month % Change",
    xaxis=dict(rangemode='tozero', range=[-100, 100], autorange="reversed"),
    yaxis=dict(rangemode='tozero', range=[-100, 100], autorange="reversed"),
    showlegend=False,
    margin=dict(l=20, r=20, t=40, b=20)
)

############ Psychrometric Calcs ############
input_zone = pd.read_csv('Dashboard Input.csv')

kw_per_ton = 3.51685  # 1 ton of refrigeration = 3.51685 kW
kw_per_MBH = 0.293071039  # 1 kBtu/h = 0.293071039 kW
dollars_per_kwh = 0.196  # $0.196 / kwH
dollars_per_kbtu = 0.01296  # $0.196 / kBtu

oat = df_psych['OAT_1'].astype('float').tolist()
df_psych['OARH'] = df_psych['OARH'] / 100
df_psych.loc[df_psych['OARH'] > 1, 'OARH'] = 1
oarh = df_psych['OARH'].tolist()

sat = df_psych['SAT'].tolist()
df_psych['SARH']=df_psych['SARH'] / 100
df_psych.loc[df_psych['SARH'] > 1, 'SARH'] = 1
sarh = df_psych['SARH'].tolist()

cfm = df_psych['CFM'].astype('float').tolist()
kw = df_psych['KW'].astype('float').tolist()
cc_load, zone_rh = [0] * len(oat), [0] * len(oat)
rh_cap = input_zone.loc[~input_zone['RH_capacity_MBH'].isna(), 'RH_capacity_MBH'].to_numpy()
col_104 = df_psych.columns.get_loc('RH_104')
col_LL111 = df_psych.columns.get_loc('RH_LL111')

for i in range(len(oat)):
    rh_vlv = df_psych.iloc[i, col_104:col_LL111 + 1].astype('float').to_numpy()
    cc_load[i], zone_rh[i] = getPsychroLoads(oat[i], oarh[i], sat[i], sarh[i], cfm[i], rh_vlv)

cc_kw = np.array(cc_load) * kw_per_ton
rh_kw = np.array(zone_rh) * kw_per_MBH

df_psych = df_psych.assign(CC_load=cc_load, CC_kw=cc_kw, Zone_RH=zone_rh, RH_kw=rh_kw)
df_psych['Fan_cost'] = df_psych['KW'] * dollars_per_kwh
df_psych = df_psych.assign(CC_load=cc_load, Zone_RH=zone_rh)
df_psych['CC_kw'] = df_psych['CC_load'] * kw_per_ton
df_psych['RH_kw'] = df_psych['Zone_RH'] * kw_per_MBH
df_psych['KW_total'] = df_psych['KW'] + df_psych['CC_kw'] + df_psych['RH_kw']
df_psych['CC_cost'] = df_psych['CC_load'] * dollars_per_kwh
df_psych['RH_cost'] = df_psych['Zone_RH'] * dollars_per_kbtu
df_psych['Total_cost'] = df_psych['Fan_cost'] + df_psych['CC_cost'] + df_psych['RH_cost']

df_cur = df_psych.iloc[-1]
for i, d in enumerate(df_cur[2:]):
    if i > 1:
        df_cur[i] = float(df_cur[i])

# reset tables
dp = df_psych.loc[df_psych.index[len(df_psych)-1] + relativedelta(days=-1):df_psych.index[len(df_psych)-1], 'OADP']
df_DAT = pd.DataFrame(columns=['Mode', 'Value'])
df_DAT['Mode'] = ['Mode', 'STPT Limit', 'Reset last 24h']
if float(df_cur['OADP']) > 58:
    mode = 'Wet'
    if df_cur['OAT_1'] > 85:
        lim = 'High (52 F)'
    elif df_cur['OAT_1'] < 50:
        lim = 'Low (55 F)'
    else:
        lim = 'Middle (52-55 F)'
    reset = 'Yes' if (dp < 58).any() else 'No'
else:
    mode = 'Dry'
    if float(df_cur['OAT_1']) > 85:
        lim = 'High (56 F)'
    elif df_cur['OAT_1'] < 50:
        lim = 'Low (65 F)'
    else:
        lim = 'Middle (56-65 F)'
    reset = 'Yes' if (dp > 58).any() else 'No'
df_DAT['Value'] = [mode, lim, reset]

t = df_psych.index[len(df_psych)-1]
t = df_psych.iloc[len(df_psych)-1].Time
df_sp = pd.DataFrame(columns=['Mode', 'Value'])
df_sp['Mode'] = ['Mode', 'STPT', 'Reset last 24h']
if t >= '07:00:00' and t <= '22:00:00':
    mode = 'Day'
    lim = '2.5 in WC'
else:
    mode = 'Night'
    lim = '1.0 in WC'
df_sp['Value'] = [mode, lim, 'Yes'] # always resets in last 24h if based on time

df_load = pd.DataFrame(columns=['Component', 'Load', 'Hourly Cost'])
df_load.loc[0] = ['Fan', '%.1f kW' % float(df_cur['KW']), '$%.1f' % df_cur['Fan_cost']]
# df_load.loc[1] = ['Cooling Coil', '%.1f kBtu/h (%.1f kW)' % 
tuple(df_cur[['CC_load', 'CC_kw']]),
#                   '$%.1f' % df_cur['CC_cost']]
# df_load.loc[2] = ['Zone Reheat', '%.1f kBtu/h (%.1f kW)' % tuple(df_cur[['Zone_RH', 'RH_kw']]),
#                   '$%.1f' % df_cur['RH_cost']]
# df_load.loc[3] = ['Total', '%.1f kW' % (
#             float(df_cur['KW']) + df_cur['CC_kw'] + df_cur['RH_kw']),
#                   '$%.1f' % (df_cur['Fan_cost'] + df_cur['CC_cost'] + df_cur[
#                       'RH_cost'])]
df_load.loc[1] = ['Cooling Coil', '%.1f tons' % max(0, df_cur['CC_load']), '']
df_load.loc[2] = ['Zone Reheat', '%.1f kBtu/h' % df_cur['Zone_RH'], '$%.1f' % df_cur['RH_cost']]
df_load.loc[3] = ['Total', '', '$%.1f' % (df_cur['Fan_cost'] + df_cur['RH_cost'])]

df_oa = pd.DataFrame(columns=['Weather', 'Value'])
df_oa['Weather'] = ['Temperature', 'Relative Humidity', 'Dew Point']
df_oa['Value'] = ['%s F' % df_cur['OAT_1'], '%.1f%%' % (df_cur['OARH']*100), ('%s F' % df_cur['OADP']) if (df_cur['OADP'] != 'No Data' or df_cur['OADP'] is None) else 'No Data']

df_sa = pd.DataFrame(columns=['Weather', 'Value'])
df_sa['Weather'] = ['Temperature', 'Setpoint', 'Relative Humidity']
df_sa['Value'] = ['%s F' % df_cur['SAT'], '%s F' % df_cur['SATSP'], '%.1f%%' % (df_cur['SARH']*100)]

df_w = pd.DataFrame(columns=['Datapoint', 'Setpoint Error'])
df_w['Datapoint'] = ['Supply Air Temp', 'Downstream Static Pressure']
df_w['Setpoint Error'] = ['%.1f%%' % (abs(float(df_cur['SAT']) - float(df_cur['SATSP']))/float(df_cur['SATSP'])*100),
                          '%.1f%%' % (abs(float(df_cur['SP_D']) - float(df_cur['SPSP_D']))/float(df_cur['SPSP_D'])*100)]

# limits taken from YTD data
df_psych['KW'] = df_psych['KW'].astype('float')
load_vals = {'Fan': dict(name='Fan kW', min=max(0, math.floor(df_psych['KW'].min())), max=math.ceil(df_psych['KW'].max()), cur=df_cur['KW']),
             'Cooling Coil': dict(name='Cooling Coil Load', min=max(0, math.floor(df_psych['CC_load'].min())), max=math.ceil(df_psych['CC_load'].max()), cur=df_cur['CC_load']),
             'Zone Reheat': dict(name='Zone Reheat Load', min=max(0, math.floor(df_psych['Zone_RH'].min())), max=math.ceil(df_psych['Zone_RH'].max()), cur=df_cur['Zone_RH']),
             'Total': dict(name='Total Load', min=max(0, math.floor(df_psych['KW_total'].min())), max=math.ceil(df_psych['KW_total'].max()), cur=df_cur['KW_total'])}

# load colorscales
scales = {}
for load in load_vals:
    scales[load] = go.Figure()
    scales[load].add_trace(go.Heatmap(
        x=np.arange(load_vals[load]['min'], load_vals[load]['max'], .1),
        z=[np.arange(load_vals[load]['min'], load_vals[load]['max'], .1)],
        colorscale='RdBu',
        reversescale=True,
        showscale=False
    ))
    scales[load].add_trace(go.Scatter(x=[max(0, load_vals[load]['cur']), max(0, load_vals[load]['cur'])], y=[-0.45, 0.45], mode='lines', line=dict(color='black', width=8)))
    scales[load].update_layout(title_x=0.5, xaxis=dict(fixedrange=True), yaxis=dict(fixedrange=True, showticklabels=False), margin=dict(l=0, r=0, t=0, b=10), paper_bgcolor='rgba(0,0,0,0)')

df_load.insert(2, 'Colorscale', [scales[l] for l in scales])

zone_cols = np.unique([c.split('_', 1)[1] for c in df_input.columns if any(prefix in c for prefix in ['T_', 'RH_', 'D_', 'CFM']) and 'OAT' not in c and c != 'CFM']).tolist()

df_zone = pd.DataFrame(columns=['Room', 'Temperature (F)', 'Temperature STPT', 'Temp STPT Error', 'Supply (CFM)', 'Reheat Valve %', 'Damper %'])
for i, zone in enumerate(zone_cols):
    df_zone.loc[i] = [zone, df_cur['T_%s' % zone], df_cur['TSP_%s' % zone], '%.1f%%' % (abs(df_cur['T_%s' % zone] - df_cur['TSP_%s' % zone])/df_cur['TSP_%s' % zone]*100),
                      df_cur['CFM_%s' % zone], df_cur['RH_%s' % zone] if zone != 'LL123' else '', df_cur['D_%s' % zone]]


############ Predictive Model ############
raw = df_psych[["Date", "Time", "SAT", "CFM", "SP", "OAT_1", "OAT_2", "OAWB", "CCoilT", "KW"]]
ahu_col = ["Date", "Time", "SAT", "CFM", "SP", "OAT_RF", "OAT_BLRS", "OAWB", "CCT", "KW"]
raw.columns = ahu_col
raw = getValidRows(raw, ahu_col)
raw = raw.astype({'SAT': 'float', 'CFM': 'float', 'SP': 'float', 'OAT_RF': 'float', 'OAT_BLRS': 'float', 'OAWB': 'float', 'CCT': 'float', 'KW': 'float'})
raw.reset_index(inplace=True)

pr_data = raw.copy()
raw.drop(['Date', 'Time'], axis=1, inplace=True)
pr_data.drop('Datetime', axis=1, inplace=True)
# divide CFM column by 1000 to get all features within an order of magnitude
pr_data['CFM'] = raw['CFM']/1000
# update column names consistent withnew CFM column
pr_col = ["Date", "Time", "SAT", "10^3 CFM", "SP", "OAT_RF", "OAT_BLRS", "OAWB", "CCT", "KW"]
pr_data.columns = pr_col

x = pr_data[['SAT', '10^3 CFM', 'SP', 'OAT_RF', 'OAT_BLRS', 'OAWB', 'CCT']]
y = pr_data['KW']
x_test_stats = x.describe().transpose()
tf_predictions = []

def pr_norm(x):
    return (x - x_test_stats['mean'])/x_test_stats['std']

normed_x = pr_norm(x)
model2 = keras.models.load_model("model_April2019.h5")
optimizer = tf.keras.optimizers.Adam(0.008)
model2.compile(loss='mean_squared_error',
                optimizer = optimizer,
                metrics=['mean_absolute_error','mean_squared_error'])
display_predictions = model2.predict(normed_x).flatten()

for val in display_predictions:
    tf_predictions.append(val)

frame = {'Predictions':tf_predictions}
tf_results = pd.DataFrame(frame)
tf_results['Datetime'] = raw['Datetime']
tf_results.sort_index(axis=0)

fig_kw = go.Figure()
fig_kw.add_scatter(x=tf_results['Datetime'], y=tf_results['Predictions'], name="Predictions", line_color='#ef553b')
fig_kw.add_scatter(x=raw['Datetime'], y=raw['KW'], name="Real-Time", line_color='#636efa')

fig_kw.update_layout(title='Fan Load Trend Data', title_x=0.5, legend=dict(x=0.73, y=1, font=dict(size=11), bgcolor='rgba(255,255,255,0.5)'),
                     xaxis_title="Date", yaxis_title="Power (kW)", margin=dict(l=20, r=20, t=40, b=20))

df_oa.to_csv('df_oa.csv', index=False)
df_sa.to_csv('df_sa.csv', index=False)
df_zone.to_csv('df_zone.csv', index=False)
df_psych.to_csv('df_psych.csv')
df_DAT.to_csv('df_DAT.csv', index=False)
df_sp.to_csv('df_sp.csv', index=False)
pc.to_csv('pc.csv')
tf_results.to_csv('tf_results.csv')

############ time series plots ############
cols_oa = ['OAT_1', 'OAT_2', 'OAWB', 'OARH', 'OADP']
cols_ahu = ['SP', 'CCoilT', 'CCoilVLV', 'SP_D', 'SPSP_D', 'PHCoil', 'PHCoilT', 'KW', 'RHCoilVLV', 'CFM', 'SARH', 'SAT', 'SATSP']
cols_load = ['CC_load', 'CC_kw', 'Zone_RH', 'RH_kw', 'KW_total']
cols_T = ['T_LL102', 'T_LL121', 'T_LL123', 'T_LL111', 'T_LL109', 'T_LL113', 'T_LL115',
       'T_LL103', 'T_LL104', 'T_LL101', 'T_LL121A', 'T_LL119', 'T_LL120',
       'T_LL120A', 'T_319', 'T_305', 'T_Atrium', 'T_201A', 'T_113_2',
       'T_113_1', 'T_224', 'T_201', 'T_101', 'T_106', 'T_105', 'T_104',
          'TSP_LL123', 'TSP_LL121', 'TSP_LL102', 'TSP_LL120A',
          'TSP_LL120', 'TSP_LL119', 'TSP_LL121A', 'TSP_LL101', 'TSP_LL104',
          'TSP_LL103', 'TSP_LL115', 'TSP_LL113', 'TSP_LL109', 'TSP_LL111',
          'TSP_319', 'TSP_104', 'TSP_105', 'TSP_106', 'TSP_101', 'TSP_201',
          'TSP_224', 'TSP_113_1', 'TSP_113_2', 'TSP_201A', 'TSP_Atrium',
          'TSP_305']
cols_SUP = ['CFM_LL123', 'CFM_LL121', 'CFM_LL102', 'CFM_LL120A',
       'CFM_LL120', 'CFM_LL119', 'CFM_LL121A', 'CFM_LL101', 'CFM_LL104',
       'CFM_LL103', 'CFM_LL115', 'CFM_LL113', 'CFM_LL109', 'CFM_LL111',
       'CFM_319', 'CFM_104', 'CFM_105', 'CFM_106', 'CFM_101', 'CFM_201',
       'CFM_224', 'CFM_113_1', 'CFM_113_2', 'CFM_201A', 'CFM_Atrium',
       'CFM_305']
cols_RH = ['RH_104', 'RH_105', 'RH_106', 'RH_101', 'RH_201', 'RH_224',
       'RH_113_1', 'RH_113_2', 'RH_201A', 'RH_Atrium', 'RH_305', 'RH_319',
       'RH_LL120A', 'RH_LL120', 'RH_LL119', 'RH_LL121A', 'RH_LL101',
       'RH_LL104', 'RH_LL103', 'RH_LL115', 'RH_LL113', 'RH_LL109', 'RH_LL102',
       'RH_LL121', 'RH_LL111']
cols_D = ['D_104', 'D_105', 'D_106', 'D_101', 'D_201', 'D_224',
       'D_113_1', 'D_113_2', 'D_201A', 'D_Atrium', 'D_305', 'D_LL120A',
       'D_LL120', 'D_LL119', 'D_LL121A', 'D_LL101', 'D_LL104', 'D_LL103',
       'D_LL115', 'D_LL113', 'D_LL109', 'D_LL111', 'D_LL123', 'D_LL121',
       'D_LL102', 'D_319']

fig_oa = go.Figure()
for c in cols_oa:
    fig_oa.add_scattergl(x=df_input.index, y=df_input[c], name=c)
fig_oa.update_layout(title='Outside Air Conditions', title_x=0.5,
                     xaxis_title="Date", margin=dict(l=20, r=20, t=40, b=20),
                     width=560, height=280)

fig_ahu = go.Figure()
for i, c in enumerate(cols_ahu):
    if i < 4:
        fig_ahu.add_scattergl(x=df_input.index, y=df_input[c], name=c)
    else:
        fig_ahu.add_scattergl(x=df_input.index, y=df_input[c], name=c, visible='legendonly')
fig_ahu.update_layout(title='AHU-3 Datapoints', title_x=0.5,
                     xaxis_title="Date", margin=dict(l=20, r=20, t=40, b=20),
                      width=560, height=280)

fig_load = go.Figure()
for c in cols_load:
    fig_load.add_scattergl(x=df_psych.index, y=df_psych[c], name=c)
fig_load.update_layout(title='AHU-3 Loads', title_x=0.5,
                     xaxis_title="Date", margin=dict(l=20, r=20, t=40, b=20),
                       width=560, height=280)

fig_T = go.Figure()
for i, c in enumerate(cols_T):
    if i < 4:
        fig_T.add_scattergl(x=df_input.index, y=df_input[c], name=c)
    else:
        fig_T.add_scattergl(x=df_input.index, y=df_input[c], name=c, visible='legendonly')

fig_T.update_layout(title='Zone Temperatures and Sepoints', title_x=0.5,
                     xaxis_title="Date", yaxis_title='Temperature (F)', margin=dict(l=20, r=20, t=40, b=20),
                    width=560, height=280)

fig_SUP = go.Figure()
for i, c in enumerate(cols_SUP):
    if i < 4:
        fig_SUP.add_scattergl(x=df_input.index, y=df_input[c], name=c)
    else:
        fig_SUP.add_scattergl(x=df_input.index, y=df_input[c], name=c, visible='legendonly')
fig_SUP.update_layout(title='Zone Supply CFM', title_x=0.5,
                     xaxis_title="Date", yaxis_title='Supply Airflow (CFM)', margin=dict(l=20, r=20, t=40, b=20),
                    width=560, height=280)

fig_RH = go.Figure()
for i, c in enumerate(cols_RH):
    if i < 4:
        fig_RH.add_scattergl(x=df_input.index, y=df_input[c], name=c)
    else:
        fig_RH.add_scattergl(x=df_input.index, y=df_input[c], name=c, visible='legendonly')
fig_RH.update_layout(title='Zone Reheat Valve Position', title_x=0.5,
                     xaxis_title="Date", yaxis_title='Valve Position (%)', margin=dict(l=20, r=20, t=40, b=20),
                     width=560, height=280)

fig_D = go.Figure()
for i, c in enumerate(cols_D):
    if i < 4:
        fig_D.add_scattergl(x=df_input.index, y=df_input[c], name=c)
    else:
        fig_D.add_scattergl(x=df_input.index, y=df_input[c], name=c, visible='legendonly')
fig_D.update_layout(title='Zone Damper Position', title_x=0.5,
                     xaxis_title="Date", yaxis_title='Damper Position (%)', margin=dict(l=20, r=20, t=40, b=20),
                    width=560, height=280)

plot(fig_oa, filename = 'fig_oa.html', auto_open=False)
plot(fig_ahu, filename = 'fig_ahu.html', auto_open=False)
plot(fig_load, filename = 'fig_load.html', auto_open=False)
plot(fig_T, filename = 'fig_T.html', auto_open=False)
plot(fig_SUP, filename = 'fig_SUP.html', auto_open=False)
plot(fig_RH, filename = 'fig_RH.html', auto_open=False)
plot(fig_D, filename = 'fig_D.html', auto_open=False)
