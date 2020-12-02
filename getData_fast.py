import math
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def getValidRows(df, cols):
    df_p = df.copy()
    for c in cols:
        df_p = df_p[(~df_p[c].isna()) & (df_p[c].astype('str') != 'No Data')]
    return df_p


df_oa = pd.read_csv('df_oa.csv')
df_sa = pd.read_csv('df_sa.csv')
df_zone = pd.read_csv('df_zone.csv')
df_psych = pd.read_csv('df_psych.csv', index_col='Datetime')
df_DAT = pd.read_csv('df_DAT.csv')
df_sp = pd.read_csv('df_sp.csv')
pc = pd.read_csv('pc.csv')
tf_results = pd.read_csv('tf_results.csv')

df_cur = df_psych.iloc[-1]

raw = df_psych[["Date", "Time", "SAT", "CFM", "SP", "OAT_1", "OAT_2", "OAWB", "CCoilT", "KW"]]
ahu_col = ["Date", "Time", "SAT", "CFM", "SP", "OAT_RF", "OAT_BLRS", "OAWB", "CCT", "KW"]
raw.columns = ahu_col
raw = getValidRows(raw, ahu_col)
raw = raw.astype({'SAT': 'float', 'CFM': 'float', 'SP': 'float', 'OAT_RF': 'float', 'OAT_BLRS': 'float', 'OAWB': 'float', 'CCT': 'float', 'KW': 'float'})
raw.reset_index(inplace=True)


fig_kw = go.Figure()
fig_kw.add_scatter(x=tf_results['Datetime'], y=tf_results['Predictions'], name="Predictions", line_color='#ef553b')
fig_kw.add_scatter(x=raw['Datetime'], y=raw['KW'], name="Real-Time", line_color='#636efa')

fig_kw.update_layout(title='Fan Load Trend Data', title_x=0.5, legend=dict(x=0.73, y=1, font=dict(size=11), bgcolor='rgba(255,255,255,0.5)'),
                     xaxis_title="Date", yaxis_title="Power (kW)", margin=dict(l=20, r=20, t=40, b=20))

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

df_load = pd.DataFrame(columns=['Component', 'Load', 'Hourly Cost'])
df_load.loc[0] = ['Fan', '%.2f kW' % float(df_cur['KW']), '$%.2f' % df_cur['Fan_cost']]
# df_load.loc[1] = ['Cooling Coil', '%.2f kBtu/h (%.2f kW)' % tuple(df_cur[['CC_load', 'CC_kw']]),
#                   '$%.2f' % df_cur['CC_cost']]
# df_load.loc[2] = ['Zone Reheat', '%.2f kBtu/h (%.2f kW)' % tuple(df_cur[['Zone_RH', 'RH_kw']]),
#                   '$%.2f' % df_cur['RH_cost']]
# df_load.loc[3] = ['Total', '%.2f kW' % (
#             float(df_cur['KW']) + df_cur['CC_kw'] + df_cur['RH_kw']),
#                   '$%.2f' % (df_cur['Fan_cost'] + df_cur['CC_cost'] + df_cur[
#                       'RH_cost'])]
df_load.loc[1] = ['Cooling Coil', '%.2f tons' % max(0, df_cur['CC_load']), '']
df_load.loc[2] = ['Zone Reheat', '%.2f kBtu/h' % df_cur['Zone_RH'], '$%.2f' % df_cur['RH_cost']]
df_load.loc[3] = ['Total', '', '$%.2f' % (df_cur['Fan_cost'] + df_cur['RH_cost'])]


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

df_w = pd.DataFrame(columns=['Datapoint', 'Setpoint Error'])
df_w['Datapoint'] = ['Supply Air Temp', 'Downstream Static Pressure']
df_w['Setpoint Error'] = ['%.2f%%' % (abs(float(df_cur['SAT']) - float(df_cur['SATSP']))/float(df_cur['SATSP'])*100),
                          '%.2f%%' % (abs(float(df_cur['SP_D']) - float(df_cur['SPSP_D']))/float(df_cur['SPSP_D'])*100)]
