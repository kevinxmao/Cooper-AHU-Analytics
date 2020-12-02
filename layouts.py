import dash_core_components as dcc
import dash_html_components as html
from components import navbar, navbar_zone, navbar_ts, oa_table, df_cur, sa_table, warnings_table, DAT_table, sp_table, load_table, \
        fig_pc, fig_kw, zone_table #, fig_oa, fig_ahu, fig_load, fig_T, fig_SUP, fig_RH, fig_D

layout_home = html.Div(children=[
    navbar,
    html.Div([
        html.Div(
            oa_table, className="myblock oa",
        ),
        html.Div(html.Div([
            html.Img(src='/assets/ahu3schematic - Copy.png', className='image'),
            html.Div(children='Outside Air Damper ', className='OADamper'),
            html.Div(children='Filter', className='Filter'),
            html.Div(children='Cooling Coil Temperature:  %s F Valve: %s%%' % (df_cur['CCoilT'], df_cur['CCoilVLV']),
                     className='CCoil'),
            html.Div(children='Heating Coil: %s F' % df_cur['PHCoilT'], className='HCoil'),
            html.Div(children='Supply Fan: %s CFM' % df_cur['CFM'], className='Supply-Fan'),
            html.Div(children='Supply Air Damper', className='SADamper'),
            html.Div(children='Downstream Static Pressure Setpoint: %s in WC Downstream Static Pressure: %s in WC' % (
            df_cur['SPSP_D'], df_cur['SP_D']), className='sp')
        ], className='schematic'), className='schemwrapper'),
        html.Div(
            sa_table, className="myblock sa",
        ),
        ],
            className="container1"
    ),

    html.Div([
        html.Div([html.Div(
            warnings_table, className="myblock",
        ),
        html.Div(
            DAT_table, className="myblock"
        ),
            html.Div(
                sp_table, className="myblock"
            )
        ], className='container3'),

        html.Div(html.Div(
            load_table, className="dattable"
        ), className='container4'),

    ], className='container2'),

    html.Div([
            html.Div(dcc.Graph(
                id='fig_pc',
                figure=fig_pc, className='myblock graph'),
                className="container5"
            ),
            html.Div(dcc.Graph(
                id='kw',
                figure=fig_kw, className='myblock graph'),
                className="container5"
            ),
            ], className="row3"
        )
])

layout_zones = html.Div([
    navbar_zone,
    html.H1(children='All Zones served by AHU-3', style={'textAlign': 'center', 'margin':'0', 'fontSize':'2.25rem', 'fontWeight':'bold'}),
    html.Br(),
    zone_table,
    html.Br(),
])