import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


fast_build = True
# fast_build = False


if fast_build:
    from GetData_fast import fig_kw, df_DAT, fig_pc, plots, df_cur, df_load, df_oa, df_sa, df_w, df_sp, \
        df_zone, scales#, fig_oa, fig_ahu, fig_load, fig_T, fig_SUP, fig_RH, fig_D
else:
    from GetData import fig_kw, df_DAT, fig_pc, plots, df_cur, df_load, df_oa, df_sa, df_w, df_sp, \
        df_zone, scales, fig_oa, fig_ahu, fig_load, fig_T, fig_SUP, fig_RH, fig_D


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(html.Div('Data as of %s' % df_cur.name, className='nav-link', style={'fontSize':'1rem', 'position':'relative', 'right':'9rem'})),
        dbc.NavItem(dbc.NavLink("Zone Datapoints", href="/zones", style={'fontSize':'1.2rem'})),
        dbc.NavItem(dbc.NavLink("Time Series Data", href="/time-series", style={'fontSize':'1.2rem'})),
    ],
    brand="AHU-3 Energy Dashboard",
    brand_href="/",
    color="#323232db",
    dark=True,
)

navbar_zone = dbc.NavbarSimple(
    children=[
        dbc.NavItem(html.Div('Data as of %s' % df_cur.name, className='nav-link', style={'fontSize':'1rem', 'position':'relative', 'right':'9rem'})),
        dbc.NavItem(dbc.NavLink("Zone Datapoints", href="/zones", style={'fontSize':'1.7rem', 'color':'white'})),
        dbc.NavItem(dbc.NavLink("Time Series Data", href="/time-series", style={'fontSize':'1.2rem'})),
    ],
    brand="AHU-3 Energy Dashboard",
    brand_href="/",
    brand_style={'fontSize':'1.2rem', 'color':'rgba(255,255,255,.5)'},
    color="#323232db",
    dark=True,
)

navbar_ts = dbc.NavbarSimple(
    children=[
        dbc.NavItem(html.Div('Data as of %s' % df_cur.name, className='nav-link', style={'fontSize':'1rem', 'position':'relative', 'right':'9rem'})),
        dbc.NavItem(dbc.NavLink("Zone Datapoints", href="/zones", style={'fontSize':'1.2rem'})),
        dbc.NavItem(dbc.NavLink("Time Series Data", href="/time-series", style={'fontSize':'1.7rem', 'color':'white'})),
    ],
    brand="AHU-3 Energy Dashboard",
    brand_href="/",
    brand_style={'fontSize':'1.2rem', 'color':'rgba(255,255,255,.5)'},
    color="#323232db",
    dark=True,
)

oa_table = dash_table.DataTable(
    id='oa-table',
    columns=[{'id': c, 'name': 'Outside Air'} for c in df_oa.columns],
    data=df_oa.to_dict('records'),
    style_table={
        'maxHeight': '50ex',
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(230, 230, 230)',
        'fontWeight': 'bold',
        'border': '1px solid black',
        'text-align': 'center',
        'fontSize': 18
    },
    style_data={
        'border': '1px solid grey',
        'text-align': 'center',
        'fontSize': 16
    },
    merge_duplicate_headers=True
)

sa_table = dash_table.DataTable(
    id='oa-table',
    columns=[{'id': c, 'name': 'Supply Air'} for c in df_sa.columns],
    data=df_sa.to_dict('records'),
    style_table={
        'maxHeight': '50ex',
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(230, 230, 230)',
        'fontWeight': 'bold',
        'border': '1px solid black',
        'text-align': 'center',
        'fontSize': 18
    },
    style_data={
        'border': '1px solid grey',
        'text-align': 'center',
        'fontSize': 16
    },
    merge_duplicate_headers=True
)

warnings_table = dash_table.DataTable(
    id='table',
    columns=[{'id': c, 'name': c} for c in df_w.columns],
    data=df_w.to_dict('records'),
    style_table={
        'maxHeight': '50ex',
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(230, 230, 230)',
        'fontWeight': 'bold',
        'border': '1px solid black',
        'text-align': 'center',
        'fontSize': 18
    },
    style_data={
        'border': '1px solid grey',
        'text-align': 'center',
        'fontSize': 16
    }
)

DAT_table = dash_table.DataTable(
    id='DATtable',
    columns=[{'id': i, 'name': 'SAT Reset'} for i in df_DAT.columns],
    data=df_DAT.to_dict('records'),
    style_table={
        'maxHeight': '50ex',
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'even'},
            'backgroundColor': 'rgb(250, 250, 250)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(225, 225, 225)',
        'border': '1px solid black',
        'text-align': 'center',
        'fontSize': 18,
        'fontWeight': 'bold'
    },
    style_data={
        'border': '1px solid grey',
        'text-align': 'center',
        'fontSize': 16,
    },
    merge_duplicate_headers=True
)


sp_table = dash_table.DataTable(
    id='SPtable',
    columns=[{'id': i, 'name': 'SP Reset'} for i in df_sp.columns],
    data=df_sp.to_dict('records'),
    style_table={
        'maxHeight': '50ex',
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'even'},
            'backgroundColor': 'rgb(250, 250, 250)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(225, 225, 225)',
        'border': '1px solid black',
        'text-align': 'center',
        'fontSize': 18,
        'fontWeight': 'bold'
    },
    style_data={
        'border': '1px solid grey',
        'text-align': 'center',
        'fontSize': 16,
    },
    merge_duplicate_headers=True
)

table_header = [
    html.Thead(html.Tr([html.Th(c) for c in df_load.columns if c != 'Colorscale']))
]
rows = []
for index, r in df_load.iterrows():
    rows.append(html.Tr([html.Td(r['Component'], style={'border': '1px solid grey'}),
                         html.Td([r['Load'], dcc.Graph(id='colorscale-%s' % r['Component'],
                                                       figure=scales[r['Component']],
                                                       className='-'.join(r['Component'].split()+['colorscale']),
                                                       config=dict(displayModeBar=False))],
                                 style={'border': '1px solid grey'}, className='load-row'),
                         html.Td(r['Hourly Cost'], style={'border': '1px solid grey'})]))

table_body = [html.Tbody(rows)]

load_table = dbc.Table(table_header + table_body, striped=False)

zone_table = dash_table.DataTable(
    id='zone-table',
    columns=[{'id': c, 'name': c} for c in df_zone.columns],
    data=df_zone.to_dict('records'),
    style_table={
        'maxHeight': '50ex',
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        }
    ],
    style_header={
        'backgroundColor': 'rgb(230, 230, 230)',
        'fontWeight': 'bold',
        'border': '1px solid black',
        'text-align': 'center',
        'fontSize': 18
    },
    style_data={
        'border': '1px solid grey',
        'text-align': 'center',
        'fontSize': 16
    },
    merge_duplicate_headers=True
)




