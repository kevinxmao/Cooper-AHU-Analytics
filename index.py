import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from app import app, server
from layouts import layout_home, layout_zones,navbar_ts #layout_ts#,
import codecs


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', className='layout')
])


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/zones':
        return layout_zones
    elif pathname == '/time-series':
        fig_oa = codecs.open("fig_oa.html", 'r')
        fig_ahu = codecs.open("fig_ahu.html", 'r')
        fig_load = codecs.open("fig_load.html", 'r')
        fig_T = codecs.open("fig_T.html", 'r')
        fig_SUP = codecs.open("fig_SUP.html", 'r')
        fig_RH = codecs.open("fig_RH.html", 'r')
        fig_D = codecs.open("fig_D.html", 'r')
        layout_ts = html.Div([
            navbar_ts,
            html.H1(children='Time Series Data',
                    style={'textAlign': 'center', 'margin': '0', 'fontSize': '2.25rem', 'fontWeight': 'bold'}),
            html.Div([
                html.Div(html.Iframe(srcDoc=fig_oa.read(), className='myframe'),
                    className="container5"
                ),
                html.Div(html.Iframe(srcDoc=fig_ahu.read(), className='myframe'),
                    className="container5"
                ),
            ], className="row3"
            ),
            html.Div([
                html.Div(html.Iframe(srcDoc=fig_load.read(), className='myframe'),
                    className="container5"
                ),
                html.Div(html.Iframe(srcDoc=fig_T.read(), className='myframe'),
                    className="container5"
                ),
            ], className="row3"
            ),
            html.Div([
                html.Div(html.Iframe(srcDoc=fig_SUP.read(), className='myframe'),
                    className="container5"
                ),
                html.Div(html.Iframe(srcDoc=fig_RH.read(), className='myframe'),
                    className="container5"
                ),
            ], className="row3"
            ),
            html.Div([
                html.Div(
                    html.Iframe(srcDoc=fig_D.read(), className='myframe'),
                    className="container5"
                )], className="row3"
            )
        ])
        return layout_ts
    else:
        return layout_home

if __name__ == '__main__':
    app.run_server()

