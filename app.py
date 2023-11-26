from memory_profiler import profile
from dash import Dash, dcc, Output, Input, State, no_update  # pip install dash
import dash_bootstrap_components as dbc
import math
import catboost
import copy
from urllib.request import urlopen
import json
from math import sqrt
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, Output, Input, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import io
import sklearn
from sklearn.model_selection import train_test_split
import pickle
import joblib

with urlopen('https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json') as response:
    harita = json.load(response)
df=pd.read_csv("https://raw.githubusercontent.com/alialtintass/Porsche/main/Porsche_old.csv", delimiter=",")
model_seri_list= pd.read_csv("https://raw.githubusercontent.com/alialtintass/Porsche/main/List.csv", delimiter=";")
df_trial=pd.read_csv("https://raw.githubusercontent.com/alialtintass/Porsche/main/df_trial.csv", delimiter=",")
X_train =pd.read_csv("https://raw.githubusercontent.com/alialtintass/Porsche/main/X_train.csv", delimiter=",")
df_grouped = pd.read_csv("https://raw.githubusercontent.com/alialtintass/Porsche/main/df_grouped.csv", delimiter=",")
url1 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part0.cbm"
url2 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part1.cbm"
url3 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part2.cbm"
url4 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part3.cbm"
url5 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part4.cbm"
url6 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part5.cbm"
url7 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part6.cbm"
url8 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part7.cbm"
url9 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part8.cbm"
url10 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part9.cbm"
url11 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part10.cbm"
url12 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part11.cbm"
url13 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part12.cbm"
url14 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part13.cbm"
url15 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part14.cbm"
url16 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part15.cbm"
url17 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part16.cbm"
url18 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part17.cbm"
url19 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part18.cbm"
url20 = "https://github.com/alialtintass/Porsche/raw/main/my_regressor_part19.cbm"
response1 = requests.get(url1)
file1 = io.BytesIO(response1.content)
response2 = requests.get(url2)
file2 = io.BytesIO(response2.content)
response3 = requests.get(url3)
file3 = io.BytesIO(response3.content)
response4 = requests.get(url4)
file4 = io.BytesIO(response4.content)
response5 = requests.get(url5)
file5 = io.BytesIO(response5.content)
response6 = requests.get(url6)
file6 = io.BytesIO(response6.content)
response7 = requests.get(url7)
file7 = io.BytesIO(response7.content)
response8 = requests.get(url8)
file8 = io.BytesIO(response8.content)
response9 = requests.get(url9)
file9 = io.BytesIO(response9.content)
response10 = requests.get(url10)
file10 = io.BytesIO(response10.content)
response11 = requests.get(url11)
file11 = io.BytesIO(response11.content)
response12 = requests.get(url12)
file12 = io.BytesIO(response12.content)
response13 = requests.get(url13)
file13 = io.BytesIO(response13.content)
response14 = requests.get(url14)
file14 = io.BytesIO(response14.content)
response15 = requests.get(url15)
file15 = io.BytesIO(response15.content)
response16 = requests.get(url16)
file16 = io.BytesIO(response16.content)
response17 = requests.get(url17)
file17 = io.BytesIO(response17.content)
response18 = requests.get(url18)
file18 = io.BytesIO(response18.content)
response19 = requests.get(url19)
file19 = io.BytesIO(response19.content)
response20 = requests.get(url20)
file20 = io.BytesIO(response20.content)
file1_bytes= file1.getvalue()
file2_bytes= file2.getvalue()
file3_bytes= file3.getvalue()
file4_bytes= file4.getvalue()
file5_bytes= file5.getvalue()
file6_bytes= file6.getvalue()
file7_bytes= file7.getvalue()
file8_bytes= file8.getvalue()
file9_bytes= file9.getvalue()
file10_bytes= file10.getvalue()
file11_bytes= file11.getvalue()
file12_bytes= file12.getvalue()
file13_bytes= file13.getvalue()
file14_bytes= file14.getvalue()
file15_bytes= file15.getvalue()
file16_bytes= file16.getvalue()
file17_bytes= file17.getvalue()
file18_bytes= file18.getvalue()
file19_bytes= file19.getvalue()
file20_bytes= file20.getvalue()
reassembled_data =file1_bytes + file2_bytes + file3_bytes + file4_bytes + file5_bytes + file6_bytes + file7_bytes + file8_bytes +  file9_bytes + file10_bytes + file11_bytes + file12_bytes + file13_bytes + file14_bytes + file15_bytes + file16_bytes +file17_bytes + file18_bytes + file19_bytes + file20_bytes
model_file = io.BytesIO(reassembled_data)
import tempfile
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(reassembled_data)

    # Load the CatBoost model from the temporary file
    model = catboost.CatBoost()
    model.load_model(temp_file.name)

df=df.drop_duplicates()
df = df.reset_index()
df['color'].replace('',np.nan,inplace=True)
df.dropna(subset=['color'], inplace=True)
df.reset_index(drop=True,inplace=True)
df[["Il", "Ilce"]] = df["location"].str.split(" ").apply(pd.Series)
df = df.drop('location', axis=1)
df['date'] = pd.to_datetime(df.date,dayfirst=True, format='%Y-%m-%d')
df['date'] =  df['date'].dt.strftime('%d/%m/%Y')
df['prices']=df['prices'].astype(str)
df.prices = df.prices.str.replace(' TL', '')
df.prices = df.prices.str.replace('.', '')
df['prices'] = pd.to_numeric(df['prices'])
df['km']=df['km'].astype(str)
df.km = df.km.str.replace(' TL', '')
df.km = df.km.str.replace(',', '')
df.km = df.km.str.replace('.', '')
df['km']=df['km'].astype(int)
df['year']=df['year'].astype(int)
choices = list()
conditions = list()
for item in harita['features']:
    conditions.append((df["Il"]==item['properties']['name']))
    choices.append(int(item['id']))

df['id'] = np.select(conditions, choices, default=0)
df['prices'] = df['prices'].apply(pd.to_numeric, downcast='float', errors='coerce')
df['km_median'] =  df['km'].groupby(df['year']).transform('median')
df['km / median']= df.km/df['km_median']
df.drop('index', axis=1, inplace=True)
df.isnull().sum()
X = copy.deepcopy(df[['year', 'km', 'color', 'model', 'seri', 'km / median']])
y = copy.deepcopy(df[['prices']])
X = pd.get_dummies(data=X, columns=['color', 'year', 'model', 'seri'])#, drop_first=True)
#let us drop stuff we select in order to eliminate multicollinearity
X.drop('year_2022', axis=1,inplace=True)
X.drop('color_Bej', axis=1,inplace=True)
X.drop('model_Taycan', axis=1,inplace=True)
X.drop('seri_Taycan', axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
data = copy.deepcopy(df[['prices','year', 'km', 'color', 'model', 'seri', 'km / median']])

df_grouped=df.groupby(['Il','id']).agg(avg_price=('prices','mean'),
                       max_price=('prices','max'),
                       min_price=('prices','min'),
                       data_counted=('Il','count')).reset_index()
df_grouped.set_index('id',inplace=True)
df_grouped['index1'] = df_grouped.index
df_grouped = df_grouped.reset_index(level=0)
df_grouped.sort_values('Il')

x_columns = list(X.columns)
unique_colors=list()
unique_years=list()
unique_models=list()
unique_seri=list()
for item in x_columns:
    if item[:5]=='color':
        unique_colors.append(item[6:])
    if item[:4]=='year':
        unique_years.append(int(item[5:]))
    if item[:5]=='model':
        unique_models.append(item[6:])
    if item[:4]=='seri':
        unique_seri.append(item[5:])
unique_years.append(2022)
unique_years.sort()
unique_colors.append('Bej')
unique_colors.sort()
unique_models.append('Taycan')
unique_models.sort()
unique_seri.append('Taycan')
unique_seri.sort()

data_colors=list(df.color.unique())
data_years=list(df.year.unique()).sort()
data_series=list(df.seri.unique())
df_trial=copy.deepcopy(df)
df_trial['date'] = pd.to_datetime(df_trial['date'], dayfirst=True)
series_weekly=df_trial.groupby([pd.Grouper(key='date', freq='W-MON')])['prices'].mean()
df_weekly=pd.DataFrame(data=series_weekly, columns=['prices'])
df_weekly.reset_index(inplace=True)
uniq_seri_all = copy.deepcopy(unique_seri)
uniq_seri_all.append('All Series')
seri_categories = list(model_seri_list['seri'].unique())
model_categories = list(model_seri_list['model'].unique())
year_seri = pd.DataFrame(df[['year', 'seri']].drop_duplicates())



app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
mytitle = dcc.Markdown(children='# An Application to Analyze The Prices Of Porsche Cars On Sahibinden')
mygraph = dcc.Graph(figure={})
myhistograms = dcc.Graph(figure={})
myscatterplot = dcc.Graph(figure={})
hist_subsection = dcc.Markdown(children='## What is The Relationship Between The Number Of The Cars On Sahibinden And Their Other Attributes?')
hist_dropdown = dcc.Dropdown(['Production year', 'Kilometer traveled', 'Color', 'Serie'], 
                             'Production year', id='hist-dropdown')
ml_title = dcc.Markdown(children='## Predict the price of your Porsche.')
slider_title = dbc.Button("Avg. prices", color="dark", className="me-1")
myslider = dcc.Slider(min=0,
                   max=8000000,
                   step=2000000,
                   value=0,
                   tooltip={"placement": "bottom", "always_visible": True},
                   updatemode='drag',
                   persistence=True,
                   persistence_type='session', # 'memory' or 'local'
                   id="myslider"
        )


seri_ddown_avg_prices = dcc.Dropdown(uniq_seri_all, 'All Series', id='seri_ddown_avg_prices')

model_dropdown = dcc.Dropdown(options=[], placeholder='Select models', id='model_dropdown')

seri_dropdown = dcc.Dropdown(options=[], placeholder='Select series', id='seri_dropdown')
color_dropdown = dcc.Dropdown(unique_colors, placeholder='Select colors', id='color_dropdown')
year_dropdown = dcc.Dropdown(id='year_dropdown',
        placeholder='Select a year',  # Set the placeholder directly
        value=None ) # Set the initial value to None  # Set the initial value to None)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([mytitle], width = 12)
    ]),
    dbc.Row([
        dbc.Col([mygraph], width = 12)
    ]),
    dbc.Row([
        dbc.Col([slider_title], width = 1),
        dbc.Col([myslider], width = 11)
    ]),
    dbc.Row([
        dbc.Col([hist_subsection], width = 12, align='end')
    ]),
    dbc.Row([
        dbc.Col([hist_dropdown], width = 3, align='center'),
        dbc.Col([myhistograms], width = 9, align='center')
    ]),
    dbc.Row([
        dbc.Col([ml_title], width = 12)
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Km traveled: '), width={"order": "first"}),
        dbc.Col(dcc.Slider(min=0,
                   max=math.ceil(X['km'].max()/100000)*100000,
                   value=100000,
                   marks=None,
                   tooltip={"placement": "bottom", "always_visible": True},
                   updatemode='drag',
                   persistence=True,
                   persistence_type='session', # 'memory' or 'local'
                   id="km_slider"
        ))
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Year: '), width={"order": "first"}),
        dbc.Col([year_dropdown], width = 12)
        #dbc.Col([dbc.Button('Submit', id='submit-year', n_clicks=0, color="primary")])
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Color: '), width={"order": "first"}),
        dbc.Col([color_dropdown], width = 12)
        #dbc.Col([dbc.Button('Submit color', id='submit-color', n_clicks=0, color="primary")])
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Seri: '), width={"order": "first"}),
        dbc.Col([seri_dropdown], width = 12)
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Model: '), width={"order": "first"}),
        dbc.Col([model_dropdown], width = 12)
    ]),
    dbc.Row([
        dbc.Col([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
    ], justify="center"),
    dbc.Row([
        dbc.Col([html.Div(children = dcc.Markdown(children = 'Prediction will be displayed here.', id= 'prediction output'))])
    ]),
    dbc.Row([
        dbc.Col(dcc.Markdown(children='## Weekly Average Prices Of Porsche Cars'), width = 12), #title
    ]),
    dbc.Row([
        dbc.Col([seri_ddown_avg_prices], width=2, align='center'),
        dbc.Col([myscatterplot], width = 10, align='center') #scatterplot
    ])
    ])


# Initialize options in the callback
@app.callback(
    Output('year_dropdown', 'options'),
    [Input('year_dropdown', 'value')]
)
def update_options(value):
    return [{'label': str(year), 'value': year} for year in unique_years]

@app.callback(
    Output('seri_dropdown', 'options'),
    Input('year_dropdown', 'value'))

def update_minor_dd(year_dropdown):
  
    major_minor = year_seri[['year', 'seri']].drop_duplicates()
    relevant_minor_options = major_minor[major_minor['year'] == year_dropdown]['seri'].values.tolist()
    
    # Create and return formatted relevant options with the same label and value
    formatted_relevant_minor_options = [{'label':x, 'value':x} for x in relevant_minor_options]
    return formatted_relevant_minor_options


@app.callback(
    Output('model_dropdown', 'options'),
    Input('seri_dropdown', 'value'))

def update_minor_dd(model_dropdown):
  
    major_minor = model_seri_list[['seri', 'model']].drop_duplicates()
    relevant_minor_options = major_minor[major_minor['seri'] == model_dropdown]['model'].values.tolist()
    
    # Create and return formatted relevant options with the same label and value
    formatted_relevant_minor_options = [{'label':x, 'value':x} for x in relevant_minor_options]
    return formatted_relevant_minor_options






@app.callback([Output(mygraph, component_property='figure'),
    Output(myhistograms, component_property='figure'),
    Output(myscatterplot, component_property='figure'),
    Output('prediction output','children')],
    [Input(myslider, component_property='value'),
    Input(hist_dropdown, component_property='value'),
    Input(seri_ddown_avg_prices, component_property='value'),
    Input('submit-val', 'n_clicks')],
    [State('km_slider', 'value'),
    State('year_dropdown', 'value'),
    State('color_dropdown', 'value'),
    State('model_dropdown', 'value'),
    State('seri_dropdown', 'value')]
)


def update_graph(user_input_1, user_input_2, user_input_3, user_input_4, km_state,
                 year_state, color_state, model_state, seri_state):
    # function arguments come from the component property of the Input
    """
    This function is necessary to create the graphs online
    """
    mask = df_grouped['avg_price']>=user_input_1
    df_grouped_fig = df_grouped[mask]
    fig = px.choropleth_mapbox(df_grouped_fig, geojson=harita, locations=df_grouped_fig['index1'],
                           color=df_grouped_fig['avg_price'],
                           color_continuous_scale='rainbow',
                           hover_name ="Il",
                           hover_data = ['data_counted'],
                           range_color=(df_grouped_fig.avg_price.min(),df_grouped_fig.avg_price.max()),
                           mapbox_style="carto-positron",
                           zoom=5, center={"lat": 38.9597594, "lon": 34.9249653},
                           opacity=0.5,
                           labels={'color':'prices','ID':'Il'},
                           title="Porsche Price Change In Turkey"
                          )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    if user_input_2 == 'Production year':
        figure = px.histogram(df, x="year", category_orders=dict(year=data_years),
                              color_discrete_sequence=['black'])
    elif user_input_2 == 'Kilometer traveled':
        figure = px.histogram(df, x="km", nbins=10, color_discrete_sequence=['black'])
    elif user_input_2 == 'Color':
        figure = px.histogram(df, x="color", category_orders = dict(color=data_colors),
                             color_discrete_sequence=['black'])
    elif user_input_2 == 'Serie':
        figure = px.histogram(df, x="seri", category_orders= dict(seri=data_series),
                             color_discrete_sequence=['black'])
    else:
        pass #display empty graph

    if user_input_3 == 'All Series':
        df_trial['date'] = pd.to_datetime(df_trial['date'], dayfirst=True)
        series_weekly=df_trial.groupby([pd.Grouper(key='date', freq='W-MON')])['prices'].mean()
        df_weekly=pd.DataFrame(data=series_weekly, columns=['prices'])
        df_weekly.reset_index(inplace=True)
        figur=px.line(df_weekly, x="date", y="prices", markers=True)
        figur.update_xaxes(tickangle=45)
    else:
        newdf = df_trial.loc[(df_trial.seri==user_input_3)]
        series_weekly=newdf.groupby([pd.Grouper(key='date', freq='W-MON')])['prices'].mean()
        df_weekly=pd.DataFrame(data=series_weekly, columns=['prices'])
        df_weekly.reset_index(inplace=True)
        figur=px.line(df_weekly, x="date", y="prices", markers=True)
        figur.update_xaxes(tickangle=45)
    if year_state is None:
        text= '# Please select Year, Color, Seri and Model '
        return [fig, figure, figur, text]  # returned objects are assigned to the component property of the Output
    else:

        input_km_median=list(df['km_median'][df['year']==int(year_state)])[0]
        model_columns=list(X_train.columns)
        input_x=[float(km_state), float(input_km_median)]+[0]*(len(model_columns)-2)
        year_input = 'year_'+str(year_state)
        color_input = 'color_'+str(color_state)
        model_input = 'model_'+str(model_state)
        seri_input = 'seri_'+ str(seri_state)
        for j in range(len(model_columns)):
            if year_input == model_columns[j]:
                input_x[j]= 1
            if color_input == model_columns[j]:
                input_x[j]= 1
            if model_input == model_columns[j]:
                input_x[j]= 1
            if seri_input == model_columns[j]:
                input_x[j]= 1
        df_input=pd.DataFrame(columns=model_columns)
        df_input[-1]=input_x
        guess= model.predict(input_x)
        text= '# The predicted price is '+ f'{round(guess,0):,.2f}'+ ' TL'
        # text= '#The predicted price is '+ '\033[1m' +str(round(guess, 2))+ ' TL \033[0m.'
        return [fig, figure, figur, text]  # returned objects are assigned to the component property of the Output
    # Run app
if __name__=='__main__':
    app.run_server(port=8065)
