import pandas as pd
import numpy as np
import dash
from dash import html
from dash import dcc
from dash import dash_table
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input,Output
from dash import callback_context
from jupyter_dash import JupyterDash  # pip install dash
from sklearn.manifold import TSNE

#app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
usernames = ['user_id','gender','age','occupation','zip'] 
users = pd.read_table('users.dat', sep='::', header=None, names=usernames, engine='python')

ratingnames = ['user_id', 'movie_id', 'rating', 'timestamp'] 
ratings = pd.read_table('ratings.dat', sep='::', header=None, names=ratingnames, engine='python')

movienames = ['movie_id', 'title', 'genres'] 
movies = pd.read_table('movies.dat', sep='::', header=None, names=movienames, engine='python', encoding="latin") 

for ind in movies.index:
    #print(sample_movies['genres'][ind])
    string=movies['genres'][ind]
    nums = string.split('|')
    #print(nums[0] )
    movies['genres'][ind]=nums[0]
    #print(sample_movies['genres'][ind])
    #print("++++++++++++++++++++")

    
data = pd.merge(pd.merge(ratings, users), movies)
aux_df=data.sample(n=1000)
df=aux_df.copy();

from sklearn.preprocessing import LabelEncoder

def Encoder(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return[df] #added [] on df

columnsToEncode = list(df.select_dtypes(include=['category','object']))
le = LabelEncoder()
for feature in columnsToEncode:
              try:
                  data[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
df = Encoder(df)

print(df[0]) #to see if the categorical data has been encoded


from sklearn import preprocessing

x = df[0].values #returns a numpy array #added [0] on df
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df2 = pd.DataFrame(x_scaled,columns=df[0].columns)  #added [0] on df


load_figure_template('LUX')


###--------------Build the figures / dropdowns------------------------------------

x = np.random.sample(100)
y = np.random.sample(100)
z = np.random.choice(a = ['a','b','c'], size = 100)


df1 = pd.DataFrame({'x':x, 'y':y, 'z':z}, index = range(100))

#fig1 = px.scatter(aux_df, x= "tsne-2d-one", y = "tsne-2d-two", color = 'genres')
sample_df_selected = df2.copy()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(sample_df_selected)
aux_df['tsne-2d-one'] = tsne_results[:,0]
aux_df['tsne-2d-two'] = tsne_results[:,1]
fig1 = px.scatter(aux_df, x= "tsne-2d-one", y = "tsne-2d-two", size='rating', color='genres', size_max=10)

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])




###---------------Create the layout of the app ------------------------

#app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.LUX])


app.layout = html.Div(
    [
        dbc.Row(
            dbc.Col([html.H1('Dashboard for Group Testing and Hipothesis Generation'), html.Hr()])
        ),
        dbc.Row(
            [
                dbc.Col(
                [
                    html.H3("Filters"),
                    html.Hr(),
                    dbc.DropdownMenu(size="sm",children=[
                        dcc.Checklist(
                        id = 'my_checklist', 
                        options=aux_df.columns.values[0:10], 
                        value=aux_df.columns.values[0:10],  
                        labelStyle = dict(display='block'))
                    ],id = 'one',label="Features for Proj"),
                    html.Br(),
                    dbc.DropdownMenu(size="sm",children=[
                      dcc.RadioItems(
                      id = 'RadioItems', 
                      options=aux_df.columns.values[0:10], 
                      value="genres",  
                      labelStyle = dict(display='block'))
                    ],id = 'two',label="Color Dimension"),
                    html.Br(),
                    dbc.DropdownMenu(size="sm",children=[
                      dcc.RadioItems(
                      id = 'SizePoints', 
                      options=aux_df.columns.values[0:10], 
                      value="rating",  
                      labelStyle = dict(display='block'))
                    ],id = 'three',label="Size Dimension")

                ],width=2
                ),
                dbc.Col(
                [
                    html.H6("Hipothesis of i-th iterarion: XXXXX-XXXXXX-XXXXXX-XXXXXXX-XXXXXX-XXXXXX"),
                    html.H6("General Data Projection of Dataframe D_i in the i-th iteration:"),
                    dbc.Row(dcc.Graph(id = 'graph1', figure = fig1, responsive=True)),
                   
                ]),
               dbc.Col(
                [
                    html.H6("Data Statistics Summary of D_i"),
                    html.H6("MEAN, MAXPVAL, MINPVAL, SUMPVAL, COVERAGE"),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    #px.bar(
                        #data_frame=df2.mean(axis=0),
                        #x=filtered_BM,
                        #y="WEIGHT",
                        #color="INDEX_NAME",
                        #opacity=0.9,
                        #barmode='group'),
                    
                    html.H6("Data Statistics for the Selected Regions (Chose Circles) (Dataframe: D_i+1):"),
                ],width=3),
                #dbc.Col(dcc.Graph(id = 'graph1', figure = fig1), width = 9, style = {'margin-left':'15px', 'margin-top':'7px', 'margin-right':'15px'})
            ]
        ),
         dbc.Row([
             html.H4("History:"),
             #dbc.Col(
             #   [html.H4("Iteration D_1:"),dcc.Graph(id = 'graph3', figure = fig1, responsive=True)]),
            # dbc.Col(
            #    [html.H4("Iteration D_2:"),dcc.Graph(id = 'graph4', figure = fig1, responsive=True)]),
            #dbc.Col(
            #    [html.H4("Iteration D_3:"),dcc.Graph(id = 'graph5', figure = fig1, responsive=True)]),

             ]),
        dbc.Row([
                    html.H4('Original Data'),
                    #dash_table.DataTable(data=aux_df[aux_df.columns[0:10]])
                    dash_table.DataTable(
                      data=aux_df.to_dict('records'),
                      columns=[{'id': c, 'name': c} for c in aux_df.columns[0:10]],
                      page_size=10
                  )
                    #generate_table(aux_df[aux_df.columns[0:10]])
        ])
    ]
)

#app.layout = html.Div([
#              dbc.Row(dbc.Col(html.Div('Dashboard for Group Testing and Hipothesis Generation'))
#                  #style={
#                  #    'textAlign': 'center'
#                      #'color': colors['text']
#                  #}
#                  ),
#              dbc.Row([
#                    dbc.Col(sidebar),
#                    dbc.Col(html.H1('General Data Projection'),width = 9, style = {'margin-left':'7px','margin-top':'5px'}),
#                    dbc.Col(dcc.Graph(id = 'graph1', figure = fig1), width = 9, style = {'margin-left':'15px', 'margin-top':'7px', 'margin-right':'15px'})
#                ]),
#              dbc.Row([
#                    html.H4(children='US Agriculture Exports (2011)'),
#                    generate_table(aux_df)
#                ])
#            ])




@app.callback(
    Output(component_id='graph1', component_property='figure'),
    Input(component_id='my_checklist', component_property='value'),
    Input(component_id='RadioItems', component_property='value'),
    Input(component_id='SizePoints', component_property='value'))
def update_chart(options_chosen1,optionschosen2,optionschosen3):
    #print(options_chosen)
    sample_df_selected = df2[options_chosen1].copy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(sample_df_selected)
    aux_df['tsne-2d-one'] = tsne_results[:,0]
    aux_df['tsne-2d-two'] = tsne_results[:,1]
    fig1 = px.scatter(aux_df, x= "tsne-2d-one", y = "tsne-2d-two", size=optionschosen3, color = optionschosen2, size_max=10)
    return (fig1)

#@app.callback(
#    Output(component_id='graph1', component_property='figure'),
#    Input(component_id='RadioItems', component_property='value'), prevent_initial_call=True
#)
#def update_chart2(option_chosen):
#    #print(options_chosen)
#    fig1 = px.scatter(aux_df, x= "tsne-2d-one", y = "tsne-2d-two", color = option_chosen)
#    return (fig1)

if __name__ == '__main__':
    app.run_server(mode='inline', debug=True)