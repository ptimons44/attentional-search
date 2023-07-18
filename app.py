import plotly.graph_objects as go
from dash import Dash, DiskcacheManager, CeleryManager, Input, Output, State, html, callback, dcc, no_update
import diskcache
import dash


import json
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# use absolute imports for custom modules
# from query_graph.main import get_llm_response, get_web_content

import time
    

cache = diskcache.Cache("./the-cache")
background_callback_manager = DiskcacheManager(cache)


app = Dash(__name__)

# app.layout = html.Div(
#     children=[
#         dcc.Store(id='researcher-dict'),
#         html.H1(children="Googolplex Research Assistant"),
#         dcc.Input(id='input', value='Enter your query here', type='text'),
#         html.Button(id='submit-button', type='submit', children='Submit'),
#         html.Div(id="gpt-response", children=""),
#         html.Div(id='loading-output'),
#         dcc.Checklist(id='url-dropdown'),
#         html.Div(id='detail-panel'),
#         dcc.Graph(id="graph", figure={}),  # main figure
#         dcc.Graph(id="graph3D", figure={}),  # 3D graph
#     ]
# )

app.layout = html.Div([
    dcc.Store(id='researcher-dict'),

    # Header
    html.Div(
        html.H1("Googolplex Research Assistant", className='header-title'),
        className='header'
    ),

    # Input and Button
    html.Div(
        [
            dcc.Input(id='input', value='Enter your query here', type='text', className='input'),
            html.Button(id='submit-button', type='submit', children='Submit', className='submit-button'),
        ],
        className='input-field'
    ),

    # Content 
    html.Div(
        [
            html.Div(id="gpt-response", children="", className='gpt-response'),
            html.Div(id='loading-output', className='loading-output'),
            dcc.Checklist(id='url-dropdown', className='url-dropdown'),
            html.Div(id='detail-panel', className='detail-panel'),
            dcc.Graph(id="graph", figure={}, className='graph'),  # main figure
            dcc.Graph(id="graph3D", figure={}, className='graph3D'),  # 3D graph
        ],
        className='content'
    )

], className='container')




def get_llm_response(value):
    return ['To determine the healthiest diet for fish, it is important to consider their natural habitat, species and feeding habits.',
 'Different fish species have varying dietary requirements, and providing them with a well-balanced and appropriate diet is crucial for their overall health and well-being.',
 '1.',
 'Research the natural habitat: Start by understanding the natural diet of the specific fish species you are caring for.',
 'For example, if you have a herbivorous fish like the convict cichlid, they naturally feed on plants and algae.',
 'Knowing this, you can include plant matter in their diet.',
 '2.',
 'Understanding feeding habits: Fish can be classified as herbivores, carnivores, or omnivores, and their feeding habits play a vital role in determining their diet.',
 'For instance, a carnivorous fish like the betta fish primarily consumes meat-based foods.',
 'It is important to feed them a diet rich in protein, such as live or frozen bloodworms or brine shrimp.',
 '3.',
 'Providing variety: Just like humans, fish benefit from a varied diet as it ensures they receive a range of nutrients.',
 'Feeding a single type of food may result in nutritional deficiencies.',
 'For example, you can offer flakes or pellets designed specifically for your fish species, while also supplementing their diet with occasional live or frozen foods, like daphnia or mysis shrimp.',
 '4.',
 'Avoid overfeeding: Overfeeding can lead to various health problems, such as obesity and poor water quality.',
 "It's essential to feed fish an appropriate amount of food, considering their size, activity level, and metabolism.",
 'Generally, it is recommended to feed fish small portions multiple times a day, rather than one large feeding.',
 '5.',
 'Observing feeding behavior: Monitor your fish during feeding time.',
 'If they show disinterest in the food or leave it uneaten, it may be an indication that the diet is not suitable or appealing to them.',
 'In such cases, adjusting the type or brand of food might be necessary.',
 "It's important to note that the above information provides a general guideline, but each fish species has specific dietary requirements, so it's advisable to research the specific needs of your fish species or consult with a knowledgeable aquatic veterinarian or fisheries expert for tailored advice."]


# generate the gpt response
@callback(
    Output('gpt-response', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input', 'value')
)
def generate_gpt_response(n_clicks, value):
    if n_clicks is None:
        return no_update
    else:
        # researcher = get_llm_response(value)
        # researcher_obj.append(researcher)
        gpt_sentences = get_llm_response(value)

        gpt_response_components = [
            html.Div(
                [
                    html.Span(
                        gpt_sentence + "  ",
                        id={'type': 'sentence', 'index': i},
                        className='clickable-paragraph'
                    ),
                    html.Button(
                        f"sentence {i} graph",
                        id={'type': 'graph-button', 'index': i},
                        n_clicks=0
                    )
                ]
            )
            for i, gpt_sentence in enumerate(gpt_sentences)
        ]
        return gpt_response_components
    
@callback(
    Output('researcher-dict', 'value'),
    Input('gpt-response', 'children')
)
def generate_researcher_dict(gpt_response):
    print("sleeping 2 seconds")
    time.sleep(2)
    with open("researcher.pkl", 'rb') as pickle_file:
        return pickle.load(pickle_file)

# Dropdown for URL selection
@callback(
    Output('url-dropdown', 'options'),
    Input('researcher-dict', 'value')
)
def update_dropdown(researcher_dict):
    unique_urls = list(set([sentence['url'] for sentence in researcher_dict.values()]))
    return [{'label': url, 'value': url} for url in unique_urls]

# side panel for text
@callback(
    Output('detail-panel', 'children'),
    Input('graph', 'clickData'),
    Input('researcher-dict', 'value')
)
def update_detail_panel(clickData, researcher_dict):    
    # Extract the key of the clicked point from the 'customdata' field
    key = str(clickData['points'][0]['customdata'])
    
    # Check if the key is in the researcher_dict
    if key in researcher_dict:
        # Look up the corresponding sentence
        sentence = researcher_dict[key]
        
        # Format the sentence and its context as Markdown
        # markdown_text = f"[Link to Paper]({sentence['url']})  \n\n{sentence['context']}"
        markdown_text = dcc.Markdown(
            f"[Link to Paper]({sentence['url']})  \n\n{sentence['context']}",
            style={
                'padding': '20px',
                'font-family': 'Roboto, sans-serif',
                'color': '#333',  # Dark grey
                'background-color': '#f5f5f5',  # Light grey
            }
        )
        
        return markdown_text
    else:
        print(f"Key {key} not found in researcher_dict")
        # return "No data"
        return dcc.Markdown("No data", style={'padding': '20px'})



# Update figure based on URL selection
@callback(
    Output('graph', 'figure'),
    [Input('url-dropdown', 'value'),
     Input('researcher-dict', 'value')]
)
def update_figure(selected_urls, researcher_dict):
    # Generate n different colors
    num_urls = len(selected_urls)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_urls))

    # Assign colors to urls
    url_to_color = {url: colors[i] for i, url in enumerate(selected_urls)}

    fig = go.Figure()

    for (sentence_i, sentence) in enumerate(researcher_dict.values()):
        if sentence['url'] in selected_urls:
            for index in sentence['relevant_sentences']:
                fig.add_trace(go.Scatter(
                    x=[index],
                    y=sentence["entailment"],
                    hovertemplate=sentence["text"],
                    mode='markers',
                    marker=dict(size=50*sentence["relevance"], color=url_to_color[sentence['url']]),
                    customdata=[sentence_i]
                ))

    fig.update_layout(
        xaxis=dict(
            title='GPT Sentence index',
            dtick=1  # tick marks every 1 unit
        ),
        yaxis=dict(
            title='Relevance'
        ),
        scattermode="group",
        scattergap=0.1,
        hoverlabel=dict(
            bgcolor="white",
            font_size=15,
            font_family="Rockwell"
        )
    )
    return fig

# update 3D graph based on gpt sentence selection
@callback(
    Output('graph3D', 'figure'),
    Input({'type': 'graph-button', 'index': dash.dependencies.ALL}, 'n_clicks'),
    State('researcher-dict', 'value')
)
def build_graph_sentence(n_clicks, researcher_dict):
    print("callback fired")
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    else:
        triggered_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        index = int(triggered_id["index"])
        
        if index is None:
            raise dash.exceptions.PreventUpdate
        else:
            # Convert the index to an integer
            index = int(index)
            
            # Generate the graph using the graph sentence
            fig3D = go.Figure()
            for sentence in researcher_dict.values():
                if index in sentence['relevant_sentences']:
                    fig3D.add_trace(go.Scatter3d(
                        x=sentence['entailment'],
                        y=sentence['contradiction'],
                        z=sentence['neutrality'],
                        hovertemplate=sentence["text"],
                        mode='markers',
                        marker=dict(size=20*sentence["relevance"]),
                    )
                )
            fig3D.update_layout(scene=dict(
                xaxis_title='Entailment',
                yaxis_title='Contradiction',
                zaxis_title='Neutral'
            ))
            return fig3D




if __name__ == '__main__':
    app.run_server(debug=True)
