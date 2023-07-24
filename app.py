# use absolute imports for custom modules
from query_graph.pipeline import get_llm_response, get_web_content, all_gpt_sentence_3D
from query_graph.logger import logger

from celerysetup import celery_app


import json
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from dash import Dash, CeleryManager, Input, Output, State, html, callback, dcc, no_update
import dash

app = Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='prelim-researcher-dict'),
    dcc.Store(id='researcher-dict'),
    dcc.Store(id="url-to-color"),
    dcc.Store(id="background-task-id"),

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

    dcc.Interval(id="interval", interval=5*1000, n_intervals=0),  # in milliseconds

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



# generate the gpt response
@callback(
    Output('gpt-response', 'children'),
    Output('prelim-researcher-dict', 'value'),
    Input('submit-button', 'n_clicks'),
    State('input', 'value')
)
def generate_gpt_response(n_clicks, value):
    if n_clicks is None:
        return no_update, no_update
    else:
        logger.debug("submit button clicked")
        researcher = get_llm_response(value)
        gpt_sentences = researcher["gpt_sentences"]

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
        return gpt_response_components, researcher
    

@celery_app.task(bind=True)
def get_web_content_task(self, researcher_dict):
    logger.info("Task started!")
    result = get_web_content(researcher_dict)
    logger.info(f"Task completed with result: {result}")
    return result
    
@callback(
    Output('background-task-id', 'data'),
    Input('gpt-response', 'children'),
    State('prelim-researcher-dict', 'value')
)
def generate_researcher_dict(gpt_response, researcher):
    logger.debug("generating researcher dict")
    task = get_web_content_task.delay(researcher)
    return {"task_id": task.id}

@callback(
    Output('researcher-dict', 'value'),
    Output('loading-output', 'children'),  # to give feedback to the user
    Input('interval', 'n_intervals'),
    State('background-task-id', 'data')
)
def update_output(n, data):
    if data:
        task_id = data["task_id"]
        task = get_web_content_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            return no_update, 'Task is still running...'
        elif task.state != 'FAILURE':
            result = task.result
            # Do anything with the result if you need to.
            return result, 'Task completed!'
        else:
            return no_update, 'An error occurred.'
    return no_update, ''

# Dropdown for URL selection
@callback(
    Output('url-dropdown', 'options'),
    Input('researcher-dict', 'value')
)
def update_dropdown(researcher_dict):
    unique_urls = list(set([sentence['url'] for sentence in researcher_dict["sentences"]]))
    return [{'label': url, 'value': url} for url in unique_urls]

# color coding for urls
@callback(
    Output('url-to-color', 'data'),
    Input('url-dropdown', 'value')
)
def map_urls_to_colors(selected_urls):
    # Generate n different colors
    num_urls = len(selected_urls)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_urls))

    # Assign colors to urls
    url_to_color = {url: colors[i] for i, url in enumerate(selected_urls)}
    return url_to_color


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
    Input("url-to-color", "data"),
    Input('researcher-dict', 'value')
)
def update_figure(url_to_color, researcher_dict):
    fig = go.Figure()

    for (sentence_i, sentence) in enumerate(researcher_dict["sentences"]):
        if sentence['url'] in url_to_color:
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
    Input('url-to-color', 'data'),
    Input({'type': 'graph-button', 'index': dash.dependencies.ALL}, 'n_clicks'),
    State('researcher-dict', 'value')
)
def build_graph_sentence(url_to_color, n_clicks, researcher_dict):
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
            for sentence in researcher_dict["sentences"]:
                if sentence["url"] in url_to_color:
                    if index in sentence['relevant_sentences']:
                        print("adding sentence", sentence["text"])
                        fig3D.add_trace(go.Scatter3d(
                        x=sentence['entailment'],
                        y=sentence['contradiction'],
                        z=sentence['neutrality'],
                        hovertemplate=sentence["text"],
                        mode='markers',
                        marker=dict(size=25*sentence["relevance"], color=url_to_color[sentence['url']]),
                    ))
            fig3D.update_layout(scene=dict(
                xaxis_title='Entailment',
                yaxis_title='Contradiction',
                zaxis_title='Neutral'
            ))
            return fig3D




if __name__ == '__main__':
    # app.run_server(debug=True)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
