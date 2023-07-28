# use absolute imports for custom modules
from query_graph.logger import logger

from celerysetup import celery_app
from redislock import RedisLock

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

from dash import Dash, CeleryManager, Input, Output, State, html, callback, dcc, no_update
import dash

from query_graph.tasks import init_researcher, query_to_sentences, compress_sentences

app = Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='researcher'),
    dcc.Store(id='top-sentences', data=[]),
    dcc.Store(id="url-to-color"),
    dcc.Store(id='query-jobs'),
    dcc.Store(id='completed-query-jobs', data=[]),
    dcc.Store(id='compression-task'),
    dcc.Store(id='compressed-sentences'),


    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds, i.e., every 5 seconds
        n_intervals=0
    ),

    # Header
    html.Div(
        html.H1("Attentional Searcher", className='header-title'),
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
        ],
        className='content'
    )

], className='container')


import joblib
@app.callback(
    [Output('gpt-response', 'children'), Output('researcher', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('input', 'value')]
)
def update_output(n_clicks, input_value):
    if not n_clicks or not input_value:
        return "Enter a query and click submit!", None

    # Directly call the function (blocking the execution until completion)
    try:
        researcher_result = joblib.load("cache/researcher.joblib")
    except FileNotFoundError:
        researcher_result = init_researcher(input_value)
    
    gpt_response_text = researcher_result["gpt_response"]

    return gpt_response_text, researcher_result



@app.callback(
    Output('query-jobs', 'data'),  # a dummy output
    [Input('researcher', 'data')]
)
def trigger_jobs(researcher):
    if not researcher:
        return dash.no_update

    # This list will hold all task_ids for tasks we're firing off
    task_ids = []

    for search_query in researcher["search_queries"]: 
        result = query_to_sentences.delay(researcher, search_query)
        task_ids.append(result.task_id)

    # Convert the list of task_ids to a comma-separated string to store in the dummy div
    return ','.join(task_ids)



@app.callback(
    [Output('top-sentences', 'data'),
     Output('completed-query-jobs', 'data')],
    [Input('interval-component', 'n_intervals')],
    [State('query-jobs', 'data'),
     State('top-sentences', 'data'),
     State('completed-query-jobs', 'data')]
)
def update_top_sentences(n, task_ids_string, existing_data, completed_tasks, k=100):
    if not task_ids_string:
        return dash.no_update, dash.no_update

    task_ids = task_ids_string.split(',')
    new_completed_tasks = []

    all_results = []

    for task_id in task_ids:
        # Check if task_id is not in completed_tasks
        if task_id not in completed_tasks:
            result = query_to_sentences.AsyncResult(task_id)
            if result.ready():  # Checks if the task is done
                all_results.extend(result.get())
                new_completed_tasks.append(task_id)  # Add task_id to completed list

    with RedisLock('top_sentences_lock') as acquired:
        if acquired:
            existing_data.extend(all_results)
            existing_data.sort(key=lambda x: x['relevance'], reverse=True)
            updated_data = existing_data[:k]
        else:
            updated_data = existing_data

    # Return updated top sentences and updated completed tasks list
    return updated_data, completed_tasks + new_completed_tasks

@app.callback(
    Output('compression-task', 'data'),
    [Input('top-sentences', 'data')]
)
def trigger_compression(sentences_data):
    if not sentences_data:
        return dash.no_update
    
    task = compress_sentences.apply_async(args=[sentences_data])
    
    return task.id


@app.callback(
    Output('compressed-sentences', 'data'),
    [Input('interval-component', 'n_intervals')],
    [State('compression-task', 'data')]
)
def update_compressed_data(n_intervals, task_id):
    if not task_id:
        return dash.no_update

    result = compress_sentences.AsyncResult(task_id)
    if result.ready():
        return result.get()
    
    return dash.no_update


import plotly.graph_objs as go

@app.callback(
    Output('graph', 'figure'),
    [Input('compressed-sentences', 'data')],
    [State('top-sentences', 'data')]
)
def update_plot(compressed_data, sentences):
    if not compressed_data or not sentences:
        return dash.no_update

    x_vals, y_vals = zip(*compressed_data)  # Unzipping the 2D coordinates

    trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        text=[s['text'] for s in sentences],
        mode='markers',
        hoverinfo='text'
    )

    layout = go.Layout(
        title='Compressed Sentences Visualization',
        hovermode='closest'
    )

    return {'data': [trace], 'layout': layout}

@app.callback(
    Output('detail-panel', 'children'),
    Input('graph', 'clickData'),
    State('top-sentences', 'data')
)
def display_click_data(clickData, sentences_data):
    if not clickData:
        return dash.no_update

    # Get the clicked point's index
    point_index = clickData['points'][0]['pointIndex']
    clicked_sentence = sentences_data[point_index]

    return [
        html.H5("Context:"),
        html.P(clicked_sentence['context']),
        html.H5("Search Query:"),
        html.P(clicked_sentence['search_query']),
        html.H5("URL:"),
        html.A(clicked_sentence['url'], href=clicked_sentence['url'], target="_blank")
    ]






if __name__ == '__main__':
    app.run_server(debug=True)
    # logger.debug("Debug message")
    # logger.info("Info message")
    # logger.warning("Warning message")
    # logger.error("Error message")
