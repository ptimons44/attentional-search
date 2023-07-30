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

#TODO: deugging cache
use_cahce = False

app = Dash(__name__)

# Header Components
header_components = [
    html.Img(src='/assets/brain.png', id='header-image'),
    html.Div(
        html.H1("Attentional Searcher", className='header-title'),
        className='header'
    )
]

# Input Components
input_components = [
    dcc.Input(id='input-query', value='Enter your query here', type='text', className='input'),
    html.Button(id='submit-query', type='submit', children='Submit', className='submit-button'),
    # number of search queries slider
    html.Div([
        html.Label('Number of queries to search:'),
        dcc.Slider(
            id='n-search-queries-slider',
            min=1,
            max=50,  # You can set this to a "reasonable" maximum
            value=10,  # Initial value
            marks={i: str(i) for i in range(0,101,10)},  # Display marks for every 10 steps
            step=1
        ),
        html.Div('Or input custom value of k:'),
        dcc.Input(
            id='n-search-queries-input',
            type='number',
            value=10,  # Initial value
        )
    ], className='n-search-queries-control'),  # We added a new className for potential styling

    html.Div(id='task-status', children='')

]



# Graph and Details
graph_details = [
    dcc.Graph(id="main-graph", figure={}, className='graph', style={'display': 'none'}),
    html.Div(id='details-panel', className='detail-panel')
]

# Stores and Intervals (mostly non-visual components that serve data/logic purposes)
data_stores_intervals = [
    dcc.Store(id='researcher-store'),
    dcc.Store(id='researcher-task-store'),
    dcc.Store(id='top-sentences-store', data=[]),
    dcc.Store(id="url-to-color-store"),
    dcc.Store(id='query-jobs-store'),
    dcc.Store(id='completed-query-jobs-store', data=[]),
    dcc.Store(id='compression-task-store'),
    dcc.Store(id='compressed-sentences-store'),
    dcc.Store(id='previous-compressed-sentences-store', data=None),
    dcc.Interval(
        id='update-interval',
        interval=5*1000,
        n_intervals=0
    ),
]

app.layout = html.Div([

    # Header Section
    *header_components,

    # Input Section
    html.Div(input_components, className='input-field'),

    # Main Content 
    html.Div(
        [
            html.Div(id="gpt-response-section", children="", className='gpt-response'),
            html.Div(graph_details, className='graph-detail-wrapper'),
            html.Div(id='loading-output-section', className='loading-output'),
            dcc.Checklist(id='url-dropdown-checklist', className='url-dropdown'),
        ],
        className='content'
    ),

    # Stores and Interval
    *data_stores_intervals

])

@app.callback(
    [Output('n-search-queries-slider', 'value'), Output('n-search-queries-input', 'value')],
    [Input('n-search-queries-slider', 'value'), Input('n-search-queries-input', 'value')]
)
def sync_slider_input(slider_val, input_val):
    ctx = dash.callback_context

    # If no components have been triggered, just return the current state
    if not ctx.triggered:
        return slider_val, input_val

    # Identify which component has triggered this callback
    component_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if component_id == 'n-search-queries-slider':
        return slider_val, slider_val
    else:
        return slider_val, input_val



import joblib
@app.callback(
    [Output('gpt-response-section', 'children'), Output('researcher-store', 'data'), Output('researcher-task-store', 'data')],
    [Input('submit-query', 'n_clicks')],
    [State('input-query', 'value'), State('n-search-queries-input', 'value')]
)
def get_llm_response(n_clicks, input_value, n_search_queries):
    if not n_clicks or not input_value:
        return "", None, None

    if use_cahce:
        researcher_result = joblib.load("cache/researcher.joblib")
        researcher_task_id = None
    else:
        # Pass the num_search_queries parameter to the init_researcher function
        task = init_researcher.apply_async(args=[input_value, n_search_queries])  # modify to use apply_async
        researcher_task_id = task.id
        researcher_result = task.get()  # this will block until the task is complete

    gpt_response_text = researcher_result["gpt_response"]

    return gpt_response_text, researcher_result, researcher_task_id


@app.callback(
    Output('query-jobs-store', 'data'),
    [Input('researcher-store', 'data')]
)
def trigger_jobs(researcher):
    if not researcher:
        return no_update

    task_ids = [query_to_sentences.delay(researcher, search_query).task_id for search_query in researcher["search_queries"]]
    return task_ids


@app.callback(
    [Output('top-sentences-store', 'data'),
     Output('completed-query-jobs-store', 'data')],
    [Input('update-interval', 'n_intervals')],
    [State('query-jobs-store', 'data'),
     State('top-sentences-store', 'data'),
     State('completed-query-jobs-store', 'data')]
)
def update_top_sentences(n, task_ids_data, existing_data, completed_tasks_data, k=100):
    
    # Convert data to the right format (list of strings)
    if task_ids_data:
        task_ids = task_ids_data
    else:
        task_ids = []
    if completed_tasks_data:
        completed_tasks = completed_tasks_data
    else:
        completed_tasks = []

    if not task_ids:
        return no_update, no_update

    pending_tasks = set(task_ids) - set(completed_tasks)
    new_completed_tasks = []

    all_results = []

    for task_id in pending_tasks:
        result = query_to_sentences.AsyncResult(task_id)
        if result.ready():
            all_results.extend(result.get())
            new_completed_tasks.append(task_id)

    with RedisLock('top_sentences_lock') as acquired:
        if acquired:
            existing_data.extend(all_results)
            existing_data.sort(key=lambda x: x['relevance'], reverse=True)
            updated_data = existing_data[:k]
        else:
            updated_data = existing_data

    # Update completed tasks
    completed_tasks += new_completed_tasks

    return updated_data, completed_tasks


@app.callback(
    Output('compression-task-store', 'data'),
    [Input('top-sentences-store', 'data'),
     Input('completed-query-jobs-store', 'data'),
     Input('query-jobs-store', 'data')]
)
def trigger_compression(sentences_data, completed_jobs_data, all_jobs_data):
    if not sentences_data or not completed_jobs_data or not all_jobs_data:
        return dash.no_update
    # Check if all the tasks are completed before triggering compression
    if set(completed_jobs_data) != set(all_jobs_data):
        return dash.no_update

    return compress_sentences.apply_async(args=[sentences_data]).id

@app.callback(
    Output('compressed-sentences-store', 'data'),
    [Input('update-interval', 'n_intervals')],
    [State('compression-task-store', 'data')]
)
def update_compressed_data(n_intervals, task_id):
    if not task_id:
        return no_update

    result = compress_sentences.AsyncResult(task_id)
    if result.ready():
        return result.get()
    
    return no_update

@app.callback(
    Output('previous-compressed-sentences-store', 'data'),
    [Input('compressed-sentences-store', 'data')]
)
def store_previous_compressed_data(compressed_data):
    return compressed_data

@app.callback(
    Output('main-graph', 'style'),
    [Input('compressed-sentences-store', 'data')]
)
def show_hide_graph(compressed_data):
    if not compressed_data:
        return {'display': 'none'}
    return {'display': 'block'} # Show the graph once there's data

@app.callback(
    Output('main-graph', 'figure'),
    [Input('compressed-sentences-store', 'data')],
    [State('top-sentences-store', 'data'),
     State('previous-compressed-sentences-store', 'data')]
)
def update_plot(compressed_data, sentences, previous_compressed_data):
    if not compressed_data or not sentences:
        return dash.no_update

    # Check if compressed data is same as previous
    if compressed_data == previous_compressed_data:
        return dash.no_update

    x_vals, y_vals, z_vals = zip(*compressed_data)  # Unzipping the 3D coordinates

    # Extract relevance values and scale them to be between min_size and max_size
    min_size = 5
    max_size = 20
    relevance_vals = [s['relevance'] for s in sentences]
    size_vals = [min_size + (max_size - min_size) * r for r in relevance_vals]

    trace = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        text=[s['text'] for s in sentences],
        mode='markers',
        marker=dict(size=size_vals),
        hoverinfo='text',
        hovertemplate="%{text}<extra></extra>"

    )


    layout = go.Layout(
        paper_bgcolor="#F6F0ED",  # Off-White for the overall chart's background
        plot_bgcolor="#F6F0ED",  # Off-White for the plotting area's background
        title_font=dict(color="#4C3B4D"),  # Dark Purple for the title text
        font=dict(color="#4C3B4D"),  # Dark Purple for other text elements
        title='Compressed Embedding Space Visualization',
        hovermode='closest'
    )

    return {'data': [trace], 'layout': layout}

@app.callback(
    Output('details-panel', 'children'),
    [Input('main-graph', 'clickData')],
    [State('top-sentences-store', 'data')]
)
def display_click_data(clickData, sentences_data):
    if not clickData:
        return dash.no_update
    # Get the clicked point's index
    point_index = clickData['points'][0]['pointNumber']
    clicked_sentence = sentences_data[point_index]
    content = [
        html.H5("Context:"),
        html.P(clicked_sentence['context']),
        html.H5("Search Query:"),
        html.P(clicked_sentence['search_query']),
        html.H5("URL:"),
        html.A(clicked_sentence['url'], href=clicked_sentence['url'], target="_blank")
    ]

    return html.Div(content)


@app.callback(
    Output('task-status', 'children'),
    [Input('update-interval', 'n_intervals')],
    [State('researcher-task-store', 'data'),
     State('query-jobs-store', 'data'),
     State('completed-query-jobs-store', 'data'),
     State('compression-task-store', 'data')]
)
def update_task_status(n_intervals, researcher_task, query_tasks, completed_query_tasks, compression_task):
    # Check status of init_researcher task first
    researcher_status = init_researcher.AsyncResult(researcher_task).status if researcher_task else None
    if researcher_status == "PENDING" or researcher_status == "STARTED":
        return "Getting LLM response..."

    # If init_researcher is finished or hasn't started yet, proceed with other tasks:

    # Initially, no tasks
    if not query_tasks:
        return "No tasks started yet."

    # Check statuses for query_to_sentences tasks
    total_tasks = len(query_tasks)
    completed_tasks = len(completed_query_tasks) if completed_query_tasks else 0
    pending_tasks = total_tasks - completed_tasks

    if pending_tasks > 0:
        return f"{completed_tasks} out of {total_tasks} search queries completed. {pending_tasks} pending..."

    # Check status for compress_sentences task
    compression_status = compress_sentences.AsyncResult(compression_task).status if compression_task else None
    if compression_status == "PENDING":
        return "Compression task is pending..."
    elif compression_status == "STARTED":
        return "Compression task is running..."
    elif compression_status == "FAILURE":
        return "Compression task failed. Please try again."

    # If none of the above, all tasks are completed
    return "All tasks completed!"




if __name__ == '__main__':
    app.run_server(debug=True)

