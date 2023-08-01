from query_graph.logger import logger
from query_graph.tasks import init_researcher, query_to_sentences, compress_sentences

from celerysetup import celery_app
from redislock import RedisLock

import plotly.graph_objects as go
import plotly.express as px

from dash import Dash, CeleryManager, Input, Output, State, html, callback, dcc, no_update, callback_context
import dash_dangerously_set_inner_html

import re

# Using a color set from Plotly Express
COLORS = px.colors.qualitative.Set1
color_map = {}  # Global variable to store color assignments

def get_color_for_query(query):
    if query not in color_map:
        color_map[query] = COLORS[len(color_map) % len(COLORS)]
    return color_map[query]


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
    # Wrap search bar, submit button in a flex container
    html.Div([
        dcc.Input(id='input-query', value='Enter your query here', type='text', className='input', style={'flex': '1'}),
        html.Button(id='submit-query', type='submit', children='Submit', className='submit-button', style={'marginLeft': '10px'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'width': '80%', 'margin': '0 auto'}),  # Centering the elements

    # Task status right below the search bar for a compact look
    html.Div(id='task-status', children='', className='task-status-text', style={'textAlign': 'center', 'marginTop': '5px'}),

    # Slider and custom value input for k in another flex container
    html.Div([
        html.Div([
            html.Label('Number of queries to search:'),
            dcc.Slider(
                id='n-search-queries-slider',
                min=1,
                max=50,
                value=10,
                marks={i: str(i) for i in range(0,50,10)},
                step=1
            ),
        ], style={'flex': '1'}),  # Giving it a flex of 1 to take up available space
        html.Div([
            html.Label('Or input custom value:'),
            dcc.Input(
                id='n-search-queries-input',
                type='number',
                value=10,
                style={'display': 'block', 'marginTop': '5px'}  # This positions the input below the label
            )
        ], style={'marginLeft': '20px'})
    ], className='n-search-queries-control', style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '10px', 'marginBottom': '0px'})
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
    dcc.Store(id='query-jobs-store'),
    dcc.Store(id='completed-query-jobs-store', data=[]),
    dcc.Store(id='compression-task-store'),
    dcc.Store(id='compressed-sentences-store'),
    dcc.Store(id='previous-compressed-sentences-store', data=None),
    dcc.Store(id='previous-selected-queries-store'),
    dcc.Interval(
        id='update-interval',
        interval=5*1000,
        n_intervals=0
    ),
    html.Div(id='graph-width-store'),  # Hidden Div to store the graph width
    html.Div(dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
        <script>
            // Sample JS code to run on a particular event, like window resize
            window.addEventListener('resize', function(){
                var graphWidth = document.getElementById("main-graph").offsetWidth;
                // Send this width value to the hidden div
                document.getElementById('graph-width-store').innerText = graphWidth;
            });
        </script>
    '''))
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
            dcc.Checklist(
                id='query-checklist',
                options=[],  # start with an empty list
                value=[],  # and no selected values
                style={'display': 'none'}
            )
        ],
        className='content'
    ),

    # Stores and Interval
    *data_stores_intervals,

])

@app.callback(
    [Output('n-search-queries-slider', 'value'), Output('n-search-queries-input', 'value')],
    [Input('n-search-queries-slider', 'value'), Input('n-search-queries-input', 'value')]
)
def sync_slider_input(slider_val, input_val):
    ctx = callback_context

    # If no components have been triggered, just return the current state
    if not ctx.triggered:
        return slider_val, input_val

    # Identify which component has triggered this callback
    component_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if component_id == 'n-search-queries-slider':
        return slider_val, slider_val
    else:
        return slider_val, input_val


def clean_word(word):
    # Remove punctuation from the word
    cleaned_word = re.sub(r'[^\w\s]', '', word)
    return cleaned_word.lower().strip()

@app.callback(
    [Output('query-checklist', 'options'),
    Output('query-checklist', 'value'),
    Output('gpt-response-section', 'children'),
    Output('researcher-store', 'data'),
    Output('researcher-task-store', 'data')],
    [Input('submit-query', 'n_clicks')],
    [State('input-query', 'value'), State('n-search-queries-input', 'value')]
)
def get_llm_response(n_clicks, input_value, n_search_queries):
    if not n_clicks or not input_value:
        return no_update, no_update, "", None, None

    # Pass the num_search_queries parameter to the init_researcher function
    task = init_researcher.apply_async(args=[input_value, n_search_queries])
    researcher_task_id = task.id
    researcher_result = task.get()

    unique_search_queries = researcher_result["search_queries"]
    options = [{'label': query, 'value': query} for query in unique_search_queries]

    gpt_response_text = researcher_result["gpt_response"]
    attentions = researcher_result["attention_to_word"]
    words = researcher_result["words"] # non stop words
    all_words = re.findall(r'(\W+|\w+)', researcher_result["gpt_response"])

    # Process and highlight gpt_response_text
    highlighted_words = []
    ptr_tracked_words = 0
    ptr_all_words = 0
    while ptr_all_words < len(all_words):
        if ptr_tracked_words < len(words):
            tok_word, word = words[ptr_tracked_words], all_words[ptr_all_words].lower()
            if tok_word == word:
                attention = attentions[ptr_tracked_words]
                ptr_tracked_words += 1
            else:
                attention = 0
    
        rgba_color = f"rgba(0, 128, 128, {attention})"
        # Check if the current component is a word or not.
        if all_words[ptr_all_words].isalnum():  # This will be True for words
            word_span = html.Span(all_words[ptr_all_words], style={'backgroundColor': rgba_color, 'marginRight': '5px'})
        else:
            word_span = html.Span(all_words[ptr_all_words])  # Just create the span without extra styles for spaces/punctuation
        highlighted_words.append(word_span)
        ptr_all_words += 1

    return options, unique_search_queries, highlighted_words, researcher_result, researcher_task_id



@app.callback(
    Output('query-jobs-store', 'data'),
    [Input('researcher-store', 'data'),
    Input('researcher-task-store', 'data')]
)
def trigger_jobs(researcher, researcher_task_id):
    if not researcher:
        return no_update

    task_ids = [query_to_sentences.delay(researcher, search_query, researcher_task_id).task_id for search_query in researcher["search_queries"]]
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
        return no_update
    # Check if all the tasks are completed before triggering compression
    if set(completed_jobs_data) != set(all_jobs_data):
        return no_update

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
    Output('previous-selected-queries-store', 'data'),
    [Input('query-checklist', 'value')]
)
def store_previous_selected_queries(selected_queries):
    return selected_queries


@app.callback(
    [Output('main-graph', 'style'),
    Output('query-checklist', 'style')],
    [Input('compressed-sentences-store', 'data')]
)
def show_hide_graph(compressed_data):
    if not compressed_data:
        return {'display': 'none'}, {'display': 'none'}
    return {'display': 'block'}, {'display': 'block'} # Show the graph once there's data

def insert_line_breaks(sentences, words_per_line=10):
    formatted_sentences = []
    for sentence in sentences:
        words = sentence.split()
        lines = [' '.join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
        formatted_sentence = '<br>'.join(lines)
        formatted_sentences.append(formatted_sentence)
    return formatted_sentences


@app.callback(
    Output('main-graph', 'figure'),
    [Input('compressed-sentences-store', 'data'),
    Input('query-checklist', 'value'),
    Input('graph-width-store', 'children')],
    [State('top-sentences-store', 'data'),
     State('previous-compressed-sentences-store', 'data'),
     State('previous-selected-queries-store', 'data')]
)
def update_plot(compressed_data, selected_queries, graph_width, sentences, previous_compressed_data, previous_selected_queries):
    if not compressed_data or not sentences or not selected_queries:
        return no_update

    # Check if update is warranted
    if (compressed_data == previous_compressed_data) and (selected_queries == previous_selected_queries):
        return no_update

    # Convert the graph_width from string to integer
    graph_width = int(graph_width) if graph_width else None

    # Use graph_width to determine the number of words or characters per line
    # This part is a bit tricky and will likely require some experimentation
    # For now, I'm just assuming 10 characters per 100 pixels of width as an example
    chars_per_line = (graph_width // 100) * 10 if graph_width else 50

    # Filter sentences and compressed data based on the selected queries
    filtered_sentences = [s for i, s in enumerate(sentences) if s['search_query'] in selected_queries]
    filtered_data = [compressed_data[i] for i, s in enumerate(sentences) if s['search_query'] in selected_queries]

    # Determine the color of each sentence using the consistent color map
    color_vals = [get_color_for_query(s['search_query']) for s in filtered_sentences]

    x_vals, y_vals, z_vals = zip(*filtered_data)

    # Extract relevance values and scale them to be between min_size and max_size
    min_size = 5
    max_size = 20
    relevance_vals = [s['relevance'] for s in filtered_sentences]
    size_vals = [min_size + (max_size - min_size) * r for r in relevance_vals]

    trace = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        text=insert_line_breaks([s['text'] for s in filtered_sentences]),
        mode='markers',
        marker=dict(color=color_vals, size=size_vals),
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
        return no_update

    # Get the clicked point's index
    point_index = clickData['points'][0]['pointNumber']
    clicked_sentence = sentences_data[point_index]
    content = [
        html.H5("Sentence:"),
        html.P(clicked_sentence['text']), 
        html.Hr(),  # This will add a horizontal line for clearer separation
        html.H5("Context:"),
        html.P(clicked_sentence['context']),
        html.Hr(),
        html.H5("Search Query:"),
        html.P(clicked_sentence['search_query']),
        html.Hr(),
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