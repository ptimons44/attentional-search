import random

import dash
from dash import dcc, html
import plotly.graph_objects as go

from dash.dependencies import Input, Output

import pandas as pd

# use absolute imports for custom modules
# from main import pipeline, deserialize_object
from query_graph.main import pipeline, deserialize_object


def df_from_researcher(researcher):
    # Create an empty list to store dictionaries of node attributes
    node_list = []

    # Iterate over each node in the researcher object
    for node in researcher.nodes:
        # Extract the attributes from the node
        text = node.text
        context = node.context
        url = node.url
        search_queries = node.search_queries
        relevant_sentences = [idx for (idx, val) in enumerate(node.relevant_sentences[0]) if val.item()]

        # Create a dictionary of node attributes
        node_dict = {
            'text': text,
            'context': context,
            'url': url,
            'search_queries': search_queries,
            'relevant_sentences': relevant_sentences,'entailment': [random.uniform(0, 1) for i in range(len(relevant_sentences))],
            'contradiction': [random.uniform(0, 1) for i in range(len(relevant_sentences))],
            'neutral': [random.uniform(0, 1) for i in range(len(relevant_sentences))],
            'relevance': random.uniform(0,1)
        }

        # Append the node dictionary to the list
        node_list.append(node_dict)

    # Create a DataFrame from the list of node dictionaries
    df = pd.DataFrame(node_list)
    return df

def get_researcher_df():
    researcher = deserialize_object("researcher.pkl")
    df = df_from_researcher(researcher)
    gpt_response = researcher.gpt_response
    gpt_sentences = researcher.gpt_sentences

    return df, gpt_response, gpt_sentences

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="QueryGraph Research Assistant"),
        dcc.Input(id='input', value='Enter your query here', type='text'),
        html.Button(id='submit-button', type='submit', children='Submit'),
        html.Div(id="gpt-response", children=""),
        dcc.Graph(id="graph", figure={}),  # main figure
        dcc.Graph(id="graph3D", figure={}),  # 3D graph
    ]
)

# generate the gpt response and main figure
@app.callback(
    [Output('gpt-response', 'children'), Output('graph', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input', 'value')]
)
def generate_main_figures(n_clicks, value):
    if n_clicks is None:
        return dash.no_update
    else:
        researcher_df, gpt_response, gpt_sentences = get_researcher_df()

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
        # generate the main graph
        if n_clicks is None:
            fig = dash.no_update
        else:
            fig = go.Figure()
            hover_text = [
                f"<b>{row['text']}</b><br>"
                f"<a href={row['url']} target='_blank'>Link to Paper</a>"
                for _, row in researcher_df.iterrows()
            ]

            for _, row in researcher_df.iterrows():
                for index in row['relevant_sentences']:
                    fig.add_trace(go.Scatter(
                        x=[index],
                        y=row["entailment"], 
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=50*row["relevance"]),
                    ))

            fig.update_layout(
                scene=dict(
                xaxis_title='GPT Sentence index',
                yaxis_title='Relevance'
                ),
                scattermode="group",
                scattergap=0.1,
                hoverlabel=dict(
                    bgcolor="white", 
                    font_size=15, 
                    font_family="Rockwell"
                )
            )

        return gpt_response_components, fig

import json

# 3D graph
@app.callback(
    Output('graph3D', 'figure'),
    [Input({'type': 'graph-button', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [dash.dependencies.State({'type': 'graph-button', 'index': dash.dependencies.ALL}, 'id')]
)
def build_graph_sentence(n_clicks, button_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    else:
        researcher_df, gpt_response, gpt_sentences = get_researcher_df()
        # Extract the index from the triggered button ID
        button_id_triggered = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        index = button_id_triggered['index']
        # Find the corresponding number of clicks for the button that triggered the callback
        n_clicks_triggered = n_clicks[button_ids.index(button_id_triggered)]
        if n_clicks_triggered is None:
            raise dash.exceptions.PreventUpdate
        else:
            # Generate the graph using the graph sentence
            fig3D = go.Figure()
            hover_text = [
                f"<b>{row['text']}</b><br>"
                f"<a href={row['url']} target='_blank'>Link to Paper</a>"
                for _, row in researcher_df.iterrows()
            ]
            for _, row in researcher_df.iterrows():
                if index in row['relevant_sentences']:
                    fig3D.add_trace(go.Scatter3d(
                        x=row['entailment'],
                        y=row['contradiction'],
                        z=row['neutral'],
                        hovertemplate=hover_text,
                        mode='markers',
                        marker=dict(size=20*row["relevance"]),
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
