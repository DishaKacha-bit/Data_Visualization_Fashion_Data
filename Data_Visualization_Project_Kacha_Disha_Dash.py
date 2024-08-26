import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from flask import send_file
import io
import plotly.figure_factory as ff
from dash.exceptions import PreventUpdate
from sklearn.preprocessing import LabelEncoder
import numpy as np
from dash import dash_table


fashion_data = pd.read_csv('/Users/dishakacha/Downloads/Data_Visualization/Data Visualization/Project/mock_fashion_data_uk_us.csv')

fashion_data_2 = pd.DataFrame(np.random.randn(100, 5), columns=['Price', 'Rating', 'Review Count', 'Age', 'Sales'])

# Keep only the numeric columns
numeric_columns = ['Price', 'Rating', 'Review Count', 'Age']
fashion_data_numeric = fashion_data_2[numeric_columns]

fashion_data_1 = pd.DataFrame({
    'Category': np.random.choice(fashion_data['Category'].unique(), 1000),
    'Purchase History': np.random.choice(fashion_data['Purchase History'].unique(), 1000),
    'Color': np.random.choice(fashion_data['Color'].unique(), 1000)
})

# Convert categorical columns to numeric using LabelEncoder
encoder = LabelEncoder()

fashion_data_1['Category'] = encoder.fit_transform(fashion_data_1['Category'])
fashion_data_1['Purchase History'] = encoder.fit_transform(fashion_data_1['Purchase History'])
fashion_data_1['Color'] = encoder.fit_transform(fashion_data_1['Color'])

fig = go.Figure(data=[go.Scatter3d(
    x=fashion_data_1['Category'],  # X-axis data (Category)
    y=fashion_data_1['Purchase History'],  # Y-axis data (Purchase History)
    z=fashion_data_1['Color'],  # Z-axis data (Color)
    mode='markers',  # Plot mode
    marker=dict(
        size=5,  # Marker size
        opacity=0.8,  # Marker opacity
        color=fashion_data_1['Color'],  # Color based on Color feature
        colorscale='Viridis',  # Color scale
        colorbar=dict(title='Color')  # Color bar title
    )
)])

# Create scatter plot
fig.update_layout(
    title='3D Scatter Plot using Tooltip',  # Title
    width=800,  # Adjust width of the plot
    height=600,
    scene=dict(
        xaxis=dict(
            title='Category',  # X-axis label
            showticklabels=False,  # Hide tick labels
            showgrid=True,  # Show grid lines
            gridcolor='rgba(255, 255, 255, 0.2)'  # Grid color
        ),
        yaxis=dict(
            title='Purchase History',  # Y-axis label
            showticklabels=False,  # Hide tick labels
            showgrid=True,  # Show grid lines
            gridcolor='rgba(255, 255, 255, 0.2)'  # Grid color
        ),
        zaxis=dict(
            title='Color',  # Z-axis label
            showticklabels=False,  # Hide tick labels
            showgrid=True,  # Show grid lines
            gridcolor='rgba(255, 255, 255, 0.2)'  # Grid color
        ),
    ),
    plot_bgcolor='rgba(255, 255, 255, 255)'  # Set plot background color
)
# Initialize the Dash app
app = dash.Dash('May app')

# Unique brands and seasons
unique_brands = fashion_data['Brand'].unique()
unique_seasons = fashion_data['Season'].unique()
unique_ratings = sorted(fashion_data['Rating'].unique())
fashion_influencers = fashion_data['Fashion Influencers'].unique()
unique_magazines = fashion_data['Fashion Magazines'].unique()
unique_categories = fashion_data['Category'].unique()
unique_purchase_history = fashion_data['Purchase History'].unique()
unique_colors = fashion_data['Color'].unique()


app.layout = html.Div(style={'background-image': 'url("/assets/background.jpeg")',
                             'background-size': 'cover',
                             'background-repeat': 'no-repeat'},
                      children=[
    html.Div([
        html.Img(src='/assets/logo.jpeg', style={"width": "50px", "height": "50px", "margin-right": "10px"}),
        html.H1("Fashion UK-US", style={"display": "inline"}),
        html.Br(),
    ], style={"text-align": "center"}),

    dcc.Tabs([
        dcc.Tab(label='Season wise Analysis', children=[
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='season-dropdown',
                        options=[{'label': season, 'value': season} for season in unique_seasons],
                        multi=True,
                        value=unique_seasons
                    ),
                    dcc.Loading(
                        id="loading-stacked-bar",
                        type="default",
                        children=[dcc.Graph(id='stacked-bar')]
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Loading(
                        id="loading-pie-chart",
                        type="default",
                        children=[dcc.Graph(id='pie-chart')]
                    )
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            html.Div([
                dcc.Loading(
                    id="loading-line-chart",
                    type="default",
                    children=[dcc.Graph(id='line-chart')]
                )
            ]),
            html.Button("Download Data for Tab 1", id="btn-download-tab1"),
            dcc.Download(id="download-data-tab1")
        ]),
        dcc.Tab(label='Rating by Magazines and Brand based on Size and Age', children=[
            html.Div([
                dcc.Checklist(
                    id='sizes-checklist',
                    options=[{'label': size, 'value': size} for size in fashion_data['Available Sizes'].unique()],
                    value=fashion_data['Available Sizes'].unique()
                ),
                dcc.RangeSlider(
                    id='age-slider',
                    min=fashion_data['Age'].min(),
                    max=fashion_data['Age'].max(),
                    step=1,
                    value=[fashion_data['Age'].min(), fashion_data['Age'].max()],
                    marks={i: str(i) for i in range(fashion_data['Age'].min(), fashion_data['Age'].max() + 1, 5)},
                    verticalHeight=50
                ),
                html.Div([
                    dcc.Loading(
                        id="loading-magazine-price-graph",
                        type="default",
                        children=[dcc.Graph(id='magazine-price-graph')]
                    ),
                    dcc.Loading(
                        id="loading-style-attributes-donut",
                        type="default",
                        children=[dcc.Graph(id='style-attributes-donut')]
                    )
                ]),
                html.Button("Download Data for Tab 2", id="btn-download-tab2"),
                dcc.Download(id="download-data-tab2")
            ])
        ]),
        dcc.Tab(label='Prices of Fashion Influencers', children=[
            html.Div([
                html.Label('Select Fashion Influencer:'),
                dcc.RadioItems(
                    id='fashion-influencer-radio',
                    options=[{'label': influencer, 'value': influencer} for influencer in fashion_influencers],
                    value=fashion_influencers[0]
                ),
                html.Div(id='selected-influencer-output'),
                dcc.Loading(
                    id="loading-line-graph",
                    type="default",
                    children=[dcc.Graph(id='line-graph')]
                )
            ]),
            html.Button("Download Data for Tab 3", id="btn-download-tab3"),
            dcc.Download(id="download-data-tab3")
        ]),
        dcc.Tab(label='Review Count based on Fashion Magazines', children=[
            html.Div([
                html.Label('Select Review Count:'),
                dcc.Slider(
                    id='review-count-slider',
                    min=fashion_data['Review Count'].min(),
                    max=fashion_data['Review Count'].max(),
                    step=1,
                    value=fashion_data['Review Count'].mean(),
                    marks={i: str(i) for i in
                           range(fashion_data['Review Count'].min(), fashion_data['Review Count'].max() + 1, 100)}
                ),
                html.Label('Select Fashion Magazine:'),
                dcc.Dropdown(
                    id='fashion-magazine-dropdown',
                    options=[{'label': magazine, 'value': magazine} for magazine in unique_magazines],
                    multi=False,
                    value=unique_magazines[0]
                ),
                html.Div(id='metric-output'),
                dcc.Loading(
                    id="loading-rug-plot",
                    type="default",
                    children=[dcc.Graph(id='rug-plot')]
                )
            ]),
            html.Button("Download Data for Tab 4", id="btn-download-tab4"),
            dcc.Download(id="download-data-tab4")
        ]),
        dcc.Tab(label='Tool Tip', children=[
            html.Div([
                dcc.Loading(
                    id="loading-scatter-plot",
                    type="default",
                    children=[dcc.Graph(id='scatter-plot', figure=fig, clear_on_unhover=True)]
                ),
                dcc.Tooltip(id='scatter-tooltip'),
                html.Button("Download Data for Tab 5", id="btn-download-tab5"),
                dcc.Download(id="download-data-tab5")
            ])
        ]),
        dcc.Tab(label='Data Processing', children=[
            html.Div([
                html.H2("Correlation Matrix"),
                dcc.Graph(id='correlation-matrix'),

                html.H2("Box Plot"),
                dcc.Graph(id='box-plot'),

                html.H2("Heatmap & Pearson Correlation Coefficient Matrix"),
                dcc.Graph(id='heatmap'),

                html.H2("Statistics"),
                dash_table.DataTable(
                    id='statistics-table',
                    style_table={'overflowX': 'scroll'},
                ),
                html.Button("Download Data for Tab 6", id="btn-download-tab6"),
                dcc.Download(id="download-data-tab6")
            ])
        ]),
        dcc.Tab(label='Feedback', children=[
            html.Div([
                html.Img(src='/assets/fashion.jpeg', style={"display": "block", "margin": "auto"}),
                html.H2("Feedback", style={"text-align": "center"}),
                dcc.Textarea(id="feedback-textarea", placeholder="Enter your feedback here...",
                             style={"width": "50%", "margin": "auto", "display": "block"}),
                html.Button("Submit", id="submit-feedback-button", n_clicks=0,
                            style={"display": "block", "margin": "auto"})
            ])
        ])
    ])
])

@app.callback(
    Output("download-data-tab1", "data"),
    Input("btn-download-tab1", "n_clicks"),
    prevent_initial_call=True
)
def download_data_tab1(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(fashion_data.to_csv, "tab1_data.csv")

@app.callback(
    Output("download-data-tab2", "data"),
    Input("btn-download-tab2", "n_clicks"),
    prevent_initial_call=True
)
def download_data_tab2(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(fashion_data.to_csv, "tab2_data.csv")

@app.callback(
    Output("download-data-tab3", "data"),
    Input("btn-download-tab3", "n_clicks"),
    prevent_initial_call=True
)
def download_data_tab3(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(fashion_data.to_csv, "tab3_data.csv")

@app.callback(
    Output("download-data-tab4", "data"),
    Input("btn-download-tab4", "n_clicks"),
    prevent_initial_call=True
)
def download_data_tab4(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(fashion_data.to_csv, "tab4_data.csv")

@app.callback(
    Output("download-data-tab5", "data"),
    Input("btn-download-tab5", "n_clicks"),
    prevent_initial_call=True
)
def download_data_tab5(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(fashion_data.to_csv, "tab5_data.csv")

@app.callback(
    Output("download-data-tab6", "data"),
    Input("btn-download-tab6", "n_clicks"),
    prevent_initial_call=True
)
def download_data_tab6(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return dcc.send_data_frame(fashion_data.to_csv, "tab6_data.csv")

# Callback to update stacked bar graph
@app.callback(
    Output('stacked-bar', 'figure'),
    [Input('season-dropdown', 'value')]
)
def update_stacked_bar(selected_seasons):
    # Filter data based on selected seasons
    filtered_data = fashion_data[fashion_data['Season'].isin(selected_seasons)]

    # Group by brand and season, calculate total price for each group
    grouped_data = filtered_data.groupby(['Brand', 'Season']).agg({'Price': 'sum'}).reset_index()

    # Create stacked bar graph
    traces = []
    for season in selected_seasons:
        season_data = grouped_data[grouped_data['Season'] == season]
        trace = go.Bar(x=season_data['Brand'], y=season_data['Price'], name=season)
        traces.append(trace)

    layout = go.Layout(
        title='Price by Brand and Season',
        barmode='stack',
        margin=dict(b=150)  # Adjust bottom margin to fit the graph properly
    )

    return {'data': traces, 'layout': layout}


# Callback to update line chart
@app.callback(
    Output('line-chart', 'figure'),
    [Input('season-dropdown', 'value')]
)
def update_line_chart(selected_seasons):
    # Filter data based on selected seasons
    filtered_data = fashion_data[fashion_data['Season'].isin(selected_seasons)]

    # Group by season, calculate mean price for each season
    season_prices = filtered_data.groupby('Season')['Price'].mean()

    # Create line chart
    fig = go.Figure(data=go.Scatter(x=season_prices.index, y=season_prices, mode='lines+markers'))
    fig.update_layout(title='Average Price by Season', xaxis_title='Season', yaxis_title='Price')

    return fig


# Callback to update pie chart
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('season-dropdown', 'value')]
)
def update_pie_chart(selected_seasons):
    # Filter data based on selected seasons
    filtered_data = fashion_data[fashion_data['Season'].isin(selected_seasons)]

    # Calculate total price for each season
    season_prices = filtered_data.groupby('Season')['Price'].sum()

    # Create pie chart
    fig = go.Figure(data=[go.Pie(labels=season_prices.index, values=season_prices)])
    fig.update_layout(title='Price Percentage by Season')

    return fig
@app.callback(
    Output('magazine-price-graph', 'figure'),
    [Input('sizes-checklist', 'value'),
     Input('age-slider', 'value')]
)
def update_magazine_price_graph(selected_sizes, age_range):
    # Filter data based on selected sizes and age range
    filtered_data = fashion_data[
        (fashion_data['Available Sizes'].isin(selected_sizes)) &
        (fashion_data['Age'] >= age_range[0]) &
        (fashion_data['Age'] <= age_range[1])
    ]

    # Group data by Magazine and Brand, then calculate average price
    grouped_data = filtered_data.groupby(['Fashion Magazines', 'Brand'])['Rating'].count().reset_index()

    # Create a grouped bar chart
    fig = px.bar(
        grouped_data,
        x='Fashion Magazines',
        y='Rating',
        color='Brand',  # This assigns different colors to each brand
        barmode='group',
        title='Count of Rating by Magazine and Brand'
    )

    # Optional: improve the layout to make the chart clearer
    fig.update_layout(
        xaxis_title='Fashion Magazines',
        yaxis_title='Count of Rating',
        legend_title='Brand'
    )

    return fig
@app.callback(
    Output('style-attributes-donut', 'figure'),
    [Input('sizes-checklist', 'value'),
     Input('age-slider', 'value')]
)
def update_style_attributes_donut(selected_sizes, age_range):
    # Filter data based on selections
    filtered_data = fashion_data[
        (fashion_data['Available Sizes'].isin(selected_sizes)) &
        (fashion_data['Age'] >= age_range[0]) &
        (fashion_data['Age'] <= age_range[1])
    ]

    # Calculate the percentage for each style attribute
    style_counts = filtered_data['Style Attributes'].value_counts(normalize=True)

    # Create a donut chart
    fig = px.pie(
        names=style_counts.index,
        values=style_counts,
        title='Style Attributes Distribution',
        hole=0.5  # This creates the donut shape
    )

    # Optional: improve the layout
    fig.update_traces(textposition='inside', textinfo='percent')
    fig.update_layout(legend_title='Style Attributes')

    return fig

# Callback to update the output based on selected fashion influencer
@app.callback(
    Output('selected-influencer-output', 'children'),
    [Input('fashion-influencer-radio', 'value')]
)
def update_selected_influencer(selected_influencer):
    return f'You selected: {selected_influencer}'

@app.callback(
    Output('line-graph', 'figure'),
    [Input('fashion-influencer-radio', 'value')]
)
def update_line_graph(selected_influencer):
    # Filter data based on selected Fashion Influencer
    filtered_data = fashion_data[fashion_data['Fashion Influencers'] == selected_influencer]

    # Group by Brand and calculate the average Price for each Brand
    brand_prices = filtered_data.groupby('Brand')['Price'].mean().reset_index()

    # Create line graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=brand_prices['Brand'], y=brand_prices['Price'], mode='lines+markers', name=selected_influencer))
    fig.update_layout(title=f'Price by Brand for {selected_influencer}', xaxis_title='Brand', yaxis_title='Price')

    return fig


@app.callback(
    Output('metric-output', 'children'),
    [Input('review-count-slider', 'value'),
     Input('fashion-magazine-dropdown', 'value')]
)
def display_selected_metric(review_count, selected_magazine):
    print("Review Count:", review_count)
    print("Selected Magazine:", selected_magazine)

    # Filter data based on selected Review Count and Fashion Magazine
    filtered_data = fashion_data[(fashion_data['Review Count'] == review_count) &
                                 (fashion_data['Fashion Magazines'] == selected_magazine)]

    print("Filtered Data:")
    print(filtered_data)

    # Check if the filtered data is empty
    if filtered_data.empty:
        return html.Div("No data available for the selected criteria.")

    # Extract the selected metric values
    time_period_highest_purchase = filtered_data['Time Period Highest Purchase'].iloc[0]
    customer_reviews = filtered_data['Customer Reviews'].iloc[0]
    social_media_comments = filtered_data['Social Media Comments'].iloc[0]
    feedback = filtered_data['feedback'].iloc[0]

    # Format the output
    output_string = html.Div([
        html.H3(f'Selected Fashion Magazine: {selected_magazine}'),
        html.P(f'Review Count: {review_count}'),
        html.P(f'Time Period Highest Purchase: {time_period_highest_purchase}'),
        html.P(f'Customer Reviews: {customer_reviews}'),
        html.P(f'Social Media Comments: {social_media_comments}'),
        html.P(f'Feedback: {feedback}')
    ])

    return output_string


@app.callback(
    Output('rug-plot', 'figure'),
    [Input('review-count-slider', 'value'),
     Input('fashion-magazine-dropdown', 'value')]
)
def update_histogram(review_count_value, magazine_value):
    # Filter data based on selected review count and fashion magazine
    filtered_data = fashion_data[(fashion_data['Review Count'] == review_count_value) &
                                 (fashion_data['Fashion Magazines'] == magazine_value)]

    # Check if filtered_data is empty
    if filtered_data.empty:
        # Handle the case when no data is available for the specified filters
        fig = go.Figure()
        fig.update_layout(title='No Data Available for Selected Filters')
        return fig

    # Create histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=filtered_data['Price'], nbinsx=20))

    # Update layout
    fig.update_layout(
        title='Price Distribution for Selected Filters',
        yaxis_title='Frequency',  # Add y-axis title
        margin=dict(t=50, b=50, l=50, r=50),  # Adjust margins for better visibility
        xaxis=dict(showticklabels=True)  # Remove x-axis tick labels
    )

    return fig

# Define callback to display tooltips
@app.callback(
    [Output('scatter-tooltip', 'show'),
     Output('scatter-tooltip', 'bbox'),
     Output('scatter-tooltip', 'children')],
    [Input('scatter-plot', 'hoverData')]
)
def display_hover(hoverData):
    if hoverData is None:
        return False, dash.no_update, dash.no_update

    pt = hoverData['points'][0]
    bbox = pt['bbox']
    num = pt['pointNumber']

    row = fashion_data.iloc[num]
    category = row['Category']
    purchase_history = row['Purchase History']
    color = row['Color']

    children = [
        html.Div([
            html.H2('Data Info'),
            html.P(f'Category: {category}'),
            html.P(f'Purchase History: {purchase_history}'),
            html.P(f'Color: {color}')
        ])
    ]

    return True, bbox, children

# Callbacks for Tab 6 components
# Callbacks for Tab 6 components
@app.callback(
    Output('correlation-matrix', 'figure'),
    Output('box-plot', 'figure'),
    Output('heatmap', 'figure'),
    Output('statistics-table', 'data'),
    Input('correlation-matrix', 'id')  # Placeholder input, change as needed
)
def update_tab6_components(_):
    # 1. Show Correlation matrix for numeric columns
    corr_matrix = fashion_data_numeric.corr()
    fig_corr_matrix = px.imshow(corr_matrix, title='Correlation Matrix')

    # 2. Box plot to check outliers for numeric columns
    fig_box_plot = px.box(fashion_data_numeric, title='Box Plot')

    # 3. Heatmap & Pearson correlation coefficient matrix for numeric columns
    fig_heatmap = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        annotation_text=corr_matrix.values.round(2),
        showscale=True
    )
    fig_heatmap.update_layout(title='Heatmap & Pearson Correlation Coefficient Matrix')

    # 4. Statistics for numeric columns
    stats_data = fashion_data_numeric.describe().reset_index()
    # Format numeric values to 2 decimal places
    stats_data = stats_data.applymap(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x)
    stats_data = stats_data.to_dict('records')

    return fig_corr_matrix, fig_box_plot, fig_heatmap, stats_data


if __name__ == '__main__':
    app.run_server(debug=True)

