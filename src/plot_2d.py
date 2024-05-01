#!/usr/bin/env python3
import plotly.graph_objects as go


def plot_2d(df, x_axis, y_axis, labels, title):
    """Helper function to easily create interactive 2D plots using Plotly.

    :param df: DataFrame containing the columns to plot
    :type df: object
    :param x_axis: X-axis column name
    :type x_axis: str
    :param y_axis: Y-axis column name
    :type y_axis: str
    :param labels: Cluster labels from clustering algorithm
    :type labels: numpy.ndarray
    :param title: Plot title
    :type title: str
    """
    fig = go.Figure(data=go.Scatter(
        x=df[x_axis],
        y=df[y_axis],
        mode='markers',  # Define the plot type, 'markers' for scatter plot
        marker=dict(
            color=labels,  # Set marker colors based on labels
            colorscale='Viridis',  # Define the color scale
            showscale=True,  # Show color scale bar
        ),
        text=[f"Cluster: {label}" for label in labels],
        hoverinfo='text'
    ))

    # Update the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        autosize=False,  # Disable autosize to set custom width and height
        width=800,  # Set the width of the plot
        height=600,  # Set the height of the plot
        # Optionally, if you want to maintain a specific aspect ratio
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    # Show the plot
    fig.show()
