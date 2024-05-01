#!/usr/bin/env python3
import plotly.graph_objects as go


def plot_3d(df, x_axis, y_axis, z_axis, labels, title, color=None):
    """Helper function to easily create interactive 3D plots using Plotly.

    :param df: DataFrame containing the columns to plot
    :type df: object
    :param x_axis: X-axis column name
    :type x_axis: str
    :param y_axis: Y-axis column name
    :type y_axis: str
    :param z_axis: Z-axis column name
    :type z_axis: str
    :param labels: Cluster labels from clustering algorithm
    :type labels: numpy.ndarray
    :param title: Plot title
    :type title: str
    :param color: Color column if a fourth dimension is desired, defaults to None
    :type color: str, optional
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x_axis],
        y=df[y_axis],
        z=df[z_axis],
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8,
            color=df[color],
            colorbar=dict(title=color)
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            zaxis_title=z_axis
        ),
        autosize=False,
        width=1000,
        height=800
    )

    fig.show()
