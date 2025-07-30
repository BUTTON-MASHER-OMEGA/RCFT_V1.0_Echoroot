import numpy as np
import plotly.graph_objs as go

class KernelDynamicsPlot:
    """
    Interactive surface plot of memory kernel K(Δt) over λ or α parameters.
    """
    def __init__(self):
        pass

    def plot_exponential(self, lambda_vals, delta_t):
        z = np.exp(-np.outer(lambda_vals, delta_t))
        fig = go.Figure(data=go.Surface(
            x=lambda_vals, y=delta_t, z=z,
            colorscale="Viridis"))
        fig.update_layout(
            title="Exponential Kernel K(Δt)=exp(-λΔt)",
            scene=dict(xaxis_title="λ", yaxis_title="Δt", zaxis_title="K"))
        return fig

    def plot_powerlaw(self, alpha_vals, delta_t):
        z = (1 + np.outer(alpha_vals, delta_t))**(-1)
        fig = go.Figure(data=go.Surface(
            x=alpha_vals, y=delta_t, z=z,
            colorscale="Cividis"))
        fig.update_layout(
            title="Power-Law Kernel K(Δt)=(1+Δt)^(-α)",
            scene=dict(xaxis_title="α", yaxis_title="Δt", zaxis_title="K"))
        return fig
