import numpy as np
import matplotlib.pyplot as plt
from .utils import ensure_array

class MemoryMassPlot:
    def __init__(self, figsize=(8,4)):
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def plot(self, times, M_matrix, state_labels=None):
        times, M_matrix = ensure_array(times), ensure_array(M_matrix)
        N = M_matrix.shape[1]
        labels = state_labels or [f"State {i}" for i in range(N)]
        self.ax.clear()                                  # clear old plot
        for j in range(N):
            self.ax.plot(times, M_matrix[:,j], label=labels[j])
        self.ax.set_title("Memory Mass M_j(t)")
        self.ax.set_xlabel("Time t")
        self.ax.set_ylabel("M_j(t)")
        self.ax.legend()
        plt.tight_layout()
        return self.fig, self.ax

    def annotate_entrainment_loops(self, bands, color='grey', alpha=0.3):
        """
        Shade each (start, end) interval in `bands` to mark ℰ‐loops.
        """
        for start, end in bands:
            self.ax.axvspan(start, end, color=color, alpha=alpha)
        return self.fig, self.ax
