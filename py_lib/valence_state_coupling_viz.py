import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class ValenceCouplingHeatmap:
    """
    Heatmap of joint P_transition vs. v_t for each state.
    """
    def __init__(self, figsize=(6,5)):
        self.fig, self.ax = plt.subplots(figsize=figsize)

    def plot(self, df, x="valence", y="state", weight="prob"):
        """
        df: DataFrame with columns [x, y, weight]
        """
        pivot = pd.pivot_table(df, index=y, columns=x, values=weight, aggfunc="mean")
        sns.heatmap(pivot, ax=self.ax, cmap="rocket", annot=True, fmt=".2f")
        self.ax.set_title("Valence-State Transition Coupling")
        self.ax.set_xlabel("Valence v_t")
        self.ax.set_ylabel("State j")
        plt.tight_layout()
        return self.fig, self.ax
