import asyncio
import numpy as np
from .memory_hmm_viz import MemoryMassPlot
from .kernel_dynamics_viz import KernelDynamicsPlot
from .valence_state_coupling_viz import ValenceCouplingHeatmap

class RealTimeRitualSimulator:
    """
    Streams HMM steps with valence inputs, updates memory mass, and refreshes plots in real time.
    """
    def __init__(self, A0, B, kernel_func, beta, state_labels=None):
        self.A0, self.B = A0, B
        self.kernel = kernel_func
        self.beta = beta
        self.labels = state_labels
        self.times, self.states, self.valences = [], [], []
        self.M = []
        # Visual components
        self.mem_plot = MemoryMassPlot()
        self.heatmap = ValenceCouplingHeatmap()

    def compute_M(self):
        t = len(self.states) - 1
        M_t = []
        for j in range(self.A0.shape[0]):
            mem = sum(
                self.valences[k] *
                int(self.states[k]==j) *
                self.kernel(t-k)
                for k in range(t+1)
            )
            M_t.append(mem)
        return np.array(M_t)

    async def step(self, next_valence, delay=0.5):
        self.times.append(len(self.times))
        self.valences.append(next_valence)
        # sample next state
        t = len(self.times) - 1
        M_t = self.compute_M()
        self.M.append(M_t)
        A_t = (self.A0 + self.beta*M_t) / np.sum(self.A0 + self.beta*M_t)
        next_state = np.random.choice(len(A_t), p=A_t)
        self.states.append(next_state)
        # update visuals
        self.mem_plot.plot(self.times, np.vstack(self.M), self.labels)
        df = {
            "valence": self.valences,
            "state": self.states,
            "prob": [A_t[s] for s in self.states]
        }
        self.heatmap.plot(pd.DataFrame(df))
        await asyncio.sleep(delay)

    def run(self, valence_stream, delay=0.5):
        """
        valence_stream: iterable of valence inputs
        """
        loop = asyncio.get_event_loop()
        tasks = [self.step(v, delay) for v in valence_stream]
        loop.run_until_complete(asyncio.gather(*tasks))

## Example

from rcft.code.visualization.real_time_simulator import RealTimeRitualSimulator
import numpy as np

# Base HMM parameters
A0 = np.array([[0.7,0.3],[0.4,0.6]])
B  = None  # not used for visualization prototype
kernel = lambda dt: np.exp(-0.1*dt)
beta = 0.5
labels = ["Calm","Excited"]

sim = RealTimeRitualSimulator(A0, B, kernel, beta, labels)
# Stream of valence scores from a ritual or live sensor
valence_stream = np.random.normal(loc=0.0, scale=1.0, size=50)
sim.run(valence_stream, delay=0.2)
