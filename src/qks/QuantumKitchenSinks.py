import numpy as np

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.providers.aer import AerSimulator


class QuantumKitchenSinks:
    """Quantum Kitchen Sinks algorithm
    
    Qiskit implementation of the Quantum Kitchen Sinks algorithm by Wilson et al. (2019).

    Deviations from the original algorithm:
    1) Random `omega` is set to be same for all QKS parameters within an episode;
    2) For the `shots` version, the most frequent measurement outcome is used.
    """
    def __init__(
        self,
        n_features,
        fm,
        n_episodes=10,
        stddev=2.0,
        seed=None,
        sampling="normal",
        backend=None,
    ):
        """ """
        self.fm = fm
        self.n_features = n_features
        self.n_episodes = n_episodes
        self.stddev = stddev
        self.seed = seed
        self.sampling = sampling
        self.backend = backend
        if self.backend is None:
            algorithm_globals.random_seed = self.seed
            self.backend = QuantumInstance(
                AerSimulator(method="statevector"),
                seed_simulator=self.seed,
                seed_transpiler=self.seed,
            )
        # ---------------------------------------------------
        self.split = int(self.n_features / self.fm.num_parameters)
        if self.split == 0:
            assert (
                False
            ), "Inconsistent intput: split = 0! Check the number of parameters!"
        # ---------------------------------------------------
        # Sampling parameters must be initialized only once!
        # ---------------------------------------------------
        self._generate_omega_and_beta()
        # print(self.omega)

    def _measurement(self, theta):
        """Measure the embedding circuit to produce the QKS feature map"""
        fm = self.fm.bind_parameters(theta)
        fm.measure_all()
        # print(fm.draw())
        result = self.backend.execute(fm)
        counts = result.get_counts(fm)
        a = max(counts, key=counts.get)
        # print('int=', counts.int_outcomes())
        result = np.array(list(a), dtype=np.float64)
        return result

    def _generate_omega_and_beta(self):
        """Setup the QKS random `omega` and `beta` parameters.
        `split` allows for reducing the dataset dimensionality by mixing different features 
        into a single randomized quantity. Random `omega` is set to be equal for all parameters 
        within an episode.
        """
        mask = np.zeros((self.n_episodes, self.n_features, self.fm.num_parameters))
        for i in range(self.fm.num_parameters):
            j0 = i * self.split
            j1 = min((i + 1) * self.split, self.n_features)
            mask[:, j0:j1, i] = 1.0
        # print(mask)
        np.random.seed(self.seed)
        if self.sampling == "normal":
            omega = np.random.normal(
                0.0, self.stddev, (self.n_episodes, self.n_features)
            )
        elif self.sampling == "uniform":
            omega = np.random.uniform(
                -self.stddev, self.stddev, (self.n_episodes, self.n_features)
            )
        else:
            assert False, "Unknown sampling type!"
        self.omega = np.zeros(
            (self.n_episodes, self.n_features, self.fm.num_parameters)
        )
        for i in range(self.fm.num_parameters):
            self.omega[:, :, i] = omega
        self.omega = mask * self.omega
        self.beta = np.random.uniform(
            0.0, 2 * np.pi, (self.n_episodes, self.fm.num_parameters)
        )
        return

    def embedding(self, X):
        """Perform the quantum feature map transformation (data emdedding)."""
        params = X.dot(self.omega) + self.beta
        # print('params: ', params.shape)
        embedding = np.zeros((X.shape[0], self.fm.num_qubits * self.n_episodes))
        for i, param in enumerate(params):
            for j, theta in enumerate(param):
                meas = self._measurement(theta)
                for k in range(self.fm.num_qubits):
                    embedding[i, j + k * self.n_episodes] = meas[k]
        return embedding
