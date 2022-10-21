import sys
import numpy as np

from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import Aer
from qiskit_machine_learning.neural_networks import OpflowQNN
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow import AerPauliExpectation, StateFn, PauliSumOp, ListOp

class QuantumKitchenSinks:
    
    def __init__(self, n_features, ansatz, episodes=10, stddev=2.0, seed=None, sampling='normal', backend=None):
        """ """
        self.ansatz = ansatz
        self.n_features = n_features
        self.n_episodes = episodes
        self.stddev = stddev
        self.seed = seed
        self.sampling = sampling
        self.backend = backend
        if self.backend is None:
            algorithm_globals.random_seed = self.seed
            self.backend = QuantumInstance(
                Aer.get_backend("statevector_simulator"), 
                seed_simulator=seed, seed_transpiler=seed,
            )
        # ---------------------------------------------------
        self.split = int(self.n_features/self.ansatz.num_parameters)
        if self.split == 0:
            assert False, "Inconsistent intput: split = 0! Check the number of parameters!"
        # ---------------------------------------------------
        # Sampling parameters must be initialized only once!
        # ---------------------------------------------------
        self._generate_omega_and_beta()
        # print(self.omega)
    
    def _measurement(self, theta):
        """ Measure the embedding circuit to produce the QKS feature map """
        fm = self.ansatz.bind_parameters(theta)
        fm.measure_all()
        # print(fm.draw())
        result = self.backend.execute(fm)
        counts = result.get_counts(fm)
        a = max(counts, key=counts.get)
        # print('int=', counts.int_outcomes())
        result = np.array(list(a), dtype=np.float64)
        return result
    
    def _generate_omega_and_beta(self):
        """ Generate QKS parameters """
        mask = np.zeros((self.n_episodes, self.n_features, self.ansatz.num_parameters))
        for i in range(self.ansatz.num_parameters):
            j0 = i*self.split
            j1 = min((i+1)*self.split, self.n_features)
            mask[:, j0:j1, i] = 1.0
        # print(mask)
        np.random.seed(self.seed)
        if self.sampling == 'normal':
            omega = np.random.normal(0.0, self.stddev, (self.n_episodes, self.n_features))
        elif self.sampling == 'uniform':
            omega = np.random.uniform(-self.stddev, self.stddev, (self.n_episodes, self.n_features))
        else:
            assert False, "Unknown sampling type!"
        self.omega = np.zeros((self.n_episodes, self.n_features, self.ansatz.num_parameters))
        for i in range(self.ansatz.num_parameters):
            self.omega[:, :, i] = omega
        self.omega = mask * self.omega
        self.beta  = np.random.uniform(0.0, 2*np.pi, (self.n_episodes, self.ansatz.num_parameters))
        return
    
    def embedding(self, X):
        """ Compute embedding """
        params = X.dot(self.omega) + self.beta
        # print('params: ', params.shape)
        embedding = np.zeros((X.shape[0], self.ansatz.num_qubits*self.n_episodes))
        for i, param in enumerate(params):
            for j, theta in enumerate(param):
                meas = self._measurement(theta)
                for k in range(self.ansatz.num_qubits):
                    embedding[i, j + k*self.n_episodes] = meas[k]
        return embedding
    


class ProjectedQuantumKitchenSinks:
    
    def __init__(self, n_features, fm, backend=None, episodes=10, stddev=2.0, seed=None, sampling='normal', projection='z', method='qnn'):
        """ """
        self.fm = fm
        self.n_features = n_features
        self.n_params = self.fm.num_parameters
        self.n_episodes = episodes
        self.stddev = stddev
        self.sampling = sampling
        self.projection = projection
        self.method = method
        self.seed = seed
        self.backend = backend
        if self.backend is None:
            backend_options = {
                'max_parallel_threads': 0,
                'max_parallel_experiments': 1,
                'max_memory_mb': 0,
                'statevector_sample_measure_opt': 10,
            }
            algorithm_globals.random_seed = self.seed
            self.backend = QuantumInstance(
                Aer.get_backend("aer_simulator_statevector"),
                # Aer.get_backend("aer_simulator"),
                shots=1024, backend_options=backend_options,
                seed_simulator=seed, seed_transpiler=seed,
            )
        
        self.split = int(self.n_features/self.n_params)
        if self.split == 0:
            assert False, "Inconsistent intput: split = 0! Check the number of parameters!"
        
        # Sampling parameters must be initialized only once!
        self._get_omega_and_beta()
        # print(self.omega)
        
        # generate projection operators
        if isinstance(self.projection, str):
            if self.projection=='x':
                self.proj_ops = self._measurement_operator(['X'])
            elif self.projection=='y':
                self.proj_ops = self._measurement_operator(['Y'])
            elif self.projection=='z':
                self.proj_ops = self._measurement_operator(['Z'])
            elif self.projection=='xyz':
                self.proj_ops = self._measurement_operator(['X', 'Y', 'Z'])
            elif self.projection=='xyz_sum':
                self.proj_ops = []
                for x, y, z in zip(self._measurement_operator(['X']), self._measurement_operator(['Y']), self._measurement_operator(['Z'])):
                    self.proj_ops.append([x[0], y[0], z[0]])
            else:
                self.proj_ops = self.projection
        
    def _get_omega_and_beta(self):
        mask = np.zeros((self.n_episodes, self.n_features, self.n_params))
        for i in range(self.n_params):
            j0 = i*self.split
            j1 = min((i+1)*self.split, self.n_features)
            mask[:, j0:j1, i] = 1.0
        # print(mask)
        np.random.seed(self.seed)
        if self.sampling == 'normal':
            omega = np.random.normal(0.0, self.stddev, (self.n_episodes, self.n_features))
        elif self.sampling == 'uniform':
            omega = np.random.uniform(-self.stddev, self.stddev, (self.n_episodes, self.n_features))
        else:
            assert False, "Unknown sampling type!"
        self.omega = np.zeros((self.n_episodes, self.n_features, self.n_params))
        for i in range(self.n_params):
            self.omega[:, :, i] = omega
        self.omega = mask * self.omega
        self.beta  = np.random.uniform(0.0, 2*np.pi, (self.n_episodes, self.n_params))
        return

    def _measurement_operator(self, gates):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            for gate in gates:
                ss = list(s)
                ss[-i-1] = gate
                op.append([(''.join(ss), 1.0)])
        return op
            
    def projected_feature_map(self, x):
        """ """
        if self.method=='statevector':
            qc = self.fm.assign_parameters(x)
            result = self.backend.execute(qc)
            sv = result.get_statevector()
            ev = []
            for op in self.proj_ops:
                o = Operator(Pauli(op[0][0]))
                if len(op)>1:
                    for i in range(1, len(op)):
                        o = o + Operator(Pauli(op[i][0]))
                ev.append(np.real(sv.expectation_value(o)))
        elif self.method=='qnn':
            fm_sfn = StateFn(self.fm)
            list_ops = []
            for op in self.proj_ops:
                op = ~StateFn(PauliSumOp.from_list(op)) @ fm_sfn
                list_ops.append(op)
            list_ops = ListOp(list_ops)
            expval = AerPauliExpectation()
            qnn = OpflowQNN(list_ops, input_params=self.fm.parameters, weight_params=[], exp_val=expval, gradient=None, quantum_instance=self.backend)
            ev = qnn.forward(x, []).flatten()
        else:
            assert False, 'Unknown method!'
        return ev
            
    def embedding(self, X):
        """ Compute embedding """
        params = X.dot(self.omega) + self.beta
        # print('params: ', params.shape)
        N = len(self.proj_ops) 
        embedding = np.zeros((X.shape[0], N*self.n_episodes))
        for i, param in enumerate(params):
            for j, theta in enumerate(param):
                meas = self.projected_feature_map(theta)
                for k in range(N):
                    embedding[i, j+k*self.n_episodes] = meas[k]
        return embedding
