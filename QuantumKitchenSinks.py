import sys
import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import Aer
from qiskit.quantum_info.operators import Operator, Pauli

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
    

class ProjectedQuantumKitchenSinksSV:
    
    def __init__(self, n_features, fm, backend=None, episodes=10, stddev=2.0, seed=None, sampling='normal', projection='z'):
        """ """
        self.fm = fm
        self.n_features = n_features
        self.n_params = self.fm.num_parameters
        self.n_episodes = episodes
        self.stddev = stddev
        self.sampling = sampling
        self.projection = projection
        self.seed = seed
        self.backend = backend
        if self.backend is None:
            backend_options = {
                'max_parallel_threads': 8,
                'max_parallel_experiments': 1,
                'max_memory_mb': 0,
                'statevector_sample_measure_opt': 10,
            }
            algorithm_globals.random_seed = self.seed
            self.backend = QuantumInstance(
                Aer.get_backend("statevector_simulator"), 
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
                self.proj_ops = self._o_x()
            elif self.projection=='y':
                self.proj_ops = self._o_y()
            elif self.projection=='z':
                self.proj_ops = self._o_z()
            elif self.projection=='xyz':
                self.proj_ops = self._o_xyz()
            elif self.projection=='xyz_sum':
                self.proj_ops = []
                for x, y, z in zip(self._o_x(), self._o_y(), self._o_z()):
                    self.proj_ops.append(x+y+z) 
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

    def _o_x(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            for gate in ['X']:
                ss = list(s)
                ss[-i-1] = gate
                op.append(Operator(Pauli(''.join(ss))))
        return op
    
    def _o_y(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            for gate in ['Y']:
                ss = list(s)
                ss[-i-1] = gate
                op.append(Operator(Pauli(''.join(ss))))
        return op
    
    def _o_z(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            for gate in ['Z']:
                ss = list(s)
                ss[-i-1] = gate
                op.append(Operator(Pauli(''.join(ss))))
        return op
    
    def _o_xyz(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            for gate in ['X', 'Y', 'Z']:
                ss = list(s)
                ss[-i-1] = gate
                op.append(Operator(Pauli(''.join(ss))))
        return op
    
    def projected_feature_map(self, x):
        """ """
        qc = QuantumCircuit(self.fm.num_qubits)

        x_dict = dict(zip(self.fm.parameters, x))
        psi_x = self.fm.assign_parameters(x_dict)
        qc.append(psi_x.to_instruction(), qc.qubits)
        result = self.backend.execute(qc)
        sv = result.get_statevector(qc, decimals=4)
    
        ev = []
        for op in self.proj_ops:
            ev.append(np.real(sv.expectation_value(op)))
            
        del result, sv 
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

#################################################################################################
#  Slower CircuitSampler version that was inspired by the Qiskit QNN titorial
#################################################################################################
from qiskit.opflow import CircuitSampler, AerPauliExpectation, StateFn, PauliSumOp, ListOp

class ProjectedQuantumKitchenSinks:
    
    def __init__(self, n_features, fm, backend=None, episodes=10, stddev=2.0, seed=None, sampling='normal', projection='z'):
        """ """
        self.fm = fm
        self.n_features = n_features
        self.n_params = self.fm.num_parameters
        self.n_episodes = episodes
        self.stddev = stddev
        self.sampling = sampling
        self.projection = projection
        self.seed = seed
        self.backend = backend
        if self.backend is None:
            algorithm_globals.random_seed = self.seed
            self.backend = QuantumInstance(
                Aer.get_backend("statevector_simulator"), 
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
                self.proj_ops = self._o_x()
            elif self.projection=='y':
                self.proj_ops = self._o_y()
            elif self.projection=='z':
                self.proj_ops = self._o_z()
            elif self.projection=='xyz':
                self.proj_ops = self._o_xyz()
            elif self.projection=='xyz_sum':
                self.proj_ops = self._o_xyz_sum()
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

    def _o_x(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            op_s = []
            for gate in ['X']:
                ss = list(s)
                ss[-i-1] = gate
                op_s.append((''.join(ss), 1.0))
            op.append(op_s)
        return op
    
    def _o_y(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            op_s = []
            for gate in ['Y']:
                ss = list(s)
                ss[-i-1] = gate
                op_s.append((''.join(ss), 1.0))
            op.append(op_s)
        return op
    
    def _o_z(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            op_s = []
            for gate in ['Z']:
                ss = list(s)
                ss[-i-1] = gate
                op_s.append((''.join(ss), 1.0))
            op.append(op_s)
        return op
    
    def _o_xyz_sum(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            op_s = []
            for gate in ['X', 'Y', 'Z']:
                ss = list(s)
                ss[-i-1] = gate
                op_s.append((''.join(ss), 1.0))
            op.append(op_s)
        return op
    
    def _o_xyz(self):
        s = 'I'*self.fm.num_qubits
        op = []
        for i in range(self.fm.num_qubits):
            for gate in ['X', 'Y', 'Z']:
                ss = list(s)
                ss[-i-1] = gate
                op.append([(''.join(ss), 1.0)])
        return op
    
    def projected_feature_map(self, x):
        """ """
        qc = QuantumCircuit(self.fm.num_qubits)

        x_dict = dict(zip(self.fm.parameters, x))
        psi_x = self.fm.assign_parameters(x_dict)
        qc.append(psi_x.to_instruction(), qc.qubits)
        # print(qc.decompose().draw())
        
        op = []
        for proj_op in self.proj_ops:
            op.append( ~StateFn(PauliSumOp.from_list(proj_op)) @ StateFn(qc) )
        op = ListOp(op)
    
        expectation = AerPauliExpectation().convert(op.reduce())
        sampler = CircuitSampler(self.backend).convert(expectation)
        return sampler.eval()
        
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
    

